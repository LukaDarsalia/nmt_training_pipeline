"""
Main NMT Trainer

Orchestrates the training pipeline using registered components for models,
trainers, evaluators, and tokenizers. Provides a unified interface for training experiments.
"""

from typing import Dict, Any, Optional
from pathlib import Path
import time
import torch
from transformers.data.data_collator import DataCollator
from transformers.generation.configuration_utils import GenerationConfig
from transformers.trainer import Trainer
import wandb
from transformers.trainer_utils import set_seed
from transformers import EncoderDecoderModel
from .registry import model_registry, trainer_registry, evaluator_registry, tokenizer_registry
from .utils.data_utils import load_datasets_from_artifact
from datasets import DatasetDict

class NMTTrainer:
    """
    Main trainer class for neural machine translation experiments.

    Uses registry system to dynamically create models, trainers, evaluators, and tokenizers
    based on YAML configuration files.
    """

    def __init__(self,
                 config: Dict[str, Any],
                 artifact: wandb.Artifact,
                 input_data_dir: str,
                 output_data_dir: str,
                 model_dir: Optional[str] = None):
        """
        Initialize the NMT trainer.

        Args:
            config: Training configuration dictionary
            artifact: WandB artifact for logging
            input_data_dir: Directory containing split datasets
            output_data_dir: Directory to save outputs
            model_dir: Optional directory containing pretrained model
        """
        self.config = config
        self.artifact = artifact
        self.input_data_dir = Path(input_data_dir)
        self.output_data_dir = Path(output_data_dir)
        self.model_dir = Path(model_dir) if model_dir else None

        # Set random seed
        seed = config.get('training', {}).get('seed', 42)
        set_seed(seed)
        torch.manual_seed(seed)

        # Initialize components
        self.device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # Create output directory
        self.output_data_dir.mkdir(parents=True, exist_ok=True)

        # Initialize later
        self.tokenizer: Any = None
        self.tokenizer_impl: Any = None
        self.model: Any = None
        self.generation_config: Optional[GenerationConfig] = None
        self.data_collator: Any = None
        self.trainer: Optional[Trainer] = None
        self.datasets: Optional[Dict[str, Any]] = None

        # Statistics tracking
        self.training_stats = {}

    def setup(self) -> None:
        """Setup all components for training."""
        print("Setting up training pipeline...")

        # Load tokenizer
        self._load_tokenizer()

        # Create model
        self._create_model()

        # Load datasets
        self._load_datasets()

        # Create trainer
        self._create_trainer()

        print("Training pipeline setup complete!")

    def _load_tokenizer(self) -> None:
        """Load the appropriate tokenizer using tokenizer registry."""
        print("Loading tokenizer...")

        try:
            # Get tokenizer configuration
            tokenizer_config = self.config.get('tokenizer', {})
            tokenizer_type = tokenizer_config.get('type', 'auto_tokenizer')
            
            # Validate tokenizer type exists
            tokenizer_registry.validate_component_exists(tokenizer_type, "Tokenizer")
            
            # Get tokenizer implementation
            tokenizer_class = tokenizer_registry.get(tokenizer_type)
            if tokenizer_class is None:
                raise ValueError(f"Tokenizer '{tokenizer_type}' not found in registry")
            
            # Create tokenizer instance
            self.tokenizer_impl = tokenizer_class(tokenizer_config)
            
            # Load the actual tokenizer
            self.tokenizer = self.tokenizer_impl.load_tokenizer()
            
            print(f"Loaded tokenizer: {tokenizer_type}")

        except Exception as e:
            raise RuntimeError(f"Failed to load tokenizer: {e}")

    def _load_datasets(self) -> None:
        """Load and tokenize datasets using tokenizer registry."""
        print("Loading datasets...")

        # Load raw datasets
        train_dataset, valid_dataset, test_dataset = load_datasets_from_artifact(str(self.input_data_dir))

        # Get tokenization configuration
        tokenization_config = self.config.get('data', {})
        
        # Add model-specific configuration to tokenization config
        model_config = self.config.get('model', {})
        if model_config.get('type') in ['encoder_decoder_pretrained', 'encoder_decoder_random', 'encoder_decoder_mixed']:
            tokenization_config['encoder_decoder_preprocessing'] = model_config.get('encoder_decoder_preprocessing', False)

        # Tokenize datasets using tokenizer implementation
        train_tokenized, valid_tokenized, test_tokenized = self.tokenizer_impl.tokenize_datasets(
            (train_dataset, valid_dataset, test_dataset),
            tokenization_config,
            self.model
        )

        self.datasets = {
            'train': train_tokenized,
            'valid': valid_tokenized,
            'test': test_tokenized
        }

        print(f"Input ids: \n{train_tokenized['input_ids'][:2]}\n")
        print(f"Labels: \n{train_tokenized['labels'][:2]}\n")
        print(f"Attention mask: \n{train_tokenized['attention_mask'][:2]}\n")

        print("Datasets loaded and tokenized successfully")

    def _create_model(self) -> None:
        """Create the model using model registry."""
        print("Creating model...")

        model_config = self.config.get('model', {})
        model_type = model_config.get('type')

        if not model_type:
            raise ValueError("Model type must be specified in config.model.type")

        # Add device to model config
        model_config['device'] = str(self.device)

        # Add model directory if loading pretrained model
        if self.model_dir and model_config.get('use_pretrained_path', False):
            model_config['model_path'] = str(self.model_dir / 'model')

        # Create model using registry
        self.model, self.generation_config, self.data_collator = model_registry.create_model(
            model_type, model_config, self.tokenizer
        )

        # Resize token embeddings if needed
        # if hasattr(self.model, 'resize_token_embeddings'):
        #     # Check if this is an EncoderDecoder model
        #     if hasattr(self.model, 'encoder') and hasattr(self.model, 'decoder'):
        #         # For EncoderDecoder models, resize encoder and decoder separately
        #         from src.training.tokenizers.encoder_decoder_tokenizers import EncoderDecoderTokenizer
        #         if isinstance(self.tokenizer, EncoderDecoderTokenizer):
        #             if hasattr(self.model.encoder, 'resize_token_embeddings'):
        #                 self.model.encoder.resize_token_embeddings(len(self.tokenizer.encoder))
        #                 print(f"Resized encoder embeddings to: {len(self.tokenizer.encoder)}")
        #             if hasattr(self.model.decoder, 'resize_token_embeddings'):
        #                 self.model.decoder.resize_token_embeddings(len(self.tokenizer.decoder))
        #                 print(f"Resized decoder embeddings to: {len(self.tokenizer.decoder)}")
        #         else:
        #             # For non-EncoderDecoderTokenizer with EncoderDecoder model, use tokenizer length
        #             if hasattr(self.model.encoder, 'resize_token_embeddings'):
        #                 self.model.encoder.resize_token_embeddings(len(self.tokenizer))
        #             if hasattr(self.model.decoder, 'resize_token_embeddings'):
        #                 self.model.decoder.resize_token_embeddings(len(self.tokenizer))

        # Log model information
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        print(f"Model created successfully:")
        print(f"  Type: {model_type}")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")

        # Log to artifact metadata
        self.artifact.metadata.update({
            'model_type': model_type,
            'total_parameters': f"{total_params:,}",
            'trainable_parameters': f"{trainable_params:,}",
            'tokenizer_type': self.config.get('tokenizer', {}).get('type', 'auto_tokenizer')
        })

    def _create_trainer(self) -> None:
        """Create the trainer using trainer registry."""
        print("Creating trainer...")

        if not self.datasets:
            raise RuntimeError("Datasets not initialized. Call setup() first.")

        trainer_config = self.config.get('trainer', {})
        trainer_type = trainer_config.get('type', 'standard_seq2seq')
        print(trainer_config)
        # Add output directory to config
        trainer_config['output_dir'] = str(self.output_data_dir)

        trainer_config['generation_config'] = self.generation_config
        # Create trainer using registry
        self.trainer = trainer_registry.create_trainer(
            trainer_type,
            trainer_config,
            self.model,
            self.tokenizer,
            self.datasets['train'],
            self.datasets['valid'],
            self.data_collator
        )

        print(f"Trainer created successfully: {trainer_type}")

    def train(self) -> None:
        """Run the training process."""
        print("Starting training...")

        if not self.trainer:
            raise RuntimeError("Trainer not initialized. Call setup() first.")
        
        if not self.datasets:
            raise RuntimeError("Datasets not initialized. Call setup() first.")

        # Record start time
        start_time = time.time()

        try:
            # Train the model
            train_result = self.trainer.train()

            # Record training time
            training_time = time.time() - start_time

            # Log training statistics
            self.training_stats = {
                'training_time_seconds': training_time,
                'training_time_formatted': f"{training_time // 3600:.0f}h {(training_time % 3600) // 60:.0f}m {training_time % 60:.0f}s",
                'total_steps': train_result.global_step,
                'epochs_completed': train_result.global_step / len(self.datasets['train']),
                'final_train_loss': train_result.training_loss
            }

            print(f"Training completed in {self.training_stats['training_time_formatted']}")
            print(f"Total steps: {self.training_stats['total_steps']}")
            print(f"Final training loss: {self.training_stats['final_train_loss']:.4f}")

        except Exception as e:
            print(f"Training failed: {e}")
            raise

    def evaluate(self) -> Dict[str, Any]:
        """Evaluate the model on validation set."""
        print("Evaluating model...")

        if not self.datasets:
            raise RuntimeError("Datasets not initialized. Call setup() first.")

        if not self.trainer:
            raise RuntimeError("Trainer not initialized. Call setup() first.")

        try:
            # Evaluate on validation set
            eval_results = self.trainer.evaluate(eval_dataset=self.datasets['valid'])

            print("Evaluation completed")
            print("Validation metrics:")
            for key, value in eval_results.items():
                if isinstance(value, (int, float)):
                    print(f"  {key}: {value:.4f}")

            return eval_results

        except Exception as e:
            print(f"Evaluation failed: {e}")
            raise

    def predict(self, dataset_name: str = 'valid') -> Dict[str, Any]:
        """Generate predictions on specified dataset."""
        print(f"Generating predictions on {dataset_name} set...")

        if not self.trainer:
            raise RuntimeError("Trainer not initialized. Call setup() first.")
        
        if not self.datasets:
            raise RuntimeError("Datasets not initialized. Call setup() first.")

        if dataset_name not in self.datasets:
            raise ValueError(f"Dataset '{dataset_name}' not found. Available: {list(self.datasets.keys())}")

        try:
            predictions = self.trainer.predict(
                self.datasets[dataset_name],
                metric_key_prefix=f"{dataset_name}"
            )

            print(f"Prediction completed on {dataset_name} set")

            return {
                'predictions': predictions.predictions,
                'label_ids': predictions.label_ids,
                'metrics': predictions.metrics
            }

        except Exception as e:
            print(f"Prediction failed: {e}")
            raise

    def save_model(self) -> None:
        """Save the trained model and tokenizer."""
        print("Saving model and tokenizer...")

        if not self.model:
            raise RuntimeError("Model not initialized. Call setup() first.")

        try:
            # Create model directory
            model_dir = self.output_data_dir / "model"
            model_dir.mkdir(parents=True, exist_ok=True)

            # Save model and tokenizer
            self.model.save_pretrained(model_dir)
            self.tokenizer.save_pretrained(model_dir)

            print(f"Model and tokenizer saved to {model_dir}")

            # Update artifact metadata
            self.artifact.metadata.update({
                'model_saved': True,
                'model_path': str(model_dir)
            })

        except Exception as e:
            print(f"Failed to save model: {e}")
            raise

    def run_full_training(self) -> Dict[str, Any]:
        """Run the complete training pipeline."""
        print("=" * 60)
        print("STARTING FULL TRAINING PIPELINE")
        print("=" * 60)

        # Setup
        self.setup()

        # Train
        self.train()

        # Save model
        self.save_model()

        # Evaluate
        eval_results = self.evaluate()

        # Generate predictions on test set
        test_predictions = self.predict('test')

        # Log final statistics
        final_stats = {
            **self.training_stats,
            'evaluation_results': eval_results,
            'test_predictions': test_predictions,
            'test_predictions_generated': True
        }

        self.artifact.metadata.update(final_stats)

        print("=" * 60)
        print("TRAINING PIPELINE COMPLETED SUCCESSFULLY")
        print("=" * 60)

        return final_stats
