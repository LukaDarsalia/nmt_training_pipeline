"""
Main NMT Trainer

Orchestrates the training pipeline using registered components for models,
trainers, and evaluators. Provides a unified interface for training experiments.
"""

from typing import Dict, Any, Optional
from pathlib import Path
import time
import torch
import wandb
from transformers import AutoTokenizer, set_seed

from .registry import model_registry, trainer_registry, evaluator_registry
from .utils.data_utils import load_datasets_from_artifact, tokenize_datasets, prepare_encoder_decoder_tokenization


class NMTTrainer:
    """
    Main trainer class for neural machine translation experiments.

    Uses registry system to dynamically create models, trainers, and evaluators
    based on YAML configuration files.
    """

    def __init__(self,
                 config: Dict[str, Any],
                 artifact: wandb.Artifact,
                 input_data_dir: str,
                 tokenizer_dir: str,
                 output_data_dir: str,
                 model_dir: Optional[str] = None):
        """
        Initialize the NMT trainer.

        Args:
            config: Training configuration dictionary
            artifact: WandB artifact for logging
            input_data_dir: Directory containing split datasets
            tokenizer_dir: Directory containing tokenizer
            output_data_dir: Directory to save outputs
            model_dir: Optional directory containing pretrained model
        """
        self.config = config
        self.artifact = artifact
        self.input_data_dir = Path(input_data_dir)
        self.tokenizer_dir = Path(tokenizer_dir)
        self.output_data_dir = Path(output_data_dir)
        self.model_dir = Path(model_dir) if model_dir else None

        # Set random seed
        seed = config.get('training', {}).get('seed', 42)
        set_seed(seed)
        torch.manual_seed(seed)

        # Initialize components
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # Create output directory
        self.output_data_dir.mkdir(parents=True, exist_ok=True)

        # Initialize later
        self.tokenizer = None
        self.model = None
        self.generation_config = None
        self.data_collator = None
        self.trainer = None
        self.datasets = None

        # Statistics tracking
        self.training_stats = {}

    def setup(self) -> None:
        """Setup all components for training."""
        print("Setting up training pipeline...")

        # Load tokenizer
        self._load_tokenizer()

        # Load datasets
        self._load_datasets()

        # Create model
        self._create_model()

        # Create trainer
        self._create_trainer()

        print("Training pipeline setup complete!")

    def _load_tokenizer(self) -> None:
        """Load the tokenizer."""
        print("Loading tokenizer...")

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_dir)
            self.tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True
            print(f"Loaded tokenizer from {self.tokenizer_dir}")
            print(f"Vocab size: {self.tokenizer.vocab_size}")
        except Exception as e:
            raise RuntimeError(f"Failed to load tokenizer from {self.tokenizer_dir}: {e}")

    def _load_datasets(self) -> None:
        """Load and tokenize datasets."""
        print("Loading datasets...")

        # Load raw datasets
        train_dataset, valid_dataset, test_dataset = load_datasets_from_artifact(str(self.input_data_dir))

        # Get tokenization configuration
        tokenization_config = self.config.get('data', {})
        model_config = self.config.get('model', {})

        # Choose tokenization method based on model type
        if model_config.get('type') in ['encoder_decoder_pretrained', 'encoder_decoder_random',
                                        'encoder_decoder_mixed']:
            # Use encoder-decoder specific tokenization
            tokenization_config['encoder_decoder_preprocessing'] = model_config.get('encoder_decoder_preprocessing',
                                                                                    False)

            train_tokenized, valid_tokenized, test_tokenized = prepare_encoder_decoder_tokenization(
                (train_dataset, valid_dataset, test_dataset),
                self.tokenizer,
                tokenization_config
            )
        else:
            # Use standard seq2seq tokenization
            train_tokenized, valid_tokenized, test_tokenized = tokenize_datasets(
                (train_dataset, valid_dataset, test_dataset),
                self.tokenizer,
                tokenization_config
            )

        self.datasets = {
            'train': train_tokenized,
            'valid': valid_tokenized,
            'test': test_tokenized
        }

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
        if hasattr(self.model, 'resize_token_embeddings'):
            self.model.resize_token_embeddings(len(self.tokenizer))

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
            'trainable_parameters': f"{trainable_params:,}"
        })

    def _create_trainer(self) -> None:
        """Create the trainer using trainer registry."""
        print("Creating trainer...")

        trainer_config = self.config.get('trainer', {})
        trainer_type = trainer_config.get('type', 'standard_seq2seq')

        # Add output directory to config
        trainer_config['output_dir'] = str(self.output_data_dir)

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

        if not self.trainer:
            raise RuntimeError("Trainer not initialized. Call setup() first.")

        try:
            # Evaluate on validation set
            eval_results = self.trainer.evaluate()

            print("Evaluation completed")
            print("Validation metrics:")
            for key, value in eval_results.items():
                if isinstance(value, (int, float)):
                    print(f"  {key}: {value:.4f}")

            return eval_results

        except Exception as e:
            print(f"Evaluation failed: {e}")
            raise

    def predict(self, dataset_name: str = 'test') -> Dict[str, Any]:
        """Generate predictions on specified dataset."""
        print(f"Generating predictions on {dataset_name} set...")

        if not self.trainer:
            raise RuntimeError("Trainer not initialized. Call setup() first.")

        if dataset_name not in self.datasets:
            raise ValueError(f"Dataset '{dataset_name}' not found. Available: {list(self.datasets.keys())}")

        try:
            # Generate predictions
            generation_config = self.config.get('generation', {})

            predictions = self.trainer.predict(
                self.datasets[dataset_name],
                max_length=generation_config.get('max_length', 128),
                num_beams=generation_config.get('num_beams', 5),
                metric_key_prefix=f"final_{dataset_name}"
            )

            print(f"Prediction completed on {dataset_name} set")

            return predictions

        except Exception as e:
            print(f"Prediction failed: {e}")
            raise

    def save_model(self) -> None:
        """Save the trained model."""
        print("Saving model...")

        if not self.model:
            raise RuntimeError("Model not initialized. Call setup() first.")

        try:
            # Create model directory
            model_dir = self.output_data_dir / "model"
            model_dir.mkdir(parents=True, exist_ok=True)

            # Save model and tokenizer
            self.model.save_pretrained(model_dir)
            self.tokenizer.save_pretrained(model_dir)

            print(f"Model saved to {model_dir}")

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
            'test_predictions_generated': True
        }

        self.artifact.metadata.update(final_stats)

        print("=" * 60)
        print("TRAINING PIPELINE COMPLETED SUCCESSFULLY")
        print("=" * 60)

        return final_stats