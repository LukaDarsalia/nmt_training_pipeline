"""
Training Callbacks

Provides various callbacks for training including early stopping,
prediction logging, and other monitoring utilities.
"""

from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import wandb
import torch
from transformers import (
    TrainerCallback,
    EarlyStoppingCallback,
    PreTrainedTokenizer
)
from transformers.integrations import WandbCallback


class WandbEvaluationCallback(WandbCallback):
    """
    Enhanced WandbCallback to log comprehensive evaluation results during training.

    This callback logs model predictions, source texts, references, and individual
    metric scores to wandb.Table at each evaluation step, providing detailed
    insights into model performance.
    """

    def __init__(self,
                 tokenizer: PreTrainedTokenizer,
                 eval_dataset: Any,
                 compute_metrics_fn: Optional[Any] = None,
                 num_samples: int = 20,
                 log_frequency: int = 2,
                 max_length: int = 128,
                 num_beams: int = 5,
                 source_column: str = 'en',
                 target_column: str = 'ka'):
        """
        Initialize the evaluation callback.

        Args:
            tokenizer: Tokenizer for decoding predictions
            eval_dataset: Evaluation dataset to sample from
            compute_metrics_fn: Optional metrics computation function
            num_samples: Number of samples to log
            log_frequency: Log every N evaluation steps
            max_length: Maximum generation length
            num_beams: Number of beams for generation
            source_column: Column name for source text
            target_column: Column name for target text
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.eval_dataset = eval_dataset
        self.compute_metrics_fn = compute_metrics_fn
        self.num_samples = min(num_samples, len(eval_dataset))
        self.log_frequency = log_frequency
        self.max_length = max_length
        self.num_beams = num_beams
        self.source_column = source_column
        self.target_column = target_column
        self.counter = 0
        self.trainer = None  # Will be set in on_train_begin

        # Sample subset of evaluation dataset
        if hasattr(eval_dataset, 'select'):
            self.sample_dataset = eval_dataset.select(range(self.num_samples))
        else:
            # If dataset doesn't have select method, take first N samples
            self.sample_dataset = eval_dataset[:self.num_samples]

    def on_train_begin(self, args, state, control, **kwargs):
        """Called at the beginning of training to store trainer reference."""
        super().on_train_begin(args, state, control, **kwargs)
        
        print(f"🔍 WandbEvaluationCallback.on_train_begin called")
        print(f"🔍 Available kwargs: {list(kwargs.keys())}")
        
        # Store trainer reference
        trainer = kwargs.get('trainer')
        if trainer is not None:
            self.trainer = trainer
            print(f"✅ WandbEvaluationCallback: Stored trainer reference")
        else:
            print(f"⚠️ WandbEvaluationCallback: No trainer found in on_train_begin kwargs")
            # Try to access trainer through other means if possible
            for key, value in kwargs.items():
                if hasattr(value, 'predict') and hasattr(value, 'model'):
                    print(f"🔍 Found potential trainer object in kwargs['{key}']")
                    self.trainer = value
                    break

    def on_evaluate(self, args, state, control, **kwargs):
        """Called after each evaluation step."""
        super().on_evaluate(args, state, control, **kwargs)

        self.counter += 1
        print(f"🔍 WandbEvaluationCallback: Evaluation step {self.counter}, frequency={self.log_frequency}")

        if self.counter % self.log_frequency == 0:
            # Try to get trainer from kwargs first, then from stored reference
            trainer = kwargs.get('trainer', self.trainer)
            
            if trainer is None:
                print("❌ No trainer found in kwargs or stored reference")
                print(f"🔍 Available kwargs keys: {list(kwargs.keys())}")
                return
            
            # Verify trainer has required methods
            if not hasattr(trainer, 'predict'):
                print(f"❌ Trainer object doesn't have 'predict' method: {type(trainer)}")
                return
            
            print(f"✅ Using trainer: {type(trainer)}")

            try:
                print(f"🚀 Starting evaluation result logging for step {state.global_step}")
                print(f"📊 Sample dataset size: {len(self.sample_dataset)}")
                
                # Generate predictions (disable automatic wandb logging to prevent test/ metrics)
                print("🤖 Generating predictions...")
                
                # Temporarily disable wandb logging for the predict call
                import os
                original_wandb_mode = os.environ.get('WANDB_MODE', '')
                os.environ['WANDB_MODE'] = 'disabled'
                
                try:
                    predictions_output = trainer.predict(self.sample_dataset, metric_key_prefix="sample_valid")
                finally:
                    # Restore original wandb mode
                    if original_wandb_mode:
                        os.environ['WANDB_MODE'] = original_wandb_mode
                    elif 'WANDB_MODE' in os.environ:
                        del os.environ['WANDB_MODE']
                print(f"✅ Predictions generated: {type(predictions_output)}")

                # Extract data for logging
                self._log_evaluation_results(predictions_output, state.global_step)
                print(f"✅ Evaluation results logged for step {state.global_step}")

            except Exception as e:
                print(f"❌ Failed to log evaluation results: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"⏭️ Skipping logging for step {self.counter} (frequency={self.log_frequency})")

    def _log_evaluation_results(self, predictions_output, global_step: int):
        """Log detailed evaluation results to wandb."""
        print(f"📝 Starting detailed evaluation logging for step {global_step}")
        
        # Prepare data for logging
        try:
            input_ids = self.sample_dataset["input_ids"]
            labels = self.sample_dataset["labels"]
            pred_ids = predictions_output.predictions
            print(f"📋 Data shapes - input_ids: {len(input_ids)}, labels: {len(labels)}, pred_ids: {len(pred_ids) if pred_ids is not None else 'None'}")
        except Exception as e:
            print(f"❌ Error extracting data: {e}")
            return

        # Handle tuple output
        if isinstance(pred_ids, tuple):
            pred_ids = pred_ids[0]
            print("🔄 Handled tuple prediction output")

        try:
            # Replace -100 with pad token id for decoding
            # Get appropriate pad token id
            if hasattr(self.tokenizer, 'decoder') and hasattr(self.tokenizer.decoder, 'pad_token_id'):
                pad_token_id = self.tokenizer.decoder.pad_token_id
            elif hasattr(self.tokenizer, 'pad_token_id') and self.tokenizer.pad_token_id is not None:
                pad_token_id = self.tokenizer.pad_token_id
            else:
                # Fallback to 0 if no pad token id found
                pad_token_id = 0
            
            print(f"🏷️ Using pad_token_id: {pad_token_id}")
            
            labels_for_decode = []
            for label_seq in labels:
                cleaned_labels = [
                    token if token != -100 else pad_token_id
                    for token in label_seq
                ]
                labels_for_decode.append(cleaned_labels)

            pred_ids_for_decode = []
            for pred_seq in pred_ids:
                cleaned_preds = [
                    token if token != -100 else pad_token_id
                    for token in pred_seq
                ]
                pred_ids_for_decode.append(cleaned_preds)
            
            print("🧹 Cleaned prediction and label sequences")
        except Exception as e:
            print(f"❌ Error cleaning sequences: {e}")
            return

        try:
            # Decode sequences
            print(f"🔤 Tokenizer type: {type(self.tokenizer)}")
            print(f"🔤 Has decoder: {hasattr(self.tokenizer, 'decoder')}")
            print(f"🔤 Has encoder: {hasattr(self.tokenizer, 'encoder')}")
            
            if hasattr(self.tokenizer, 'decoder') and hasattr(self.tokenizer, 'encoder'):
                # Handle EncoderDecoder tokenizer
                print("🔤 Using EncoderDecoder tokenizer")
                try:
                    inputs = self.tokenizer.encoder.batch_decode(input_ids, skip_special_tokens=True)
                    targets = self.tokenizer.decoder.batch_decode(labels_for_decode, skip_special_tokens=True)
                    predictions_text = self.tokenizer.decoder.batch_decode(pred_ids_for_decode, skip_special_tokens=True)
                except Exception as e:
                    print(f"⚠️ Error with EncoderDecoder tokenizer, falling back to regular: {e}")
                    inputs = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
                    targets = self.tokenizer.batch_decode(labels_for_decode, skip_special_tokens=True)
                    predictions_text = self.tokenizer.batch_decode(pred_ids_for_decode, skip_special_tokens=True)
            else:
                # Handle regular tokenizer
                print("🔤 Using regular tokenizer")
                inputs = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
                targets = self.tokenizer.batch_decode(labels_for_decode, skip_special_tokens=True)
                predictions_text = self.tokenizer.batch_decode(pred_ids_for_decode, skip_special_tokens=True)
            
            print(f"✅ Decoded {len(inputs)} inputs, {len(targets)} targets, {len(predictions_text)} predictions")
            print(f"📄 Sample input: {inputs[0][:100]}...")
            print(f"📄 Sample target: {targets[0][:100]}...")
            print(f"📄 Sample prediction: {predictions_text[0][:100]}...")
            
        except Exception as e:
            print(f"❌ Error decoding sequences: {e}")
            import traceback
            traceback.print_exc()
            return

        try:
            # Extract source sentences from the original dataset
            print("🔍 Extracting source sentences...")
            sources = self._extract_source_sentences()
            print(f"✅ Extracted {len(sources)} source sentences")
            print(f"📄 Sample source: {sources[0][:100]}...")
        except Exception as e:
            print(f"❌ Error extracting sources: {e}")
            sources = ["N/A"] * len(inputs)

        try:
            # Compute individual metrics if available
            print("📊 Computing individual metrics...")
            print(f"📊 Input lengths - predictions: {len(predictions_text)}, targets: {len(targets)}, sources: {len(sources)}")
            individual_metrics = self._compute_individual_metrics(
                predictions_text, targets, sources
            )
            print(f"✅ Computed metrics: {list(individual_metrics.keys())}")
            if individual_metrics:
                for metric_name, scores in individual_metrics.items():
                    print(f"📊 {metric_name}: {len(scores)} scores, sample: {scores[:3] if scores else 'empty'}")
        except Exception as e:
            print(f"❌ Error computing individual metrics: {e}")
            import traceback
            traceback.print_exc()
            individual_metrics = {}

        try:
            # Create comprehensive dataframe for logging
            print("📊 Creating logging dataframe...")
            df_data = {
                "step": [global_step] * len(inputs),
                "source": inputs,
                "reference": targets,
                "prediction": predictions_text,
                # Remove input_tokens column - it's confusing and not needed
            }

            # Add individual metric scores if available
            if individual_metrics:
                print(f"📊 Adding individual metrics: {list(individual_metrics.keys())}")
                for metric_name, scores in individual_metrics.items():
                    if len(scores) == len(inputs):
                        df_data[f"{metric_name}_score"] = scores
                        print(f"✅ Added {metric_name}_score column with {len(scores)} values")
                    else:
                        print(f"⚠️ Metric {metric_name} has wrong length: {len(scores)} vs {len(inputs)}")
            else:
                print("⚠️ No individual metrics computed")

            df = pd.DataFrame(df_data)
            print(f"✅ Created dataframe with shape: {df.shape}")
            print(f"📊 Columns: {list(df.columns)}")
            
            # Print sample of the dataframe for debugging
            print("📋 Sample dataframe content:")
            for i, row in df.head(3).iterrows():
                print(f"  Row {i}: source='{row['source'][:50]}...', prediction='{row['prediction'][:50]}...'")
                for col in df.columns:
                    if col.endswith('_score'):
                        print(f"    {col}: {row[col]}")

            # Log main evaluation table
            print("📤 Logging to wandb...")
            table = wandb.Table(dataframe=df)
            wandb.log({"evaluation_results": table}, step=global_step)
            print(f"✅ Logged evaluation_results table to wandb for step {global_step}")

            # Log summary statistics
            self._log_summary_statistics(df, global_step, individual_metrics)
            print("✅ Logged summary statistics")

        except Exception as e:
            print(f"❌ Error creating/logging dataframe: {e}")
            import traceback
            traceback.print_exc()

    def _extract_source_sentences(self) -> List[str]:
        """Extract source sentences from the evaluation dataset."""
        sources = []
        try:
            # First try to get from the original eval_dataset (before tokenization)
            if hasattr(self.eval_dataset, 'features') and self.source_column in self.eval_dataset.features:
                # Get the first num_samples from the original dataset
                original_sources = self.eval_dataset[self.source_column][:self.num_samples]
                sources = [str(src) for src in original_sources]
                print(f"✅ Extracted {len(sources)} source sentences from original dataset")
            elif hasattr(self.eval_dataset, '__getitem__'):
                # Try to get sources from original dataset items
                for i in range(min(self.num_samples, len(self.eval_dataset))):
                    item = self.eval_dataset[i]
                    if isinstance(item, dict) and self.source_column in item:
                        sources.append(str(item[self.source_column]))
                    else:
                        sources.append("N/A")
                print(f"✅ Extracted {len(sources)} source sentences from original dataset items")
            else:
                # Fallback: try sample_dataset 
                if hasattr(self.sample_dataset, 'features') and self.source_column in self.sample_dataset.features:
                    sources = [str(src) for src in self.sample_dataset[self.source_column]]
                else:
                    sources = ["N/A"] * self.num_samples
                print(f"⚠️ Using fallback source extraction: {len(sources)} sources")
        except Exception as e:
            print(f"Warning: Could not extract source sentences: {e}")
            import traceback
            traceback.print_exc()
            sources = ["N/A"] * self.num_samples

        # Ensure correct length
        if len(sources) != self.num_samples:
            if len(sources) < self.num_samples:
                sources.extend(["N/A"] * (self.num_samples - len(sources)))
            else:
                sources = sources[:self.num_samples]

        return sources

    def _compute_individual_metrics(self, 
                                   predictions: List[str], 
                                   references: List[str], 
                                   sources: List[str]) -> Dict[str, List[float]]:
        """Compute individual metric scores for each sample."""
        # Always compute individual metrics regardless of compute_metrics_fn
        print(f"🔍 Computing individual metrics for {len(predictions)} predictions, {len(references)} references, {len(sources)} sources")
        try:
            # Import evaluators to compute individual scores
            from ..evaluators.custom_metrics import get_individual_georgian_comet_scores
            from sacrebleu.metrics import BLEU, CHRF

            individual_metrics = {}

            # Compute individual BLEU scores
            try:
                bleu_metric = BLEU()
                bleu_scores = []
                for pred, ref in zip(predictions, references):
                    try:
                        score = bleu_metric.sentence_score(pred, [ref])
                        bleu_scores.append(float(score.score))
                    except:
                        bleu_scores.append(0.0)
                individual_metrics["bleu"] = bleu_scores
            except Exception as e:
                print(f"Warning: Could not compute individual BLEU scores: {e}")

            # Compute individual chrF scores
            try:
                chrf_metric = CHRF(word_order=2)
                chrf_scores = []
                for pred, ref in zip(predictions, references):
                    try:
                        score = chrf_metric.sentence_score(pred, [ref])
                        chrf_scores.append(float(score.score))
                    except:
                        chrf_scores.append(0.0)
                individual_metrics["chrf"] = chrf_scores
            except Exception as e:
                print(f"Warning: Could not compute individual chrF scores: {e}")

            # Compute individual COMET scores if sources are available
            try:
                if sources and any(s != "N/A" and s.strip() for s in sources):
                    comet_config = {
                        "model_name": "Darsala/georgian_comet",
                        "batch_size": 8,  # Smaller batch size for individual scoring
                        "device": "cuda",
                        "gpus": 1
                    }
                    comet_scores = get_individual_georgian_comet_scores(
                        predictions, references, sources, comet_config
                    )
                    individual_metrics["comet"] = comet_scores
            except Exception as e:
                print(f"Warning: Could not compute individual COMET scores: {e}")
            
            return individual_metrics

        except Exception as e:
            print(f"Warning: Could not compute individual metrics: {e}")
            return {}

    def _log_summary_statistics(self, 
                               df: pd.DataFrame, 
                               global_step: int,
                               individual_metrics: Dict[str, List[float]]):
        """Log summary statistics for the evaluation batch."""
        summary_stats = {
            "eval_samples_count": len(df),
            "avg_prediction_length": df["prediction"].str.len().mean(),
            "avg_reference_length": df["reference"].str.len().mean(),
            "avg_source_length": df["source"].str.len().mean(),
        }

        # Add metric statistics
        for metric_name, scores in individual_metrics.items():
            if scores:
                summary_stats.update({
                    f"eval_{metric_name}_mean": sum(scores) / len(scores),
                    f"eval_{metric_name}_std": pd.Series(scores).std(),
                    f"eval_{metric_name}_min": min(scores),
                    f"eval_{metric_name}_max": max(scores),
                })

        wandb.log(summary_stats, step=global_step)


# Keep the original class for backward compatibility
class WandbPredictionProgressCallback(WandbEvaluationCallback):
    """Backward compatibility alias for WandbEvaluationCallback."""
    pass


def get_early_stopping_callback(config: Dict[str, Any]) -> EarlyStoppingCallback:
    """
    Create an early stopping callback with specified configuration.

    Args:
        config: Early stopping configuration

    Returns:
        Configured EarlyStoppingCallback
    """
    patience = config.get('patience', 3)
    threshold = config.get('threshold', 0.0)

    return EarlyStoppingCallback(
        early_stopping_patience=patience,
        early_stopping_threshold=threshold
    )


class CustomEarlyStoppingCallback(TrainerCallback):
    """
    Custom early stopping callback with additional features.
    """

    def __init__(self,
                 patience: int = 3,
                 threshold: float = 0.0,
                 greater_is_better: bool = False,
                 min_delta: float = 0.0):
        """
        Initialize custom early stopping callback.

        Args:
            patience: Number of evaluations with no improvement before stopping
            threshold: Minimum change to qualify as improvement
            greater_is_better: Whether higher metric values are better
            min_delta: Minimum change to qualify as improvement
        """
        self.patience = patience
        self.threshold = threshold
        self.greater_is_better = greater_is_better
        self.min_delta = min_delta
        self.patience_counter = 0
        self.best_metric = float('-inf') if greater_is_better else float('inf')

    def on_evaluate(self, args, state, control, **kwargs):
        """Called after each evaluation."""
        metric_to_check = args.metric_for_best_model
        if not metric_to_check.startswith("eval_"):
            metric_to_check = f"eval_{metric_to_check}"

        logs = kwargs.get('logs', {})
        current_metric = logs.get(metric_to_check)

        if current_metric is None:
            return

        # Check if this is an improvement
        if self.greater_is_better:
            is_improvement = current_metric > self.best_metric + self.min_delta
        else:
            is_improvement = current_metric < self.best_metric - self.min_delta

        if is_improvement:
            self.best_metric = current_metric
            self.patience_counter = 0
        else:
            self.patience_counter += 1

        # Stop training if patience exceeded
        if self.patience_counter >= self.patience:
            control.should_training_stop = True