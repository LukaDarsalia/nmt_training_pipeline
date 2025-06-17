"""
Training Callbacks

Provides various callbacks for training including early stopping,
prediction logging, and other monitoring utilities.
"""

from typing import Dict, Any

import pandas as pd
import wandb
from transformers import (
    TrainerCallback,
    EarlyStoppingCallback,
    PreTrainedTokenizer
)
from transformers.integrations import WandbCallback


class WandbPredictionProgressCallback(WandbCallback):
    """
    Custom WandbCallback to log model predictions during training.

    This callback logs model predictions and labels to a wandb.Table at each
    evaluation step during training, allowing visualization of model progress.
    """

    def __init__(self,
                 tokenizer: PreTrainedTokenizer,
                 eval_dataset: Any,
                 num_samples: int = 20,
                 log_frequency: int = 2,
                 max_length: int = 128,
                 num_beams: int = 5):
        """
        Initialize the prediction progress callback.

        Args:
            tokenizer: Tokenizer for decoding predictions
            eval_dataset: Evaluation dataset to sample from
            num_samples: Number of samples to log
            log_frequency: Log every N evaluation steps
            max_length: Maximum generation length
            num_beams: Number of beams for generation
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.eval_dataset = eval_dataset
        self.num_samples = min(num_samples, len(eval_dataset))
        self.log_frequency = log_frequency
        self.max_length = max_length
        self.num_beams = num_beams
        self.counter = 0

        # Sample subset of evaluation dataset
        if hasattr(eval_dataset, 'select'):
            self.sample_dataset = eval_dataset.select(range(self.num_samples))
        else:
            # If dataset doesn't have select method, take first N samples
            self.sample_dataset = eval_dataset[:self.num_samples]

    def on_evaluate(self, args, state, control, **kwargs):
        """Called after each evaluation step."""
        super().on_evaluate(args, state, control, **kwargs)

        self.counter += 1

        if self.counter % self.log_frequency == 0:
            trainer = kwargs.get('trainer')
            if trainer is None:
                return

            # Generate predictions
            predictions = trainer.predict(
                self.sample_dataset,
                max_length=self.max_length,
                num_beams=self.num_beams
            )

            # Prepare data for logging
            input_ids = self.sample_dataset["input_ids"]
            labels = self.sample_dataset["labels"]
            pred_ids = predictions.predictions

            # Handle tuple output
            if isinstance(pred_ids, tuple):
                pred_ids = pred_ids[0]

            # Replace -100 with pad token id for decoding
            labels_for_decode = []
            for label_seq in labels:
                cleaned_labels = [
                    token if token != -100 else self.tokenizer.pad_token_id
                    for token in label_seq
                ]
                labels_for_decode.append(cleaned_labels)

            pred_ids_for_decode = []
            for pred_seq in pred_ids:
                cleaned_preds = [
                    token if token != -100 else self.tokenizer.pad_token_id
                    for token in pred_seq
                ]
                pred_ids_for_decode.append(cleaned_preds)

            # Decode sequences
            inputs = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
            targets = self.tokenizer.batch_decode(labels_for_decode, skip_special_tokens=True)
            predictions_text = self.tokenizer.batch_decode(pred_ids_for_decode, skip_special_tokens=True)

            # Create dataframe for logging
            df = pd.DataFrame({
                "input": inputs,
                "target": targets,
                "prediction": predictions_text,
                "step": [state.global_step] * len(inputs)
            })

            # Log to wandb
            table = wandb.Table(dataframe=df)
            wandb.log({"sample_predictions": table}, step=state.global_step)


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