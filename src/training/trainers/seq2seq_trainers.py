"""
Seq2Seq Trainer Implementations

Provides various sequence-to-sequence training strategies using HuggingFace Transformers.
"""

from typing import Dict, Any, Optional, List
import torch
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrainerCallback,
    EarlyStoppingCallback,
    PreTrainedTokenizer
)
from ..registry.trainer_registry import register_trainer
from ..utils.callbacks import WandbPredictionProgressCallback, get_early_stopping_callback


@register_trainer("standard_seq2seq", "Standard sequence-to-sequence trainer with cosine scheduler")
def create_standard_seq2seq_trainer(config: Dict[str, Any],
                                    model: torch.nn.Module,
                                    tokenizer: PreTrainedTokenizer,
                                    train_dataset: Any,
                                    eval_dataset: Any,
                                    data_collator: Any) -> Seq2SeqTrainer:
    """
    Create a standard Seq2Seq trainer with cosine learning rate scheduler.

    Args:
        config: Training configuration parameters
        model: The model to train
        tokenizer: Tokenizer for the model
        train_dataset: Training dataset
        eval_dataset: Evaluation dataset
        data_collator: Data collator for batching

    Returns:
        Configured Seq2SeqTrainer instance
    """
    # Get training parameters
    training_params = config.get('training', {})
    generation_params = config.get('generation', {})

    # Create training arguments
    args = Seq2SeqTrainingArguments(
        output_dir=config.get('output_dir', 'output'),

        # Training parameters
        num_train_epochs=training_params.get('num_epochs', 10),
        per_device_train_batch_size=training_params.get('train_batch_size', 16),
        per_device_eval_batch_size=training_params.get('eval_batch_size', 16),
        gradient_accumulation_steps=training_params.get('gradient_accumulation_steps', 1),

        # Learning rate and optimization
        learning_rate=training_params.get('learning_rate', 5e-5),
        lr_scheduler_type=training_params.get('lr_scheduler_type', 'cosine'),
        warmup_ratio=training_params.get('warmup_ratio', 0.1),
        weight_decay=training_params.get('weight_decay', 0.01),

        # Evaluation and logging
        evaluation_strategy=training_params.get('evaluation_strategy', 'steps'),
        eval_steps=training_params.get('eval_steps', 500),
        logging_steps=training_params.get('logging_steps', 100),
        save_steps=training_params.get('save_steps', 500),
        save_total_limit=training_params.get('save_total_limit', 3),

        # Generation parameters
        predict_with_generate=True,
        generation_max_length=generation_params.get('max_length', 128),
        generation_num_beams=generation_params.get('num_beams', 1),

        # Other parameters
        fp16=training_params.get('fp16', True),
        dataloader_num_workers=training_params.get('num_workers', 4),
        seed=training_params.get('seed', 42),
        data_seed=training_params.get('seed', 42),

        # Best model tracking
        metric_for_best_model=training_params.get('metric_for_best_model', 'eval_loss'),
        greater_is_better=training_params.get('greater_is_better', False),
        load_best_model_at_end=True,

        # Label smoothing
        label_smoothing_factor=training_params.get('label_smoothing_factor', 0.1),

        # Reporting
        report_to=training_params.get('report_to', 'wandb'),
        logging_first_step=True,

        # Push to hub
        push_to_hub=False,
    )

    # Create callbacks
    callbacks = []

    # Early stopping
    early_stopping_config = training_params.get('early_stopping', {})
    if early_stopping_config.get('enabled', True):
        early_stopping_callback = get_early_stopping_callback(early_stopping_config)
        callbacks.append(early_stopping_callback)

    # Prediction logging
    prediction_config = training_params.get('prediction_logging', {})
    if prediction_config.get('enabled', True):
        prediction_callback = WandbPredictionProgressCallback(
            tokenizer=tokenizer,
            eval_dataset=eval_dataset,
            num_samples=prediction_config.get('num_samples', 20),
            log_frequency=prediction_config.get('frequency', 2),
            max_length=generation_params.get('max_length', 128),
            num_beams=generation_params.get('eval_num_beams', 5)
        )
        callbacks.append(prediction_callback)

    # Create trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=callbacks
    )

    return trainer


@register_trainer("seq2seq_with_metrics", "Seq2Seq trainer with custom evaluation metrics")
def create_seq2seq_with_metrics_trainer(config: Dict[str, Any],
                                        model: torch.nn.Module,
                                        tokenizer: PreTrainedTokenizer,
                                        train_dataset: Any,
                                        eval_dataset: Any,
                                        data_collator: Any) -> Seq2SeqTrainer:
    """
    Create a Seq2Seq trainer with custom evaluation metrics.

    Args:
        config: Training configuration parameters including evaluator settings
        model: The model to train
        tokenizer: Tokenizer for the model
        train_dataset: Training dataset
        eval_dataset: Evaluation dataset
        data_collator: Data collator for batching

    Returns:
        Configured Seq2SeqTrainer instance with custom metrics
    """
    # Import evaluator registry here to avoid circular imports
    from ..registry.evaluator_registry import evaluator_registry

    # Get training parameters
    training_params = config.get('training', {})
    generation_params = config.get('generation', {})
    evaluation_params = config.get('evaluation', {})

    # Create evaluator
    evaluator_configs = evaluation_params.get('evaluators', [])
    if evaluator_configs:
        compute_metrics_fn = evaluator_registry.create_combined_evaluator(evaluator_configs)

        def compute_metrics(eval_preds):
            """Compute metrics for evaluation."""
            preds, labels = eval_preds

            # Handle tuple output from model
            if isinstance(preds, tuple):
                preds = preds[0]

            # Replace -100 with pad token id for decoding
            preds = torch.where(
                torch.tensor(preds) == -100,
                tokenizer.pad_token_id,
                torch.tensor(preds)
            )
            labels = torch.where(
                torch.tensor(labels) == -100,
                tokenizer.pad_token_id,
                torch.tensor(labels)
            )

            # Decode predictions and labels
            decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

            # Post-process predictions and labels
            decoded_preds = [pred.strip() for pred in decoded_preds]
            decoded_labels = [label.strip() for label in decoded_labels]

            # Compute metrics
            return compute_metrics_fn(decoded_preds, decoded_labels)
    else:
        compute_metrics_fn = None

    # Create training arguments
    args = Seq2SeqTrainingArguments(
        output_dir=config.get('output_dir', 'output'),

        # Training parameters
        num_train_epochs=training_params.get('num_epochs', 10),
        per_device_train_batch_size=training_params.get('train_batch_size', 16),
        per_device_eval_batch_size=training_params.get('eval_batch_size', 16),
        gradient_accumulation_steps=training_params.get('gradient_accumulation_steps', 1),

        # Learning rate and optimization
        learning_rate=training_params.get('learning_rate', 5e-5),
        lr_scheduler_type=training_params.get('lr_scheduler_type', 'cosine'),
        warmup_ratio=training_params.get('warmup_ratio', 0.1),
        weight_decay=training_params.get('weight_decay', 0.01),

        # Evaluation and logging
        evaluation_strategy=training_params.get('evaluation_strategy', 'steps'),
        eval_steps=training_params.get('eval_steps', 500),
        logging_steps=training_params.get('logging_steps', 100),
        save_steps=training_params.get('save_steps', 500),
        save_total_limit=training_params.get('save_total_limit', 3),

        # Generation parameters
        predict_with_generate=True,
        generation_max_length=generation_params.get('max_length', 128),
        generation_num_beams=generation_params.get('num_beams', 1),

        # Other parameters
        fp16=training_params.get('fp16', True),
        dataloader_num_workers=training_params.get('num_workers', 4),
        seed=training_params.get('seed', 42),
        data_seed=training_params.get('seed', 42),

        # Best model tracking
        metric_for_best_model=training_params.get('metric_for_best_model', 'eval_loss'),
        greater_is_better=training_params.get('greater_is_better', False),
        load_best_model_at_end=True,

        # Label smoothing
        label_smoothing_factor=training_params.get('label_smoothing_factor', 0.1),

        # Reporting
        report_to=training_params.get('report_to', 'wandb'),
        logging_first_step=True,

        # Push to hub
        push_to_hub=False,
    )

    # Create callbacks
    callbacks = []

    # Early stopping
    early_stopping_config = training_params.get('early_stopping', {})
    if early_stopping_config.get('enabled', True):
        early_stopping_callback = get_early_stopping_callback(early_stopping_config)
        callbacks.append(early_stopping_callback)

    # Create trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics_fn,
        callbacks=callbacks
    )

    return trainer