"""NMT training pipeline with registry-based extensibility."""

import os
import inspect
from pathlib import Path
from typing import Any, Dict, List, Tuple

import wandb
import yaml
from datasets import load_dataset
from transformers import (
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

from .models import *  # register model functions
from .registry import model_registry, trainer_registry
from .evaluation import metric_registry


@trainer_registry.register("seq2seq_trainer", "HuggingFace Seq2SeqTrainer")
def default_trainer(
    model: Any,
    tokenizer: Any,
    train_dataset: Any,
    eval_dataset: Any,
    training_args: Seq2SeqTrainingArguments,
) -> Seq2SeqTrainer:
    """Create a standard :class:`~transformers.Seq2SeqTrainer`."""

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    return trainer


class NMTTrainer:
    """Main interface for running training experiments."""

    def __init__(self, artifact: wandb.Artifact, dataset_dir: str, config_path: str = "config/training.yaml") -> None:
        """Initialize the trainer with an artifact, dataset location and config file."""

        self.artifact = artifact
        self.dataset_dir = Path(dataset_dir)
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self._log_config_to_wandb()

    def _load_config(self) -> Dict[str, Any]:
        """Load YAML configuration from :attr:`config_path`."""

        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        with open(self.config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def _log_config_to_wandb(self) -> None:
        """Attach configuration and source files to the wandb artifact."""

        self.artifact.add_file(str(self.config_path), name="training_config.yaml")
        code_files = [
            "src/trainer/trainer.py",
            "src/trainer/models.py",
            "src/trainer/registry.py",
            "src/trainer/evaluation.py",
        ]
        for file_path in code_files:
            if Path(file_path).exists():
                self.artifact.add_file(file_path, name=f"training_code/{Path(file_path).name}")

    # ------------------------------------------------------------------
    def _load_datasets(self) -> Tuple[Any, Any, Any]:
        """Load train, validation and test datasets from parquet files."""

        ds_cfg = self.config.get("dataset", {})
        train_file = self.dataset_dir / ds_cfg.get("train_file", "train.parquet")
        valid_file = self.dataset_dir / ds_cfg.get("valid_file", "valid.parquet")
        test_file = self.dataset_dir / ds_cfg.get("test_file", "test.parquet")

        dataset = load_dataset(
            "parquet",
            data_files={
                "train": str(train_file),
                "validation": str(valid_file),
                "test": str(test_file),
            },
        )
        return dataset["train"], dataset["validation"], dataset["test"]

    def _build_model(self) -> Tuple[Any, Any]:
        """Create model and tokenizer based on the configured architecture."""

        model_cfg = self.config.get("model", {})
        architecture = model_cfg.get("architecture", "auto")
        model_fn = model_registry.get_model(architecture)
        if model_fn is None:
            raise ValueError(
                f"Model architecture '{architecture}' not found. Available: {list(model_registry.list_models().keys())}"
            )
        return model_fn(model_cfg)

    def _tokenize_dataset(self, tokenizer: Any, train_ds: Any, valid_ds: Any, test_ds: Any) -> Tuple[Any, Any, Any]:
        """Tokenize raw datasets using the provided tokenizer."""

        cfg = self.config.get("trainer", {})
        src_len = cfg.get("max_source_length", 128)
        tgt_len = cfg.get("max_target_length", 128)

        enc_tok = getattr(tokenizer, "encoder_tokenizer", tokenizer)
        dec_tok = getattr(tokenizer, "decoder_tokenizer", tokenizer)

        def preprocess(examples):
            model_inputs = enc_tok(
                examples["en"],
                max_length=src_len,
                truncation=True,
            )
            labels = dec_tok(
                examples["ka"],
                max_length=tgt_len,
                truncation=True,
            )
            model_inputs["labels"] = labels["input_ids"]
            return model_inputs

        train_ds = train_ds.map(preprocess, batched=True)
        valid_ds = valid_ds.map(preprocess, batched=True)
        test_ds = test_ds.map(preprocess, batched=True)
        return train_ds, valid_ds, test_ds

    def _build_trainer(self, model: Any, tokenizer: Any, train_ds: Any, valid_ds: Any, test_ds: Any) -> Tuple[Seq2SeqTrainer, str]:
        """Instantiate the trainer and attach callbacks/metrics."""

        cfg = self.config.get("trainer", {})
        output_dir = cfg.get("output_dir", "model_output")
        os.makedirs(output_dir, exist_ok=True)

        arg_values = {
            "output_dir": output_dir,
            "num_train_epochs": cfg.get("num_train_epochs", 3),
            "per_device_train_batch_size": cfg.get("per_device_train_batch_size", 8),
            "per_device_eval_batch_size": cfg.get("per_device_eval_batch_size", 8),
            "gradient_accumulation_steps": cfg.get("gradient_accumulation_steps", 1),
            "learning_rate": cfg.get("learning_rate", 5e-5),
            "weight_decay": cfg.get("weight_decay", 0.0),
            "warmup_steps": cfg.get("warmup_steps", 0),
            "lr_scheduler_type": cfg.get("lr_scheduler_type", "cosine"),
            "evaluation_strategy": cfg.get("evaluation_strategy", "epoch"),
            "save_strategy": cfg.get("save_strategy", "epoch"),
            "logging_steps": cfg.get("logging_steps", 100),
            "report_to": "wandb",
        }

        valid_args = inspect.signature(Seq2SeqTrainingArguments.__init__).parameters
        filtered_args = {k: v for k, v in arg_values.items() if k in valid_args}

        training_args = Seq2SeqTrainingArguments(**filtered_args)

        metrics = cfg.get("metrics", [])

        def compute_metrics(eval_preds: Tuple[Any, Any]) -> Dict[str, float]:
            preds, labels = eval_preds
            if isinstance(preds, tuple):
                preds = preds[0]
            decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
            decoded_preds = [p.strip() for p in decoded_preds]
            decoded_labels = [l.strip() for l in decoded_labels]

            all_scores: Dict[str, float] = {}
            for name in metrics:
                metric_fn = metric_registry.get_metric(name)
                if metric_fn is None:
                    raise ValueError(
                        f"Metric '{name}' not found. Available: {list(metric_registry.list_metrics().keys())}"
                    )
                all_scores.update(metric_fn(decoded_preds, decoded_labels))
            return all_scores

        trainer_type = cfg.get("trainer_type", "seq2seq_trainer")
        trainer_fn = trainer_registry.get_trainer(trainer_type)
        if trainer_fn is None:
            raise ValueError(
                f"Trainer '{trainer_type}' not found. Available: {list(trainer_registry.list_trainers().keys())}"
            )
        trainer = trainer_fn(
            model,
            tokenizer,
            train_ds,
            valid_ds,
            training_args,
        )

        if metrics:
            trainer.compute_metrics = compute_metrics

        patience = cfg.get("early_stopping_patience")
        if patience and patience > 0:
            threshold = cfg.get("early_stopping_threshold", 0.0)
            trainer.add_callback(
                EarlyStoppingCallback(
                    early_stopping_patience=patience,
                    early_stopping_threshold=threshold,
                )
            )

        return trainer, output_dir

    # ------------------------------------------------------------------
    def run_training(self) -> None:
        """Execute the full training loop."""

        train_ds, valid_ds, test_ds = self._load_datasets()
        model, tokenizer = self._build_model()
        train_ds, valid_ds, test_ds = self._tokenize_dataset(tokenizer, train_ds, valid_ds, test_ds)
        trainer, output_dir = self._build_trainer(model, tokenizer, train_ds, valid_ds, test_ds)
        trainer.train()
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)
        self.artifact.add_dir(output_dir)
        self.artifact.metadata.update({
            "num_train_samples": len(train_ds),
            "num_valid_samples": len(valid_ds),
            "num_test_samples": len(test_ds),
        })
        print(f"Model saved to {output_dir}")
