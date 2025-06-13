"""Training Module

A configurable training pipeline for NMT experiments using HuggingFace
Transformers. Models, tokenizers and trainers are selected via simple
registry systems, enabling easy experimentation with different
architectures.
"""

from .registry import (
    model_registry,
    trainer_registry,
    register_model,
    register_trainer,
)
from .evaluation import metric_registry, register_metric
from .trainer import NMTTrainer

__all__ = [
    'model_registry',
    'trainer_registry',
    'metric_registry',
    'register_model',
    'register_trainer',
    'register_metric',
    'NMTTrainer',
]
