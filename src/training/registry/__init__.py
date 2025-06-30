"""
Registry System for Training Components

Provides registry classes for models, trainers, evaluators, and tokenizers to enable
flexible experimentation and easy addition of new components.
"""

from .base import BaseRegistry
from .model_registry import ModelRegistry, model_registry
from .trainer_registry import TrainerRegistry, trainer_registry
from .evaluator_registry import EvaluatorRegistry, evaluator_registry
from .tokenizer_registry import TokenizerRegistry, tokenizer_registry

__all__ = [
    'BaseRegistry',
    'ModelRegistry',
    'TrainerRegistry',
    'EvaluatorRegistry',
    'TokenizerRegistry',
    'model_registry',
    'trainer_registry',
    'evaluator_registry',
    'tokenizer_registry'
]