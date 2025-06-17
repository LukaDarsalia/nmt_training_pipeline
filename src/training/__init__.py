"""
Training Pipeline Module

A flexible, registry-based training system for neural machine translation models.
Supports various model architectures, training strategies, and evaluation metrics.
"""

from .registry import model_registry, trainer_registry, evaluator_registry
from .trainer import NMTTrainer

# Import all implementations to register them
from . import models, trainers, evaluators

__all__ = [
    'model_registry',
    'trainer_registry',
    'evaluator_registry',
    'NMTTrainer',
    'models',
    'trainers',
    'evaluators'
]