"""
Trainer Implementations

Contains various training strategies and configurations for neural machine translation.
All trainers are registered with the trainer_registry for easy experimentation.
"""

# Import all trainer implementations to register them
from . import seq2seq_trainers
from . import custom_trainers

__all__ = [
    'seq2seq_trainers',
    'custom_trainers'
]