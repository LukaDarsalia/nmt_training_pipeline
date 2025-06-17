"""
Model Implementations

Contains various model architectures and configurations for neural machine translation.
All models are registered with the model_registry for easy experimentation.
"""

# Import all model implementations to register them
from . import marian_models
from . import encoder_decoder_models
from . import custom_models

__all__ = [
    'marian_models',
    'encoder_decoder_models',
    'custom_models'
]