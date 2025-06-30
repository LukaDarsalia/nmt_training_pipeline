"""
Model Implementations

This module contains various model architectures and their implementations.
Import all model files to register their components.
"""

# Import model implementations to register them
from . import encoder_decoder_scratch_models
from . import m2m100_models
from . import encoder_decoder_models
from . import marian_models

# Custom models can be added here
# from . import custom_models

__all__ = [
    'encoder_decoder_scratch_models',
    'm2m100_models', 
    'encoder_decoder_models',
    'marian_models',
]