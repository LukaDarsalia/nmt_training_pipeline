"""
Evaluator Implementations

Contains various evaluation metrics for neural machine translation.
All evaluators are registered with the evaluator_registry for easy experimentation.
"""

# Import all evaluator implementations to register them
from . import standard_metrics
from . import custom_metrics

__all__ = [
    'standard_metrics',
    'custom_metrics'
]