"""
Model Registry

Registry for different model architectures including pretrained NMT models,
encoder-decoder combinations, and custom architectures.
"""

from typing import Callable, Dict, Any, Tuple
import torch.nn as nn
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.generation.configuration_utils import GenerationConfig
from .base import BaseRegistry


class ModelRegistry(BaseRegistry):
    """Registry for model architectures and configurations."""

    def validate_component_output(self, output: Any, component_name: str) -> Any:
        """
        Validate that model component returns correct format.

        Args:
            output: Should be tuple of (model, generation_config, data_collator)
            component_name: Name of the model component

        Returns:
            Validated output tuple

        Raises:
            ValueError: If output format is incorrect
        """
        if not isinstance(output, tuple) or len(output) != 3:
            raise ValueError(
                f"Model component '{component_name}' must return tuple of "
                f"(model, generation_config, data_collator). Got: {type(output)}"
            )

        model, generation_config, data_collator = output

        if not isinstance(model, nn.Module):
            raise ValueError(
                f"Model component '{component_name}' must return nn.Module as first element. "
                f"Got: {type(model)}"
            )

        if not isinstance(generation_config, GenerationConfig):
            raise ValueError(
                f"Model component '{component_name}' must return GenerationConfig as second element. "
                f"Got: {type(generation_config)}"
            )

        return output

    def create_model(self,
                     model_name: str,
                     config: Dict[str, Any],
                     tokenizer: PreTrainedTokenizer) -> Tuple[nn.Module, GenerationConfig, Any]:
        """
        Create a model using registered components.

        Args:
            model_name: Name of the registered model component
            config: Model configuration parameters
            tokenizer: Tokenizer for the model

        Returns:
            Tuple of (model, generation_config, data_collator)

        Raises:
            ValueError: If model not found or validation fails
        """
        self.validate_component_exists(model_name, "Model")

        model_func = self.get(model_name)
        if model_func is None:
            raise ValueError(f"Model '{model_name}' not found in registry")

        output = model_func(config, tokenizer)

        return self.validate_component_output(output, model_name)


# Global model registry instance
model_registry = ModelRegistry()


def register_model(name: str, description: str = "") -> Callable:
    """Convenience function to register a model."""
    return model_registry.register(name, description)
