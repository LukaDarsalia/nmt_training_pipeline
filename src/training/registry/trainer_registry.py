"""
Trainer Registry

Registry for different training strategies and approaches.
"""

from typing import Callable, Dict, Any
from transformers.trainer import Trainer
from .base import BaseRegistry


class TrainerRegistry(BaseRegistry):
    """Registry for training strategies and configurations."""

    def validate_component_output(self, output: Any, component_name: str) -> Any:
        """
        Validate that trainer component returns correct format.

        Args:
            output: Should be a Trainer instance
            component_name: Name of the trainer component

        Returns:
            Validated trainer instance

        Raises:
            ValueError: If output format is incorrect
        """
        if not isinstance(output, Trainer):
            raise ValueError(
                f"Trainer component '{component_name}' must return Trainer instance. "
                f"Got: {type(output)}"
            )

        return output

    def create_trainer(self,
                       trainer_name: str,
                       config: Dict[str, Any],
                       model: Any,
                       tokenizer: Any,
                       train_dataset: Any,
                       eval_dataset: Any,
                       data_collator: Any) -> Trainer:
        """
        Create a trainer using registered components.

        Args:
            trainer_name: Name of the registered trainer component
            config: Trainer configuration parameters
            model: The model to train
            tokenizer: Tokenizer for the model
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset
            data_collator: Data collator for batching

        Returns:
            Configured trainer instance

        Raises:
            ValueError: If trainer not found or validation fails
        """
        self.validate_component_exists(trainer_name, "Trainer")

        trainer_func = self.get(trainer_name)
        if trainer_func is None:
            raise ValueError(f"Trainer '{trainer_name}' not found in registry")

        trainer = trainer_func(
            config=config,
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator
        )

        return self.validate_component_output(trainer, trainer_name)


# Global trainer registry instance
trainer_registry = TrainerRegistry()


def register_trainer(name: str, description: str = "") -> Callable:
    """Convenience function to register a trainer."""
    return trainer_registry.register(name, description)