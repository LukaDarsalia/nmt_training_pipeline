"""Registries for models and trainers."""

from typing import Callable, Dict, Optional


class ModelRegistry:
    """Registry mapping model names to factory functions."""

    def __init__(self) -> None:
        """Initialize an empty registry."""

        self._models: Dict[str, Callable] = {}
        self._descriptions: Dict[str, str] = {}

    def register(self, name: str, description: str = "") -> Callable:
        """Return a decorator registering ``func`` under ``name``."""

        def decorator(func: Callable) -> Callable:
            self._models[name] = func
            self._descriptions[name] = description
            return func

        return decorator

    def get_model(self, name: str) -> Optional[Callable]:
        """Return the model factory registered as ``name`` if it exists."""

        return self._models.get(name)

    def list_models(self) -> Dict[str, str]:
        """Return a mapping of registered model names to descriptions."""

        return {name: self._descriptions.get(name, "") for name in self._models}


class TrainerRegistry:
    """Registry mapping trainer names to factory functions."""

    def __init__(self) -> None:
        """Initialize an empty trainer registry."""

        self._trainers: Dict[str, Callable] = {}
        self._descriptions: Dict[str, str] = {}

    def register(self, name: str, description: str = "") -> Callable:
        """Return a decorator registering ``func`` under ``name``."""

        def decorator(func: Callable) -> Callable:
            self._trainers[name] = func
            self._descriptions[name] = description
            return func

        return decorator

    def get_trainer(self, name: str) -> Optional[Callable]:
        """Return the trainer factory registered as ``name`` if it exists."""

        return self._trainers.get(name)

    def list_trainers(self) -> Dict[str, str]:
        """Return a mapping of registered trainer names to descriptions."""

        return {name: self._descriptions.get(name, "") for name in self._trainers}


# Global registries
model_registry = ModelRegistry()
trainer_registry = TrainerRegistry()


# Convenience decorators
def register_model(name: str, description: str = "") -> Callable:
    """Decorator to register a model factory in :data:`model_registry`."""

    return model_registry.register(name, description)


def register_trainer(name: str, description: str = "") -> Callable:
    """Decorator to register a trainer factory in :data:`trainer_registry`."""

    return trainer_registry.register(name, description)
