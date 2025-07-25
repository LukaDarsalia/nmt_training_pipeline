"""
Base Registry Class

Provides common functionality for all registry types including registration,
retrieval, and validation of components.
"""

from typing import Dict, Callable, Optional, Any
from abc import ABC, abstractmethod


class BaseRegistry(ABC):
    """Base class for all component registries."""

    def __init__(self):
        """Initialize the registry with empty component and description dictionaries."""
        self._components: Dict[str, Callable[..., Callable[..., Any]]] = {}
        self._descriptions: Dict[str, str] = {}

    def register(self, name: str, description: str = "") -> Callable:
        """
        Decorator to register a component.

        Args:
            name: Unique name for the component
            description: Optional description of what this component does

        Returns:
            Decorator function

        Example:
            @registry.register("my_component", "Description of component")
            def my_component(config):
                return SomeClass(config)
        """

        def decorator(func: Callable) -> Callable:
            if name in self._components:
                raise ValueError(f"Component '{name}' already registered in {self.__class__.__name__}")

            self._components[name] = func
            self._descriptions[name] = description
            return func

        return decorator

    def get(self, name: str) -> Optional[Callable[..., Callable[..., Any]]]:
        """
        Get a registered component by name.

        Args:
            name: Name of the component to retrieve

        Returns:
            Component function or None if not found
        """
        return self._components.get(name)

    def list_components(self) -> Dict[str, str]:
        """
        Get all registered components with their descriptions.

        Returns:
            Dictionary mapping component names to descriptions
        """
        return {name: self._descriptions.get(name, "") for name in self._components.keys()}

    def validate_component_exists(self, name: str, component_type: str) -> None:
        """
        Validate that a component exists and raise informative error if not.

        Args:
            name: Name of the component to check
            component_type: Type of component for error message

        Raises:
            ValueError: If component not found with helpful error message
        """
        if name not in self._components:
            available = list(self._components.keys())
            raise ValueError(
                f"{component_type} '{name}' not found. "
                f"Available {component_type.lower()}s: {available}. "
                f"Make sure the component is registered with @{component_type.lower()}_registry.register decorator."
            )

    @abstractmethod
    def validate_component_output(self, output: Any, component_name: str) -> Any:
        """
        Validate the output of a component.

        Args:
            output: The output to validate
            component_name: Name of the component for error reporting

        Returns:
            Validated output

        Raises:
            ValueError: If validation fails
        """
        pass

    def __len__(self) -> int:
        """Return the number of registered components."""
        return len(self._components)

    def __contains__(self, name: str) -> bool:
        """Check if a component is registered."""
        return name in self._components

    def __repr__(self) -> str:
        """Return string representation of the registry."""
        components = list(self._components.keys())
        return f"{self.__class__.__name__}({len(components)} components: {components})"