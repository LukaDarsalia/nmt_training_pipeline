"""
Dataset Loader Registry

A registry system for dataset loaders that ensures consistent output format
and provides a simple interface for adding new loaders.
"""

from typing import Dict, Callable, Optional

import pandas as pd


class LoaderRegistry:
    """Registry for dataset loading functions."""

    def __init__(self):
        self._loaders: Dict[str, Callable] = {}
        self._descriptions: Dict[str, str] = {}

    def register(self, name: str, description: str = "") -> Callable:
        """
        Decorator to register a loader function.

        Args:
            name: Name of the loader function
            description: Description of what this loader does

        Returns:
            Decorator function
        """

        def decorator(func: Callable) -> Callable:
            self._loaders[name] = func
            self._descriptions[name] = description
            return func

        return decorator

    def get_loader(self, name: str) -> Optional[Callable]:
        """Get a loader function by name."""
        return self._loaders.get(name)

    def list_loaders(self) -> Dict[str, str]:
        """Get all registered loaders with their descriptions."""
        return {name: self._descriptions.get(name, "") for name in self._loaders.keys()}

    def validate_loader_output(self, df: pd.DataFrame, loader_name: str) -> pd.DataFrame:
        """
        Validate that loader output has required columns for MultilingualDataset.

        Args:
            df: DataFrame from loader
            loader_name: Name of the loader for error reporting

        Returns:
            Validated DataFrame

        Raises:
            ValueError: If required columns are missing
        """
        required_columns = ['title', 'ka', 'en', 'domain', 'id']
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            raise ValueError(
                f"Loader '{loader_name}' output missing required columns: {missing_columns}. "
                f"Required columns: {required_columns}"
            )

        # Ensure no null values in critical columns
        for col in ['ka', 'en', 'id']:
            if df[col].isnull().any():
                print(f"Warning: Loader '{loader_name}' has null values in column '{col}'")

        return df


# Global registry instance
loader_registry = LoaderRegistry()


def register_loader(name: str, description: str = "") -> Callable:
    """Convenience function to register a loader."""
    return loader_registry.register(name, description)