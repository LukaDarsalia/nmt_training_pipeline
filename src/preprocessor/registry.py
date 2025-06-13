"""
Data Preprocessor Registry

A registry system for data preprocessing and augmentation functions.
"""

from typing import Dict, Callable, Optional

import pandas as pd


class PreprocessorRegistry:
    """Registry for data preprocessing and augmentation functions."""

    def __init__(self):
        self._preprocessors: Dict[str, Callable] = {}
        self._augmenters: Dict[str, Callable] = {}
        self._descriptions: Dict[str, str] = {}

    def register_preprocessor(self, name: str, description: str = "") -> Callable:
        """
        Decorator to register a preprocessing function.
        Preprocessing functions modify existing data in-place.

        Args:
            name: Name of the preprocessor function
            description: Description of what this preprocessor does

        Returns:
            Decorator function
        """

        def decorator(func: Callable) -> Callable:
            self._preprocessors[name] = func
            self._descriptions[name] = description
            return func

        return decorator

    def register_augmenter(self, name: str, description: str = "") -> Callable:
        """
        Decorator to register an augmentation function.
        Augmentation functions create new data rows.

        Args:
            name: Name of the augmenter function
            description: Description of what this augmenter does

        Returns:
            Decorator function
        """

        def decorator(func: Callable) -> Callable:
            self._augmenters[name] = func
            self._descriptions[name] = description
            return func

        return decorator

    def get_preprocessor(self, name: str) -> Optional[Callable]:
        """Get a preprocessor function by name."""
        return self._preprocessors.get(name)

    def get_augmenter(self, name: str) -> Optional[Callable]:
        """Get an augmenter function by name."""
        return self._augmenters.get(name)

    def list_preprocessors(self) -> Dict[str, str]:
        """Get all registered preprocessors with their descriptions."""
        return {name: self._descriptions.get(name, "") for name in self._preprocessors.keys()}

    def list_augmenters(self) -> Dict[str, str]:
        """Get all registered augmenters with their descriptions."""
        return {name: self._descriptions.get(name, "") for name in self._augmenters.keys()}

    def validate_preprocessor_output(self,
                                   processed_df: pd.DataFrame,
                                   original_df: pd.DataFrame,
                                   preprocessor_name: str) -> pd.DataFrame:
        """
        Validate that preprocessor output maintains dataset integrity.

        Args:
            processed_df: DataFrame after preprocessing
            original_df: Original DataFrame
            preprocessor_name: Name of the preprocessor for error reporting

        Returns:
            Validated processed DataFrame

        Raises:
            ValueError: If validation fails
        """
        original_len = len(original_df)
        processed_len = len(processed_df)

        # Check that row count is preserved
        if processed_len != original_len:
            raise ValueError(
                f"Preprocessor '{preprocessor_name}' changed row count: "
                f"original={original_len}, processed={processed_len}"
            )

        # Check that columns are preserved
        expected_columns = set(original_df.columns)
        if set(processed_df.columns) != expected_columns:
            raise ValueError(
                f"Preprocessor '{preprocessor_name}' changed columns. "
                f"Expected: {expected_columns}, Got: {set(processed_df.columns)}"
            )

        return processed_df

    def validate_augmenter_output(self,
                                augmented_df: pd.DataFrame,
                                original_df: pd.DataFrame,
                                augmenter_name: str) -> pd.DataFrame:
        """
        Validate that augmenter output has correct structure.

        Args:
            augmented_df: DataFrame with augmented data
            original_df: Original DataFrame
            augmenter_name: Name of the augmenter for error reporting

        Returns:
            Validated augmented DataFrame

        Raises:
            ValueError: If validation fails
        """
        # Check that columns match original
        expected_columns = set(original_df.columns)
        if set(augmented_df.columns) != expected_columns:
            raise ValueError(
                f"Augmenter '{augmenter_name}' produced wrong columns. "
                f"Expected: {expected_columns}, Got: {set(augmented_df.columns)}"
            )

        # Augmented data should be non-empty if enabled
        if len(augmented_df) == 0:
            print(f"Warning: Augmenter '{augmenter_name}' produced no new data")

        return augmented_df


# Global registry instance
preprocessor_registry = PreprocessorRegistry()


def register_preprocessor(name: str, description: str = "") -> Callable:
    """Convenience function to register a preprocessor."""
    return preprocessor_registry.register_preprocessor(name, description)


def register_augmenter(name: str, description: str = "") -> Callable:
    """Convenience function to register an augmenter."""
    return preprocessor_registry.register_augmenter(name, description)
