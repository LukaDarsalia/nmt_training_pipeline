"""
Data Cleaner Registry

A registry system for data cleaning functions that ensures consistent output format
and provides a simple interface for adding new cleaning functions.
"""

from typing import Dict, Callable, Optional, Tuple
import pandas as pd


class CleanerRegistry:
    """Registry for data cleaning functions."""

    def __init__(self):
        self._cleaners: Dict[str, Callable] = {}
        self._descriptions: Dict[str, str] = {}

    def register(self, name: str, description: str = "") -> Callable:
        """
        Decorator to register a cleaner function.

        Args:
            name: Name of the cleaner function
            description: Description of what this cleaner does

        Returns:
            Decorator function
        """

        def decorator(func: Callable) -> Callable:
            self._cleaners[name] = func
            self._descriptions[name] = description
            return func

        return decorator

    def get_cleaner(self, name: str) -> Optional[Callable]:
        """Get a cleaner function by name."""
        return self._cleaners.get(name)

    def list_cleaners(self) -> Dict[str, str]:
        """Get all registered cleaners with their descriptions."""
        return {name: self._descriptions.get(name, "") for name in self._cleaners.keys()}

    def validate_cleaner_output(self,
                              cleaned_df: pd.DataFrame,
                              dropped_df: pd.DataFrame,
                              original_df: pd.DataFrame,
                              cleaner_name: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Validate that cleaner output is consistent with input.

        Args:
            cleaned_df: DataFrame with cleaned data
            dropped_df: DataFrame with dropped data
            original_df: Original DataFrame
            cleaner_name: Name of the cleaner for error reporting

        Returns:
            Validated (cleaned_df, dropped_df) tuple

        Raises:
            ValueError: If validation fails
        """
        original_len = len(original_df)
        cleaned_len = len(cleaned_df)
        dropped_len = len(dropped_df)

        # Check that total rows match
        if cleaned_len + dropped_len != original_len:
            raise ValueError(
                f"Cleaner '{cleaner_name}' output inconsistent: "
                f"original={original_len}, cleaned={cleaned_len}, dropped={dropped_len}"
            )

        # Check that columns are preserved in cleaned_df
        expected_columns = set(original_df.columns)
        if set(cleaned_df.columns) != expected_columns:
            raise ValueError(
                f"Cleaner '{cleaner_name}' changed columns in cleaned_df. "
                f"Expected: {expected_columns}, Got: {set(cleaned_df.columns)}"
            )

        # Allow dropped_df to have additional columns (for logging purposes)
        # but ensure it has at least the original columns
        if len(dropped_df) > 0:
            dropped_columns = set(dropped_df.columns)
            if not expected_columns.issubset(dropped_columns):
                missing_cols = expected_columns - dropped_columns
                raise ValueError(
                    f"Cleaner '{cleaner_name}' missing required columns in dropped_df: {missing_cols}"
                )

        return cleaned_df, dropped_df


# Global registry instance
cleaner_registry = CleanerRegistry()


def register_cleaner(name: str, description: str = "") -> Callable:
    """Convenience function to register a cleaner."""
    return cleaner_registry.register(name, description)
