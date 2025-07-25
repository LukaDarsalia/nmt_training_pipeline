"""
Tokenizer Registry

Registry for different tokenizer types and their associated tokenization methods.
Provides a unified interface for loading tokenizers and tokenizing datasets.
"""

from typing import Callable, Dict, Any, Tuple, Optional
from abc import ABC, abstractmethod

from transformers.tokenization_utils import PreTrainedTokenizer
from datasets import Dataset, DatasetDict


class BaseTokenizer(ABC):
    """Base class for all tokenizer implementations."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the tokenizer.
        
        Args:
            config: Tokenizer configuration
        """
        self.config = config
        self.tokenizer: Optional[PreTrainedTokenizer] = None
    
    @abstractmethod
    def load_tokenizer(self) -> PreTrainedTokenizer:
        """
        Load and configure the tokenizer.
        
        Returns:
            Configured tokenizer
        """
        pass
    
    def tokenize_datasets(self, 
                         datasets: Tuple[Dataset, Dataset, Dataset],
                         tokenization_config: Dict[str, Any]) -> Tuple[Dataset, Dataset, Dataset]:
        """
        Tokenize datasets using the loaded tokenizer.
        
        Args:
            datasets: Tuple of (train, valid, test) datasets
            tokenization_config: Tokenization configuration
            
        Returns:
            Tuple of tokenized datasets
        """
        if self.tokenizer is None:
            self.tokenizer = self.load_tokenizer()
        
        return self._default_tokenize_datasets(datasets, self.tokenizer, tokenization_config)
    
    def _default_tokenize_datasets(self,
                                  datasets: Tuple[Dataset, Dataset, Dataset],
                                  tokenizer: PreTrainedTokenizer,
                                  config: Dict[str, Any]) -> Tuple[Dataset, Dataset, Dataset]:
        """
        Default tokenization method for seq2seq models.
        
        Args:
            datasets: Tuple of (train, valid, test) datasets
            tokenizer: Tokenizer to use
            config: Tokenization configuration
            
        Returns:
            Tuple of tokenized datasets
        """
        train_dataset, valid_dataset, test_dataset = datasets
        
        # Get configuration parameters
        max_length = config.get('max_length', 128)
        source_column = config.get('source_column', 'en')
        target_column = config.get('target_column', 'ka')
        prefix = config.get('target_prefix', '')
        
        def preprocess_function(examples):
            """Preprocess function for tokenization."""
            # Get source and target texts
            inputs = [str(ex) for ex in examples[source_column]]
            targets = [prefix + str(ex) for ex in examples[target_column]]
            
            # Tokenize
            model_inputs = tokenizer(
                inputs,
                text_target=targets,
                max_length=max_length,
                truncation=True,
                padding='max_length'
            )
            
            return model_inputs
        
        # Create dataset dict for easier processing
        dataset_dict = DatasetDict({
            'train': train_dataset,
            'valid': valid_dataset,
            'test': test_dataset
        })
        
        # Tokenize all datasets
        tokenized_datasets = dataset_dict.map(
            preprocess_function,
            batched=True,
            remove_columns=dataset_dict["train"].column_names,
            desc="Tokenizing datasets"
        )
        
        print(f"Tokenization complete:")
        print(f"  Max length: {max_length}")
        print(f"  Source column: {source_column}")
        print(f"  Target column: {target_column}")
        print(f"  Target prefix: '{prefix}'")
        
        return (
            tokenized_datasets['train'],
            tokenized_datasets['valid'],
            tokenized_datasets['test']
        )


class TokenizerRegistry:
    """Registry for tokenizer implementations."""
    
    def __init__(self):
        """Initialize the registry."""
        self._tokenizers: Dict[str, Callable] = {}
        self._descriptions: Dict[str, str] = {}
    
    def register(self, name: str, description: str = "") -> Callable:
        """
        Decorator to register a tokenizer implementation.
        
        Args:
            name: Name of the tokenizer
            description: Description of the tokenizer
            
        Returns:
            Decorator function
        """
        def decorator(func: Callable) -> Callable:
            self._tokenizers[name] = func
            self._descriptions[name] = description
            return func
        return decorator
    
    def get(self, name: str) -> Optional[Callable]:
        """
        Get a tokenizer implementation by name.
        
        Args:
            name: Name of the tokenizer
            
        Returns:
            Tokenizer implementation or None if not found
        """
        return self._tokenizers.get(name)
    
    def list_components(self) -> Dict[str, str]:
        """
        List all registered tokenizers with descriptions.
        
        Returns:
            Dictionary of tokenizer names and descriptions
        """
        return {name: self._descriptions.get(name, "") for name in self._tokenizers.keys()}
    
    def validate_component_exists(self, name: str, component_type: str) -> None:
        """
        Validate that a component exists in the registry.
        
        Args:
            name: Name of the component
            component_type: Type of component for error message
            
        Raises:
            ValueError: If component doesn't exist
        """
        if name not in self._tokenizers:
            available = list(self._tokenizers.keys())
            raise ValueError(
                f"{component_type} '{name}' not found in tokenizer registry. "
                f"Available tokenizers: {available}"
            )


# Global tokenizer registry instance
tokenizer_registry = TokenizerRegistry()


def register_tokenizer(name: str, description: str = "") -> Callable:
    """Convenience function to register a tokenizer."""
    return tokenizer_registry.register(name, description) 