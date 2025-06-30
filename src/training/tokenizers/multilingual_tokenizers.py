"""
Multilingual Tokenizer Implementations

Provides tokenizer implementations for multilingual models like M2M100 and mBART.
"""

from typing import Dict, Any, Tuple

from transformers import AutoTokenizer
from transformers.tokenization_utils import PreTrainedTokenizer
from datasets import Dataset, DatasetDict

from ..registry.tokenizer_registry import BaseTokenizer, register_tokenizer


@register_tokenizer("m2m100_multilingual", "M2M100 multilingual tokenizer")
class M2M100MultilingualTokenizer(BaseTokenizer):
    """M2M100 multilingual tokenizer implementation."""
    
    def load_tokenizer(self) -> PreTrainedTokenizer:
        """
        Load M2M100 tokenizer from model name.
        
        Returns:
            Configured M2M100 tokenizer
        """
        model_name = self.config.get('model_name', 'facebook/m2m100_418M')
        print(f"Loading M2M100 tokenizer from: {model_name}")
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Suppress deprecation warnings
        tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True
        
        # Set source and target languages if specified
        source_lang = self.config.get('source_lang', 'en')
        target_lang = self.config.get('target_lang', 'ka')
        if hasattr(tokenizer, 'src_lang'):
            tokenizer.src_lang = source_lang
            print(f"Set source language to: {source_lang}")
        if hasattr(tokenizer, 'tgt_lang'):
            tokenizer.tgt_lang = target_lang
            print(f"Set target language to: {target_lang}")
        
        print(f"Loaded M2M100 tokenizer: {model_name}")
        print(f"Vocab size: {tokenizer.vocab_size}")
        
        return tokenizer
    
    def tokenize_datasets(self, 
                         datasets: Tuple[Dataset, Dataset, Dataset],
                         tokenization_config: Dict[str, Any]) -> Tuple[Dataset, Dataset, Dataset]:
        """
        Tokenize datasets for M2M100 multilingual model.
        
        Args:
            datasets: Tuple of (train, valid, test) datasets
            tokenization_config: Tokenization configuration
            
        Returns:
            Tuple of tokenized datasets
        """
        if self.tokenizer is None:
            self.tokenizer = self.load_tokenizer()
        
        train_dataset, valid_dataset, test_dataset = datasets
        
        # Get configuration parameters
        max_length = tokenization_config.get('max_length', 128)
        source_column = tokenization_config.get('source_column', 'en')
        target_column = tokenization_config.get('target_column', 'ka')
        source_lang = tokenization_config.get('source_lang', 'en')
        target_lang = tokenization_config.get('target_lang', 'ka')
        
        def preprocess_function(examples):
            """Preprocess function for M2M100 tokenization."""
            if self.tokenizer is None:
                raise ValueError("Tokenizer is not loaded")
            # Set source and target languages in tokenizer
            if hasattr(self.tokenizer, 'src_lang'):
                self.tokenizer.src_lang = source_lang
            if hasattr(self.tokenizer, 'tgt_lang'):
                self.tokenizer.tgt_lang = target_lang
            
            # Get source and target texts
            inputs = [str(ex) for ex in examples[source_column]]
            targets = [str(ex) for ex in examples[target_column]]
            
            # Tokenize
            model_inputs = self.tokenizer(
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
            desc="Tokenizing M2M100 datasets"
        )
        
        print(f"M2M100 tokenization complete:")
        print(f"  Max length: {max_length}")
        print(f"  Source language: {source_lang}")
        print(f"  Target language: {target_lang}")
        print(f"  Source column: {source_column}")
        print(f"  Target column: {target_column}")
        
        return (
            tokenized_datasets['train'],
            tokenized_datasets['valid'],
            tokenized_datasets['test']
        )


@register_tokenizer("mbart_multilingual", "mBART multilingual tokenizer")
class MBartMultilingualTokenizer(BaseTokenizer):
    """mBART multilingual tokenizer implementation."""
    
    def load_tokenizer(self) -> PreTrainedTokenizer:
        """
        Load mBART tokenizer from model name.
        
        Returns:
            Configured mBART tokenizer
        """
        model_name = self.config.get('model_name', 'facebook/mbart-large-50-many-to-many-mmt')
        print(f"Loading mBART tokenizer from: {model_name}")
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Suppress deprecation warnings
        tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True
        
        print(f"Loaded mBART tokenizer: {model_name}")
        print(f"Vocab size: {tokenizer.vocab_size}")
        
        return tokenizer
    
    def tokenize_datasets(self, 
                         datasets: Tuple[Dataset, Dataset, Dataset],
                         tokenization_config: Dict[str, Any]) -> Tuple[Dataset, Dataset, Dataset]:
        """
        Tokenize datasets for mBART multilingual model.
        
        Args:
            datasets: Tuple of (train, valid, test) datasets
            tokenization_config: Tokenization configuration
            
        Returns:
            Tuple of tokenized datasets
        """
        if self.tokenizer is None:
            self.tokenizer = self.load_tokenizer()
        
        train_dataset, valid_dataset, test_dataset = datasets
        
        # Get configuration parameters
        max_length = tokenization_config.get('max_length', 128)
        source_column = tokenization_config.get('source_column', 'en')
        target_column = tokenization_config.get('target_column', 'ka')
        source_lang = tokenization_config.get('source_lang', 'en')
        target_lang = tokenization_config.get('target_lang', 'ka')
        
        def preprocess_function(examples):
            """Preprocess function for mBART tokenization."""
            if self.tokenizer is None:
                raise ValueError("Tokenizer is not loaded")
            # Get source and target texts
            inputs = [str(ex) for ex in examples[source_column]]
            targets = [str(ex) for ex in examples[target_column]]
            
            # Tokenize
            model_inputs = self.tokenizer(
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
            desc="Tokenizing mBART datasets"
        )
        
        print(f"mBART tokenization complete:")
        print(f"  Max length: {max_length}")
        print(f"  Source language: {source_lang}")
        print(f"  Target language: {target_lang}")
        print(f"  Source column: {source_column}")
        print(f"  Target column: {target_column}")
        
        return (
            tokenized_datasets['train'],
            tokenized_datasets['valid'],
            tokenized_datasets['test']
        ) 