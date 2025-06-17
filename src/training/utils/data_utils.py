"""
Training Data Utilities

Provides utility functions for loading and processing data for training,
including support for multilingual models and automatic tokenizer handling.
"""

from typing import Tuple, Dict, Any
from pathlib import Path
import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import PreTrainedTokenizer


def load_datasets_from_artifact(artifact_path: str) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Load train, validation, and test datasets from an artifact directory.

    Args:
        artifact_path: Path to the downloaded artifact directory

    Returns:
        Tuple of (train_dataset, valid_dataset, test_dataset)

    Raises:
        FileNotFoundError: If required split files are not found
    """
    artifact_dir = Path(artifact_path)

    # Define expected file paths
    train_path = artifact_dir / "train.parquet"
    valid_path = artifact_dir / "valid.parquet"
    test_path = artifact_dir / "test.parquet"

    # Check if files exist
    missing_files = []
    if not train_path.exists():
        missing_files.append("train.parquet")
    if not valid_path.exists():
        missing_files.append("valid.parquet")
    if not test_path.exists():
        missing_files.append("test.parquet")

    if missing_files:
        raise FileNotFoundError(
            f"Missing required split files in {artifact_dir}: {missing_files}"
        )

    # Load datasets
    train_df = pd.read_parquet(train_path)
    valid_df = pd.read_parquet(valid_path)
    test_df = pd.read_parquet(test_path)

    # Convert to HuggingFace datasets
    train_dataset = Dataset.from_pandas(train_df)
    valid_dataset = Dataset.from_pandas(valid_df)
    test_dataset = Dataset.from_pandas(test_df)

    # Remove pandas index column if present
    columns_to_remove = ["__index_level_0__"]
    for col in columns_to_remove:
        if col in train_dataset.features:
            train_dataset = train_dataset.remove_columns([col])
        if col in valid_dataset.features:
            valid_dataset = valid_dataset.remove_columns([col])
        if col in test_dataset.features:
            test_dataset = test_dataset.remove_columns([col])

    print(f"Loaded datasets:")
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Valid: {len(valid_dataset)} samples")
    print(f"  Test: {len(test_dataset)} samples")

    return train_dataset, valid_dataset, test_dataset


def _is_multilingual_tokenizer(tokenizer: PreTrainedTokenizer) -> bool:
    """
    Check if tokenizer is a multilingual model tokenizer.

    Args:
        tokenizer: Tokenizer to check

    Returns:
        True if it's a multilingual tokenizer
    """
    # Check for common multilingual tokenizer attributes
    multilingual_indicators = [
        'src_lang',           # M2M100
        'lang_code_to_id',    # mBART
        'get_lang_id',        # M2M100 method
        'tgt_lang'            # Some multilingual models
    ]

    return any(hasattr(tokenizer, indicator) for indicator in multilingual_indicators)


def tokenize_datasets(
        datasets: Tuple[Dataset, Dataset, Dataset],
        tokenizer: PreTrainedTokenizer,
        config: Dict[str, Any]
) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Tokenize datasets for training with automatic multilingual detection.

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

    # Check if this is a multilingual model that needs special handling
    multilingual_model = config.get('multilingual_model', False) or _is_multilingual_tokenizer(tokenizer)

    if multilingual_model:
        print("Detected multilingual tokenizer, using multilingual tokenization")
        return tokenize_multilingual_datasets(datasets, tokenizer, config)

    print("Using standard seq2seq tokenization")

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


def tokenize_multilingual_datasets(
        datasets: Tuple[Dataset, Dataset, Dataset],
        tokenizer: PreTrainedTokenizer,
        config: Dict[str, Any]
) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Tokenize datasets for multilingual models (M2M100, mBART, etc.).

    Args:
        datasets: Tuple of (train, valid, test) datasets
        tokenizer: Multilingual tokenizer (M2M100Tokenizer, MBart50Tokenizer, etc.)
        config: Tokenization configuration

    Returns:
        Tuple of tokenized datasets
    """
    train_dataset, valid_dataset, test_dataset = datasets

    # Get configuration parameters
    max_length = config.get('max_length', 128)
    source_column = config.get('source_column', 'en')
    target_column = config.get('target_column', 'ka')
    source_lang = config.get('source_lang', 'en')
    target_lang = config.get('target_lang', 'ka')

    def preprocess_multilingual_function(examples):
        """Preprocess function for multilingual models."""
        # Get source and target texts
        inputs = [str(ex) for ex in examples[source_column]]
        targets = [str(ex) for ex in examples[target_column]]

        # Set source language for tokenizer if supported
        if hasattr(tokenizer, 'src_lang'):
            original_src_lang = getattr(tokenizer, 'src_lang', None)
            tokenizer.src_lang = source_lang

        # Tokenize inputs
        model_inputs = tokenizer(
            inputs,
            max_length=max_length,
            truncation=True,
            padding='max_length'
        )

        # Tokenize targets with target tokenizer context
        if hasattr(tokenizer, 'as_target_tokenizer'):
            with tokenizer.as_target_tokenizer():
                labels = tokenizer(
                    targets,
                    max_length=max_length,
                    truncation=True,
                    padding='max_length'
                )
        else:
            # For tokenizers that don't have as_target_tokenizer context
            # Set target language if supported
            if hasattr(tokenizer, 'tgt_lang'):
                original_tgt_lang = getattr(tokenizer, 'tgt_lang', None)
                tokenizer.tgt_lang = target_lang

            labels = tokenizer(
                targets,
                max_length=max_length,
                truncation=True,
                padding='max_length'
            )

            # Restore original target language
            if hasattr(tokenizer, 'tgt_lang') and 'original_tgt_lang' in locals():
                tokenizer.tgt_lang = original_tgt_lang

        model_inputs["labels"] = labels["input_ids"]

        # Handle special tokens for M2M100
        if hasattr(tokenizer, 'get_lang_id'):
            try:
                target_lang_id = tokenizer.get_lang_id(target_lang)
                # Set forced BOS token for target language
                for i in range(len(model_inputs["labels"])):
                    labels_list = model_inputs["labels"][i].copy()
                    # Find first non-pad token and replace with target lang id
                    for j, token_id in enumerate(labels_list):
                        if token_id != tokenizer.pad_token_id:
                            labels_list[j] = target_lang_id
                            break
                    model_inputs["labels"][i] = labels_list
            except Exception as e:
                print(f"Warning: Could not set target language token: {e}")

        # Restore original source language
        if hasattr(tokenizer, 'src_lang') and 'original_src_lang' in locals():
            tokenizer.src_lang = original_src_lang

        return model_inputs

    # Create dataset dict for easier processing
    dataset_dict = DatasetDict({
        'train': train_dataset,
        'valid': valid_dataset,
        'test': test_dataset
    })

    # Tokenize all datasets
    tokenized_datasets = dataset_dict.map(
        preprocess_multilingual_function,
        batched=True,
        remove_columns=dataset_dict["train"].column_names,
        desc="Tokenizing multilingual datasets"
    )

    print(f"Multilingual tokenization complete:")
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


def prepare_encoder_decoder_tokenization(
        datasets: Tuple[Dataset, Dataset, Dataset],
        tokenizer: PreTrainedTokenizer,
        config: Dict[str, Any]
) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Prepare datasets for encoder-decoder models with special tokenization.

    Args:
        datasets: Tuple of (train, valid, test) datasets
        tokenizer: Tokenizer to use (typically from decoder model)
        config: Tokenization configuration

    Returns:
        Tuple of tokenized datasets with encoder-decoder specific processing
    """
    train_dataset, valid_dataset, test_dataset = datasets

    # Get configuration parameters
    max_length = config.get('max_length', 128)
    source_column = config.get('source_column', 'en')
    target_column = config.get('target_column', 'ka')

    def preprocess_encoder_decoder(examples):
        """Preprocess function for encoder-decoder models."""
        inputs = [str(ex) for ex in examples[source_column]]
        targets = [str(ex) for ex in examples[target_column]]

        # Tokenize inputs and targets separately
        model_inputs = tokenizer(
            inputs,
            max_length=max_length,
            truncation=True,
            padding='max_length'
        )

        # Tokenize targets
        if hasattr(tokenizer, 'as_target_tokenizer'):
            with tokenizer.as_target_tokenizer():
                labels = tokenizer(
                    targets,
                    max_length=max_length,
                    truncation=True,
                    padding='max_length'
                )
        else:
            labels = tokenizer(
                targets,
                max_length=max_length,
                truncation=True,
                padding='max_length'
            )

        model_inputs["labels"] = labels["input_ids"]

        # Special preprocessing for encoder-decoder models
        if config.get('encoder_decoder_preprocessing', False):
            print("Applying encoder-decoder specific preprocessing...")

            # Apply custom preprocessing logic here
            for i, input_ids in enumerate(model_inputs['input_ids']):
                input_ids = list(input_ids)  # Convert to list for manipulation

                # Ensure we have proper special tokens
                if tokenizer.bos_token_id is not None and input_ids[0] != tokenizer.bos_token_id:
                    # Add BOS token at the beginning
                    input_ids = [tokenizer.bos_token_id] + input_ids[:-1]

                # Replace SEP token with EOS token if present
                if tokenizer.sep_token_id is not None and tokenizer.sep_token_id in input_ids:
                    sep_idx = input_ids.index(tokenizer.sep_token_id)
                    input_ids[sep_idx] = tokenizer.eos_token_id or tokenizer.sep_token_id

                # Ensure EOS token at the end (before padding)
                if tokenizer.eos_token_id is not None:
                    # Find last non-pad token
                    last_non_pad = len(input_ids) - 1
                    while last_non_pad >= 0 and input_ids[last_non_pad] == tokenizer.pad_token_id:
                        last_non_pad -= 1

                    if last_non_pad >= 0 and input_ids[last_non_pad] != tokenizer.eos_token_id:
                        if last_non_pad < len(input_ids) - 1:
                            input_ids[last_non_pad + 1] = tokenizer.eos_token_id
                        else:
                            input_ids[last_non_pad] = tokenizer.eos_token_id

                model_inputs['input_ids'][i] = input_ids

            # Process labels similarly
            for i, label_ids in enumerate(model_inputs['labels']):
                label_ids = list(label_ids)  # Convert to list for manipulation

                if tokenizer.bos_token_id is not None and label_ids[0] != tokenizer.bos_token_id:
                    label_ids = [tokenizer.bos_token_id] + label_ids[:-1]

                model_inputs['labels'][i] = label_ids

        return model_inputs

    # Create dataset dict
    dataset_dict = DatasetDict({
        'train': train_dataset,
        'valid': valid_dataset,
        'test': test_dataset
    })

    # Tokenize all datasets
    tokenized_datasets = dataset_dict.map(
        preprocess_encoder_decoder,
        batched=True,
        remove_columns=dataset_dict["train"].column_names,
        desc="Tokenizing for encoder-decoder"
    )

    print(f"Encoder-decoder tokenization complete:")
    print(f"  Max length: {max_length}")
    print(f"  Source column: {source_column}")
    print(f"  Target column: {target_column}")
    print(f"  Special preprocessing: {config.get('encoder_decoder_preprocessing', False)}")

    return (
        tokenized_datasets['train'],
        tokenized_datasets['valid'],
        tokenized_datasets['test']
    )