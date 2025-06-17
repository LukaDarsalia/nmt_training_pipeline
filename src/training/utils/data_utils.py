"""
Training Data Utilities

Provides utility functions for loading and processing data for training.
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


def tokenize_datasets(
        datasets: Tuple[Dataset, Dataset, Dataset],
        tokenizer: PreTrainedTokenizer,
        config: Dict[str, Any]
) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Tokenize datasets for training.

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


def prepare_encoder_decoder_tokenization(
        datasets: Tuple[Dataset, Dataset, Dataset],
        tokenizer: PreTrainedTokenizer,
        config: Dict[str, Any]
) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Prepare datasets for encoder-decoder models with special tokenization.

    Args:
        datasets: Tuple of (train, valid, test) datasets
        tokenizer: Tokenizer to use
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
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                targets,
                max_length=max_length,
                truncation=True,
                padding='max_length'
            )

        model_inputs["labels"] = labels["input_ids"]

        # Special processing for encoder-decoder models
        if config.get('encoder_decoder_preprocessing', False):
            # Apply custom preprocessing logic here
            for i, input_ids in enumerate(model_inputs['input_ids']):
                # Move last token to first position and set first token to BOS
                input_ids = input_ids[-1:] + input_ids[:-1]
                input_ids[0] = tokenizer.cls_token_id or tokenizer.bos_token_id

                # Replace SEP token with EOS token
                if tokenizer.sep_token_id in input_ids:
                    sep_idx = input_ids.index(tokenizer.sep_token_id)
                    input_ids[sep_idx] = tokenizer.eos_token_id
                elif tokenizer.pad_token_id in input_ids:
                    pad_idx = input_ids.index(tokenizer.pad_token_id)
                    input_ids[pad_idx] = tokenizer.eos_token_id
                else:
                    input_ids[-1] = tokenizer.eos_token_id

                # Remove UNK tokens and pad
                count_unks = input_ids.count(tokenizer.unk_token_id or 1)
                input_ids = [token for token in input_ids if token != (tokenizer.unk_token_id or 1)]
                input_ids.extend([tokenizer.pad_token_id] * count_unks)

                model_inputs['input_ids'][i] = input_ids

            # Process labels similarly
            for i, label_ids in enumerate(model_inputs['labels']):
                label_ids = label_ids[-1:] + label_ids[:-1]
                label_ids[0] = tokenizer.bos_token_id
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