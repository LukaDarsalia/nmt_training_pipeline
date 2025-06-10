"""
Dataset Loading Functions

This module contains all the dataset loading functions registered with the loader registry.
Each function should return a pandas DataFrame with columns: ['title', 'ka', 'en', 'domain', 'id']
"""

from typing import Dict, Any

import pandas as pd
from datasets import load_dataset

from .registry import register_loader


@register_loader("load_en_ka_corpora", "Load English-Georgian parallel corpus from HuggingFace")
def load_en_ka_corpora(config: Dict[str, Any]) -> pd.DataFrame:
    """
    Load English-Georgian corpora dataset.

    Args:
        config: Dataset configuration from YAML

    Returns:
        DataFrame with required columns
    """
    ds = load_dataset(config["source"])
    df = ds[config.get("split", "train")].to_pandas()

    # The dataset already has the right format, just ensure column consistency
    if 'title' not in df.columns:
        df['title'] = None

    return df


@register_loader("load_flores_devtest", "Load FLORES test set")
def load_flores_devtest(config: Dict[str, Any]) -> pd.DataFrame:
    """
    Load FLORES development and test sets for Georgian-English.

    Args:
        config: Dataset configuration from YAML

    Returns:
        DataFrame with required columns
    """
    params = config.get("params", {})
    ka_lang = params.get("ka_lang", "kat_Geor")
    en_lang = params.get("en_lang", "eng_Latn")
    split = params.get("split", "devtest")

    ds_ka = load_dataset(config["source"], ka_lang, split=split)
    ds_en = load_dataset(config["source"], en_lang, split=split)

    ka_df = ds_ka.to_pandas()
    en_df = ds_en.to_pandas()

    data = {
        'title': [None] * len(en_df),
        'ka': ka_df['text'].tolist(),
        'en': en_df['text'].tolist(),
        'domain': en_df['topic'].tolist(),
        'id': [f"{row['domain']}_{row['id']}" for _, row in en_df.iterrows()]
    }

    return pd.DataFrame(data)


# Example of how to add a new loader
@register_loader("load_custom_csv", "Load data from a CSV file")
def load_custom_csv(config: Dict[str, Any]) -> pd.DataFrame:
    """
    Example loader for CSV files.

    Args:
        config: Dataset configuration from YAML

    Returns:
        DataFrame with required columns
    """
    df = pd.read_csv(config["source"])

    # Map your CSV columns to required format
    # This is just an example - adapt based on your CSV structure
    mapped_df = pd.DataFrame({
        'title': df.get('title', None),
        'ka': df['georgian_text'],  # Adapt column names as needed
        'en': df['english_text'],  # Adapt column names as needed
        'domain': df.get('category', 'general'),
        'id': df.index.map(lambda x: f"custom_{x}")
    })

    return mapped_df
