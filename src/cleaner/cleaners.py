"""
Data Cleaning Functions

This module contains all the data cleaning functions registered with the cleaner registry.
Each function should take a DataFrame and return (cleaned_df, dropped_df) tuple.
"""

import math
import re
import warnings
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from .registry import register_cleaner

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


def _is_predominantly_georgian(text: str, threshold: float = 0.1, require_georgian: bool = False) -> bool:
    """
    Check if text is predominantly Georgian based on character analysis.

    Args:
        text: Input text to analyze
        threshold: Minimum ratio of Georgian characters required
        require_georgian: If True, require Georgian characters to make up majority

    Returns:
        True if text meets Georgian character criteria
    """
    if not text or not isinstance(text, str):
        return False

    # Georgian Unicode ranges: ა-ჰ (U+10A0-U+10F0) and Ⴀ-Ⴥ (U+10A0-U+10F0)
    georgian_chars = re.sub(r'[^ა-ჰᲐ-Ჰ\d\s~`!@#$%^&*()\-_=+\[{\]}|\'\";:/?.,<>–—\-´''""„…]', '', text)

    if require_georgian:
        return len(text) * (1 - threshold) < len(georgian_chars)

    georgian_only = re.sub(r'[^ა-ჰᲐ-Ჰ]', '', text)
    return math.ceil(len(text) * threshold) <= len(georgian_only)


def _remove_english_suffix_preserve_punctuation(text: str) -> str:
    """
    Remove English suffixes from text while preserving punctuation.

    Args:
        text: Input text

    Returns:
        Text with English suffixes removed
    """
    # Pattern for English suffix before punctuation at end
    english_suffix_pattern = re.compile(r'\s*[A-Za-z]+(\b|_)(?=[^\u10A0-\u10FF]*$)')
    return re.sub(english_suffix_pattern, '', text).strip()


def _remove_roman_numerals(text: str) -> str:
    """
    Remove Roman numerals from text.

    Args:
        text: Input text

    Returns:
        Text with Roman numerals removed
    """
    roman_pattern = re.compile(
        r'\bM{0,4}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})\b',
        re.IGNORECASE
    )
    return re.sub(roman_pattern, '', text).strip()


def _is_empty_or_whitespace(text: str) -> bool:
    """
    Check if text is empty, whitespace, or contains only punctuation.

    Args:
        text: Input text to check

    Returns:
        True if text is considered empty
    """
    if not text or not isinstance(text, str):
        return True

    # Remove whitespace and common punctuation
    cleaned = re.sub(r'[\s\[\]{};:\'"",<.>/?!@#$%^&*()_+\-=|\\`~]', '', text)
    return len(cleaned) == 0


@register_cleaner("filter_georgian_purity", "Remove Georgian texts with too many Latin characters")
def filter_georgian_purity(df: pd.DataFrame, config: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Filter out Georgian texts that contain too many Latin characters.

    Args:
        df: Input DataFrame with 'ka' column
        config: Configuration parameters with 'threshold' (default: 0.5)

    Returns:
        Tuple of (cleaned_df, dropped_df)
    """
    threshold = config.get('threshold', 0.5)

    def check_row(row):
        return _is_predominantly_georgian(row['ka'], threshold=threshold, require_georgian=True)

    tqdm.pandas(desc="Filtering Georgian purity")
    mask = df.progress_apply(check_row, axis=1)

    cleaned_df = df[mask].copy()
    dropped_df = df[~mask].copy()

    return cleaned_df, dropped_df


@register_cleaner("filter_shared_latin_characters", "Remove rows where Latin chars in Georgian don't appear in English")
def filter_shared_latin_characters(df: pd.DataFrame, config: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Filter rows based on whether Latin characters in Georgian text also appear in English text.

    Args:
        df: Input DataFrame with 'en' and 'ka' columns
        config: Configuration parameters (currently unused)

    Returns:
        Tuple of (cleaned_df, dropped_df)
    """
    non_georgian_alpha_pattern = re.compile('[A-Za-z]')

    def check_row(row):
        en_text = str(row['en']).lower()
        ka_text = str(row['ka']).lower()

        # Clean Georgian text
        ka_text = _remove_roman_numerals(ka_text)
        ka_text = _remove_english_suffix_preserve_punctuation(ka_text)

        # Find Latin characters in Georgian text
        latin_chars = non_georgian_alpha_pattern.findall(ka_text)
        unique_chars = set(latin_chars)

        # Check if all Latin chars from Georgian appear in English
        return all(char in en_text for char in unique_chars)

    tqdm.pandas(desc="Filtering shared Latin characters")
    mask = df.progress_apply(check_row, axis=1)

    cleaned_df = df[mask].copy()
    dropped_df = df[~mask].copy()

    return cleaned_df, dropped_df


@register_cleaner("filter_empty_content", "Remove rows with empty or whitespace-only content")
def filter_empty_content(df: pd.DataFrame, config: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Remove rows where English or Georgian text is empty or contains only whitespace/punctuation.

    Args:
        df: Input DataFrame with 'en' and 'ka' columns
        config: Configuration parameters (currently unused)

    Returns:
        Tuple of (cleaned_df, dropped_df)
    """
    def check_row(row):
        en_empty = _is_empty_or_whitespace(str(row['en']))
        ka_empty = _is_empty_or_whitespace(str(row['ka']))
        return not (en_empty or ka_empty)

    tqdm.pandas(desc="Filtering empty content")
    mask = df.progress_apply(check_row, axis=1)

    cleaned_df = df[mask].copy()
    dropped_df = df[~mask].copy()

    return cleaned_df, dropped_df


@register_cleaner("filter_similarity_threshold", "Remove rows with low semantic similarity between EN and KA")
def filter_similarity_threshold(df: pd.DataFrame, config: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Filter rows based on semantic similarity between English and Georgian texts using Jina embeddings.

    Args:
        df: Input DataFrame with 'en' and 'ka' columns
        config: Configuration with 'threshold' (default: 0.5), 'batch_size' (default: 32)

    Returns:
        Tuple of (cleaned_df, dropped_df)
    """
    threshold = config.get('threshold', 0.5)
    batch_size = config.get('batch_size', 32)
    model_name = config.get('model_name', 'jinaai/jina-embeddings-v3')

    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(model_name, trust_remote_code=True)
        use_sentence_transformers = True
    except ImportError:
        try:
            from transformers import AutoModel
            model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
            use_sentence_transformers = False
        except ImportError:
            raise ImportError(
                "Neither sentence-transformers nor transformers library found. "
                "Please install one of them to use similarity filtering."
            )

    # Prepare texts
    en_texts = df['en'].astype(str).tolist()
    ka_texts = df['ka'].astype(str).tolist()

    similarities = []
    task = "text-matching"

    # Process in batches
    num_batches = (len(df) + batch_size - 1) // batch_size

    for i in tqdm(range(num_batches), desc="Computing similarities"):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(df))

        batch_en = en_texts[start_idx:end_idx]
        batch_ka = ka_texts[start_idx:end_idx]

        try:
            if use_sentence_transformers:
                # Encode both languages together for efficiency
                all_texts = batch_en + batch_ka
                embeddings = model.encode(all_texts, task=task, prompt_name=task)

                # Split embeddings
                en_embeddings = embeddings[:len(batch_en)]
                ka_embeddings = embeddings[len(batch_en):]
            else:
                # Use transformers library
                all_texts = batch_en + batch_ka
                embeddings = model.encode(all_texts, task=task)

                # Split embeddings
                en_embeddings = embeddings[:len(batch_en)]
                ka_embeddings = embeddings[len(batch_en):]

            # Compute cosine similarities
            batch_similarities = []
            for en_emb, ka_emb in zip(en_embeddings, ka_embeddings):
                # Normalize embeddings
                en_norm = en_emb / np.linalg.norm(en_emb)
                ka_norm = ka_emb / np.linalg.norm(ka_emb)

                # Compute cosine similarity
                similarity = np.dot(en_norm, ka_norm)
                batch_similarities.append(similarity)

            similarities.extend(batch_similarities)

        except Exception as e:
            print(f"Warning: Error computing similarities for batch {i}: {e}")
            # Fill with average similarity for this batch
            batch_similarities = [threshold] * len(batch_en)
            similarities.extend(batch_similarities)

    # Create mask based on similarity threshold
    similarities = np.array(similarities)
    mask = similarities >= threshold

    # Create mask based on similarity threshold
    mask = similarities >= threshold

    # Create output dataframes with original columns only
    cleaned_df = df[mask].copy()
    dropped_df = df[~mask].copy()

    # Add similarity scores only to dropped_df for logging purposes
    # (this helps with debugging which pairs were dropped due to low similarity)
    if len(dropped_df) > 0:
        dropped_df['similarity_score'] = similarities[~mask]

    return cleaned_df, dropped_df


@register_cleaner("filter_length_ratio", "Remove rows with extreme length differences between EN and KA")
def filter_length_ratio(df: pd.DataFrame, config: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Filter rows where the length ratio between English and Georgian texts is extreme.

    Args:
        df: Input DataFrame with 'en' and 'ka' columns
        config: Configuration with 'min_ratio' (default: 0.3) and 'max_ratio' (default: 3.0)

    Returns:
        Tuple of (cleaned_df, dropped_df)
    """
    min_ratio = config.get('min_ratio', 0.3)
    max_ratio = config.get('max_ratio', 3.0)

    def check_row(row):
        en_len = len(str(row['en']))
        ka_len = len(str(row['ka']))

        if en_len == 0 or ka_len == 0:
            return False

        ratio = en_len / ka_len
        return min_ratio <= ratio <= max_ratio

    tqdm.pandas(desc="Filtering length ratios")
    mask = df.progress_apply(check_row, axis=1)

    cleaned_df = df[mask].copy()
    dropped_df = df[~mask].copy()

    return cleaned_df, dropped_df


@register_cleaner("filter_minimum_length", "Remove rows with very short texts")
def filter_minimum_length(df: pd.DataFrame, config: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Filter rows where English or Georgian texts are too short.

    Args:
        df: Input DataFrame with 'en' and 'ka' columns
        config: Configuration with 'min_en_length' (default: 5) and 'min_ka_length' (default: 5)

    Returns:
        Tuple of (cleaned_df, dropped_df)
    """
    min_en_length = config.get('min_en_length', 5)
    min_ka_length = config.get('min_ka_length', 5)

    def check_row(row):
        en_len = len(str(row['en']).strip())
        ka_len = len(str(row['ka']).strip())
        return en_len >= min_en_length and ka_len >= min_ka_length

    tqdm.pandas(desc="Filtering minimum length")
    mask = df.progress_apply(check_row, axis=1)

    cleaned_df = df[mask].copy()
    dropped_df = df[~mask].copy()

    return cleaned_df, dropped_df
