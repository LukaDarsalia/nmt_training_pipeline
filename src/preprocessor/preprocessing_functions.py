"""
Text Preprocessing Functions

This module contains preprocessing functions that modify existing data in-place.
These functions clean and normalize text without changing the number of rows.
"""

import re
# Suppress warnings for cleaner output
import warnings
from typing import Dict, Any, List

import pandas as pd
from tqdm import tqdm

from .registry import register_preprocessor

warnings.filterwarnings('ignore')

# Try to import eng library for British to American conversion
try:
    import eng
    ENG_AVAILABLE = True
except ImportError:
    ENG_AVAILABLE = False
    print("Warning: 'eng' library not available. British to American conversion will be skipped.")


def _process_dataframe_columns(df: pd.DataFrame,
                              func: callable,
                              columns: List[str] = None) -> pd.DataFrame:
    """
    Helper function to apply a function to specified columns of a DataFrame.

    Args:
        df: Input DataFrame
        func: Function to apply to each cell
        columns: List of column names to process (default: ['en', 'ka'])

    Returns:
        DataFrame with processed columns
    """
    if columns is None:
        columns = ["en", "ka"]

    columns = columns if isinstance(columns, list) else [columns]
    df_copy = df.copy()

    for col in columns:
        if col in df_copy.columns:
            tqdm.pandas(desc=f"Processing {col}")
            df_copy[col] = df_copy[col].astype(str).progress_apply(func)

    return df_copy


@register_preprocessor("convert_to_american", "Convert British English to American English")
def convert_to_american(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """
    Convert British English words to American English equivalents.
    This ensures consistency and helps model training.

    Args:
        df: Input DataFrame
        config: Configuration parameters with 'columns' (default: ['en'])

    Returns:
        DataFrame with American English text
    """
    if not ENG_AVAILABLE:
        print("Warning: Skipping British to American conversion - 'eng' library not available")
        return df.copy()

    columns = config.get('columns', ['en'])

    def convert_text(text: str) -> str:
        if not text or pd.isna(text):
            return text

        try:
            # Convert each word using eng.TextFixer
            words = text.split()
            converted_words = []

            for word in words:
                try:
                    converted = eng.TextFixer(content=word).apply()
                    converted_words.append(converted)
                except Exception:
                    # If conversion fails for a word, keep original
                    converted_words.append(word)

            return ' '.join(converted_words)
        except Exception:
            return text

    return _process_dataframe_columns(df, convert_text, columns)


@register_preprocessor("normalize_characters", "Standardize dashes, apostrophes, and quotes")
def normalize_characters(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """
    Normalize various Unicode characters to standard ASCII equivalents.
    Standardizes dashes, apostrophes, and quotation marks.

    Args:
        df: Input DataFrame
        config: Configuration parameters with 'columns' (default: ['en', 'ka'])

    Returns:
        DataFrame with normalized characters
    """
    columns = config.get('columns', ['en', 'ka'])

    def normalize_text(text: str) -> str:
        if not text or pd.isna(text):
            return text

        # Normalize dashes
        dashes = ['-', '–', '—', '−']
        for dash in dashes:
            text = text.replace(dash, "-")

        # Normalize apostrophes
        apostrophes = ["'", '`', '´', ''', ''', 'ʻ']
        for apos in apostrophes:
            text = text.replace(apos, "'")

        # Normalize quotation marks
        double_quotes = ['"', '"', '"', '„', '«', '»']
        for quote in double_quotes:
            text = text.replace(quote, '"')

        return text

    return _process_dataframe_columns(df, normalize_text, columns)


@register_preprocessor("remove_extra_whitespaces", "Remove extra whitespaces and normalize spacing")
def remove_extra_whitespaces(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """
    Remove extra whitespaces, normalize spacing, and trim text.

    Args:
        df: Input DataFrame
        config: Configuration parameters with 'columns' (default: ['en', 'ka'])

    Returns:
        DataFrame with normalized whitespace
    """
    columns = config.get('columns', ['en', 'ka'])

    def clean_whitespace(text: str) -> str:
        if not text or pd.isna(text):
            return text

        # Replace multiple spaces with single space
        text = re.sub(r' {2,}', ' ', text)

        # Replace multiple newlines with single newline
        text = re.sub(r'\n{2,}', '\n', text)

        # Replace tabs with spaces
        text = text.replace('\t', ' ')

        # Strip leading and trailing whitespace
        text = text.strip()

        return text

    return _process_dataframe_columns(df, clean_whitespace, columns)


@register_preprocessor("lowercase_english", "Convert English text to lowercase")
def lowercase_english(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """
    Convert English text to lowercase. Useful for English-to-Georgian translation.

    Args:
        df: Input DataFrame
        config: Configuration parameters with 'columns' (default: ['en'])

    Returns:
        DataFrame with lowercase English text
    """
    columns = config.get('columns', ['en'])
    preserve_proper_nouns = config.get('preserve_proper_nouns', False)

    def lowercase_text(text: str) -> str:
        if not text or pd.isna(text):
            return text

        if preserve_proper_nouns:
            # Simple heuristic: preserve words that start with uppercase
            # and are likely proper nouns (not at sentence start)
            words = text.split()
            result_words = []

            for i, word in enumerate(words):
                # If it's the first word or after sentence-ending punctuation
                if i == 0 or any(words[i-1].endswith(p) for p in '.!?'):
                    result_words.append(word.lower())
                # If it starts with uppercase and might be a proper noun
                elif word[0].isupper() and len(word) > 1:
                    result_words.append(word)  # Keep as is
                else:
                    result_words.append(word.lower())

            return ' '.join(result_words)
        else:
            return text.lower()

    return _process_dataframe_columns(df, lowercase_text, columns)


@register_preprocessor("sync_punctuation", "Synchronize punctuation between English and Georgian")
def sync_punctuation(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """
    Ensure punctuation consistency between English and Georgian texts.
    If one text has ending punctuation and the other doesn't, add it.

    Args:
        df: Input DataFrame with 'en' and 'ka' columns
        config: Configuration parameters with 'strategy' ('add_missing' or 'remove_all')

    Returns:
        DataFrame with synchronized punctuation
    """
    strategy = config.get('strategy', 'add_missing')  # 'add_missing' or 'remove_all'
    punctuation_marks = config.get('punctuation_marks', '.!?;:')

    def get_ending_punctuation(text: str) -> str:
        """Get the ending punctuation of a text."""
        if not text or pd.isna(text):
            return ""

        text = text.strip()
        if text and text[-1] in punctuation_marks:
            return text[-1]
        return ""

    def add_punctuation(text: str, punct: str) -> str:
        """Add punctuation to text if it doesn't already have it."""
        if not text or pd.isna(text):
            return text

        text = text.strip()
        if text and text[-1] not in punctuation_marks and punct:
            return text + punct
        return text

    def remove_punctuation(text: str) -> str:
        """Remove ending punctuation from text."""
        if not text or pd.isna(text):
            return text

        text = text.strip()
        if text and text[-1] in punctuation_marks:
            return text[:-1]
        return text

    def sync_row(row):
        en_text = str(row['en']) if pd.notna(row['en']) else ""
        ka_text = str(row['ka']) if pd.notna(row['ka']) else ""

        en_punct = get_ending_punctuation(en_text)
        ka_punct = get_ending_punctuation(ka_text)

        if strategy == 'add_missing':
            # If one has punctuation and the other doesn't, add it
            if en_punct and not ka_punct:
                ka_text = add_punctuation(ka_text, en_punct)
            elif ka_punct and not en_punct:
                en_text = add_punctuation(en_text, ka_punct)
        elif strategy == 'remove_all':
            # Remove all ending punctuation
            en_text = remove_punctuation(en_text)
            ka_text = remove_punctuation(ka_text)

        return pd.Series({'en': en_text, 'ka': ka_text})

    # Apply synchronization
    tqdm.pandas(desc="Synchronizing punctuation")
    synced = df.progress_apply(sync_row, axis=1)

    # Update the dataframe
    result_df = df.copy()
    result_df['en'] = synced['en']
    result_df['ka'] = synced['ka']

    return result_df


@register_preprocessor("remove_brackets_content", "Remove content within brackets and parentheses")
def remove_brackets_content(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """
    Remove content within brackets, parentheses, or other specified delimiters.
    Useful for removing editorial notes, references, etc.

    Args:
        df: Input DataFrame
        config: Configuration with 'columns', 'bracket_types' (['()'], ['[]'], ['{}'])

    Returns:
        DataFrame with bracket content removed
    """
    columns = config.get('columns', ['en', 'ka'])
    bracket_types = config.get('bracket_types', ['()'])  # ['()', '[]', '{}']

    def remove_brackets(text: str) -> str:
        if not text or pd.isna(text):
            return text

        for bracket_type in bracket_types:
            if bracket_type == '()':
                text = re.sub(r'\([^)]*\)', '', text)
            elif bracket_type == '[]':
                text = re.sub(r'\[[^\]]*\]', '', text)
            elif bracket_type == '{}':
                text = re.sub(r'\{[^}]*\}', '', text)
            elif bracket_type == '<>':
                text = re.sub(r'<[^>]*>', '', text)

        # Clean up extra spaces
        text = re.sub(r' {2,}', ' ', text)
        text = text.strip()

        return text

    return _process_dataframe_columns(df, remove_brackets, columns)


@register_preprocessor("fix_encoding_issues", "Fix common encoding issues and special characters")
def fix_encoding_issues(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """
    Fix common encoding issues and normalize special characters.

    Args:
        df: Input DataFrame
        config: Configuration parameters with 'columns'

    Returns:
        DataFrame with fixed encoding issues
    """
    columns = config.get('columns', ['en', 'ka'])

    def fix_text(text: str) -> str:
        if not text or pd.isna(text):
            return text

        # Fix common encoding issues
        replacements = {
            'â€™': "'",      # Right single quotation mark
            'â€œ': '"',      # Left double quotation mark
            'â€': '"',       # Right double quotation mark
            'â€"': '—',      # Em dash
            'â€"': '–',      # En dash
            'Ã¡': 'á',       # a with acute
            'Ã©': 'é',       # e with acute
            'Ã­': 'í',       # i with acute
            'Ã³': 'ó',       # o with acute
            'Ãº': 'ú',       # u with acute
            'â€¦': '...',    # Ellipsis
            'Â': '',         # Non-breaking space artifacts
            'â€': '',        # Various artifacts
        }

        for bad, good in replacements.items():
            text = text.replace(bad, good)

        # Remove or replace other problematic characters
        text = text.replace('\ufeff', '')  # BOM
        text = text.replace('\u200b', '')  # Zero width space
        text = text.replace('\u200c', '')  # Zero width non-joiner
        text = text.replace('\u200d', '')  # Zero width joiner

        return text

    return _process_dataframe_columns(df, fix_text, columns)
