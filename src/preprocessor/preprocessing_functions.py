"""
Text Preprocessing Functions

This module contains preprocessing functions that modify existing data in-place.
These functions clean and normalize text without changing the number of rows.
"""
import html
import re
# Suppress warnings for cleaner output
import warnings
from typing import Dict, Any, List, Optional

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

    # Pre-compile replacement mapping for efficiency
    replacements = {
        # Dashes
        '–': '-',  # En dash
        '—': '-',  # Em dash
        '−': '-',  # Minus sign
        '‐': '-',  # Hyphen
        '‑': '-',  # Non-breaking hyphen

        # Apostrophes and single quotes
        '`': "'",  # Grave accent
        '´': "'",  # Acute accent
        'ʻ': "'",  # Modifier letter turned comma
        'ʼ': "'",  # Modifier letter apostrophe

        # Double quotes
        '„': '"',  # Double low-9 quotation mark
        '«': '"',  # Left-pointing double angle quotation mark
        '»': '"',  # Right-pointing double angle quotation mark
        '‟': '"',  # Double high-reversed-9 quotation mark

        # Other punctuation
        '…': '...',  # Horizontal ellipsis
    }

    def normalize_text(text: str) -> str:
        if not text or pd.isna(text) or text == 'nan':
            return text

        # Apply all replacements
        for old, new in replacements.items():
            text = text.replace(old, new)

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

    # Pre-compile regex patterns for efficiency
    multiple_spaces = re.compile(r' {2,}')
    multiple_newlines = re.compile(r'\n{2,}')
    tab_pattern = re.compile(r'\t+')

    def clean_whitespace(text: str) -> str:
        if not text or pd.isna(text) or text == 'nan':
            return text

        # Replace multiple spaces with single space
        text = multiple_spaces.sub(' ', text)

        # Replace multiple newlines with single newline
        text = multiple_newlines.sub('\n', text)

        # Replace tabs with spaces
        text = tab_pattern.sub(' ', text)

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

    # Pre-compile pattern for sentence endings
    sentence_end_pattern = re.compile(r'[.!?]\s*$')

    def lowercase_text(text: str) -> str:
        if not text or pd.isna(text) or text == 'nan':
            return text

        if preserve_proper_nouns:
            # Simple heuristic: preserve words that start with uppercase
            # and are likely proper nouns (not at sentence start)
            words = text.split()
            result_words = []

            for i, word in enumerate(words):
                # If it's the first word or after sentence-ending punctuation
                if i == 0 or (i > 0 and sentence_end_pattern.search(words[i-1])):
                    result_words.append(word.lower())
                # If it starts with uppercase and might be a proper noun
                elif word and word[0].isupper() and len(word) > 1:
                    # Check if entire word is uppercase (acronym)
                    if word.isupper():
                        result_words.append(word)  # Keep acronyms as is
                    else:
                        result_words.append(word)  # Keep proper nouns
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
    If both texts end with different punctuation marks, replace the Georgian one
    (or preferred language) to match the English punctuation (default behaviour).

    Args:
        df: Input DataFrame with 'en' and 'ka' columns
        config: Configuration parameters
            strategy: 'sync' (default) | 'remove_all'
            punctuation_marks: iterable of accepted punctuation marks
            prefer: which language to treat as authoritative if both have
                    punctuation but they differ ('en' or 'ka')
    Returns:
        DataFrame with synchronized punctuation
    """
    strategy = config.get('strategy', 'sync')          # 'sync' or 'remove_all'
    punctuation_marks = config.get(
        'punctuation_marks',
        '.!?;:,'
    )
    prefer_lang = config.get('prefer', 'en')           # 'en' or 'ka'

    # Pre‑compile regex for trailing quotes/brackets/spaces
    _trail_re = re.compile(r'[\s\'"”’»\)\]\}]*$')

    def _split_trailing(text: str):
        """Return (core, trailing) where trailing = quotes/spaces at end."""
        m = _trail_re.search(text)
        trailing = m.group(0) if m else ''
        core = text[:-len(trailing)] if trailing else text
        return core, trailing

    def _get_end_punct(text: str) -> str:
        """Return last punctuation mark (if any) ignoring trailing quotes."""
        if not text or pd.isna(text):
            return ''
        core, _ = _split_trailing(text.strip())
        return core[-1] if core and core[-1] in punctuation_marks else ''

    def _replace_or_add_punct(text: str, punct: str) -> str:
        """Ensure text ends with punct (handling quotes)."""
        if not text or pd.isna(text):
            return text
        text = str(text).rstrip()
        core, trailing = _split_trailing(text)
        if core and core[-1] in punctuation_marks:
            core = core[:-1] + punct   # replace
        else:
            core = core + punct        # add
        return core + trailing

    def _remove_punct(text: str) -> str:
        """Remove ending punctuation (before trailing quotes)."""
        if not text or pd.isna(text):
            return text
        text = str(text).rstrip()
        core, trailing = _split_trailing(text)
        if core and core[-1] in punctuation_marks:
            core = core[:-1]
        return core + trailing

    def _sync_row(row):
        en_txt, ka_txt = str(row['en']), str(row['ka'])
        en_punct, ka_punct = _get_end_punct(en_txt), _get_end_punct(ka_txt)

        if strategy == 'remove_all':
            en_txt, ka_txt = _remove_punct(en_txt), _remove_punct(ka_txt)

        else:  # 'sync'
            # When both missing nothing to do
            if not en_punct and not ka_punct:
                pass
            # One side missing – copy from the other
            elif en_punct and not ka_punct:
                ka_txt = _replace_or_add_punct(ka_txt, en_punct)
            elif ka_punct and not en_punct:
                en_txt = _replace_or_add_punct(en_txt, ka_punct)
            # Both present but differ – enforce preferred language punctuation
            elif en_punct != ka_punct:
                if prefer_lang == 'en':
                    ka_txt = _replace_or_add_punct(ka_txt, en_punct)
                else:
                    en_txt = _replace_or_add_punct(en_txt, ka_punct)

        return pd.Series({'en': en_txt, 'ka': ka_txt})

    tqdm.pandas(desc="Synchronising punctuation")
    synced = df.progress_apply(_sync_row, axis=1)
    result_df = df.copy()
    result_df[['en', 'ka']] = synced[['en', 'ka']]

    return result_df


@register_preprocessor("remove_brackets_content", "Remove content within brackets and parentheses")
def remove_brackets_content(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """
    Remove bracketed content in the specified columns.
    Optionally limit the operation to rows whose `id` starts with one of the
    dataset names given in config['datasets'].

    Extra config keys:
        - domains: List[str] of dataset prefixes (e.g. ['ლექსიკოგრაფია']).

    Other keys are unchanged:
        - columns
        - bracket_types
    """
    columns       = config.get("columns", ["en", "ka"])
    bracket_types = config.get("bracket_types", ["()"])      # ['()', '[]', '{}', '<>']
    domains      = set(config.get("domains", []))          # new

    # ── regex compile ───────────────────────────────────────────────────────────
    patterns = {}
    for bt in bracket_types:
        if bt == "()":
            patterns[bt] = re.compile(r"\([^)]*\)")
        elif bt == "[]":
            patterns[bt] = re.compile(r"\[[^\]]*\]")
        elif bt == "{}":
            patterns[bt] = re.compile(r"\{[^}]*\}")
        elif bt == "<>":
            patterns[bt] = re.compile(r"<[^>]*>")

    whitespace_pattern = re.compile(r" {2,}")

    def remove_brackets(text: str) -> str:
        if not text or pd.isna(text) or text == "nan":
            return text
        for pat in patterns.values():
            text = pat.sub("", text)
        return whitespace_pattern.sub(" ", text).strip()

    # ── apply ───────────────────────────────────────────────────────────────────
    if domains:
        if "id" not in df.columns:
            raise ValueError("Column 'id' is required when using 'datasets'.")
        # mask rows whose id prefix matches any dataset name
        print(domains)
        mask = df["domain"].astype(str).isin(domains)
        df_out = df.copy()
        df_out.loc[mask, columns] = _process_dataframe_columns(
            df_out.loc[mask], remove_brackets, columns
        )[columns]
        return df_out

    # fall-back: process the whole DataFrame
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

    # Common encoding issues and their fixes
    replacements = {
        # Common UTF-8 decode errors
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
        'Ã±': 'ñ',       # n with tilde
        'Ã¼': 'ü',       # u with diaeresis
        'Ã¤': 'ä',       # a with diaeresis
        'Ã¶': 'ö',       # o with diaeresis
        'Ã§': 'ç',       # c with cedilla
        'Ã¨': 'è',       # e with grave
        'Ã ': 'à',       # a with grave
        'â€¦': '...',    # Ellipsis
        'Â': '',         # Non-breaking space artifacts
        'â€': '',        # Various artifacts
        'Ââ': '',        # Double artifacts
        '&nbsp;': ' ',   # HTML non-breaking space
        '&amp;': '&',    # HTML ampersand
        '&lt;': '<',     # HTML less than
        '&gt;': '>',     # HTML greater than
        '&quot;': '"',   # HTML quote
        '&#39;': "'",    # HTML apostrophe
    }

    def fix_text(text: str) -> str:
        if not text or pd.isna(text) or text == 'nan':
            return text

        # Apply all replacements
        for bad, good in replacements.items():
            text = text.replace(bad, good)

        # Remove zero-width characters
        zero_width_chars = [
            '\ufeff',  # BOM
            '\u200b',  # Zero width space
            '\u200c',  # Zero width non-joiner
            '\u200d',  # Zero width joiner
            '\u200e',  # Left-to-right mark
            '\u200f',  # Right-to-left mark
            '\u202a',  # Left-to-right embedding
            '\u202b',  # Right-to-left embedding
            '\u202c',  # Pop directional formatting
            '\u202d',  # Left-to-right override
            '\u202e',  # Right-to-left override
        ]

        for char in zero_width_chars:
            text = text.replace(char, '')

        # Normalize whitespace after cleaning
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()

        return text

    return _process_dataframe_columns(df, fix_text, columns)


@register_preprocessor("remove_html_tags", "Remove HTML tags from text")
def remove_html_tags(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """
    Remove HTML tags and decode HTML entities.

    Args:
        df: Input DataFrame
        config: Configuration parameters with 'columns'

    Returns:
        DataFrame with HTML tags removed
    """
    columns = config.get("columns", ["en", "ka"])

    tag_re = re.compile(r'</?[A-Za-z][A-Za-z0-9]*(?:\s[^<>]*?)?/?>')
    ws_re = re.compile(r'\s+')

    def clean(text: str) -> str:
        if not text or pd.isna(text) or str(text).lower() == "nan":
            return text
        text = tag_re.sub(" ", str(text))  # drop real tags
        text = html.unescape(text)  # decode &nbsp; etc.
        return ws_re.sub(" ", text).strip()  # normalise spaces

    return _process_dataframe_columns(df, clean, columns)