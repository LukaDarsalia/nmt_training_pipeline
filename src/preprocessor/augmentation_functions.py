"""
Data Augmentation Functions

This module contains augmentation functions that create new data rows.
These functions generate additional training examples to improve model robustness.
"""

import random
import re
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from .registry import register_augmenter

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)


def _generate_char_list(ranges: List[str]) -> List[str]:
    """
    Generate character list from ranges.

    Args:
        ranges: List of character ranges like ['a-z', 'A-Z', 'ა-ჰ']

    Returns:
        List of characters
    """
    char_list = []
    for r in ranges:
        if '-' in r and len(r) == 3:  # Format like 'a-z'
            start, end = r.split('-')
            char_list.extend([chr(c) for c in range(ord(start), ord(end) + 1)])
        else:
            char_list.extend(list(r))
    return char_list


def _apply_synthetic_noise(text: str, chars: List[str], noise_types: List[str], p: float = 0.1) -> str:
    """
    Apply synthetic noise to text.

    Args:
        text: Input text
        chars: Available characters for noise
        noise_types: Types of noise to apply
        p: Probability of applying noise

    Returns:
        Text with applied noise
    """
    if not text or random.random() >= p:
        return text

    augmented_text = text
    noise_type = random.choice(noise_types)

    if noise_type == 'delete' and len(augmented_text) > 1:
        idx = random.randint(0, len(augmented_text) - 1)
        augmented_text = augmented_text[:idx] + augmented_text[idx + 1:]

    elif noise_type == 'insert' and len(augmented_text) > 0:
        idx = random.randint(0, len(augmented_text))
        augmented_text = augmented_text[:idx] + random.choice(chars) + augmented_text[idx:]

    elif noise_type == 'duplicate' and len(augmented_text) > 0:
        idx = random.randint(0, len(augmented_text) - 1)
        augmented_text = augmented_text[:idx] + augmented_text[idx] + augmented_text[idx:]

    elif noise_type == 'substitute' and len(augmented_text) > 0:
        idx = random.randint(0, len(augmented_text) - 1)
        augmented_text = augmented_text[:idx] + random.choice(chars) + augmented_text[idx + 1:]

    elif noise_type == 'swap' and len(augmented_text) > 1:
        idx = random.randint(0, len(augmented_text) - 2)
        augmented_text = (augmented_text[:idx] +
                         augmented_text[idx + 1] +
                         augmented_text[idx] +
                         augmented_text[idx + 2:])

    return augmented_text


@register_augmenter("synthetic_noise", "Add synthetic noise to text for robustness")
def synthetic_noise_augmentation(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """
    Apply synthetic noise to text to improve model robustness.

    Args:
        df: Input DataFrame
        config: Configuration with 'percentage', 'noise_types', 'char_ranges', 'target_column'

    Returns:
        DataFrame with noise-augmented examples
    """
    percentage = config.get('percentage', 0.1)
    noise_types = config.get('noise_types', ['delete', 'insert', 'duplicate', 'substitute', 'swap'])
    char_ranges = config.get('char_ranges', ['a-z'])
    target_column = config.get('target_column', 'en')  # Which column to add noise to

    chars = _generate_char_list(char_ranges)
    augmented_data = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Applying synthetic noise"):
        if random.random() < percentage:
            new_row = row.copy()
            new_row[target_column] = _apply_synthetic_noise(
                str(row[target_column]), chars, noise_types, p=1.0
            )
            new_row['id'] = f"{row['id']}_synthetic_noise"
            augmented_data.append(new_row)

    return pd.DataFrame(augmented_data)


@register_augmenter("sentence_concatenation", "Concatenate consecutive sentences")
def sentence_concatenation_augmentation(
    df: pd.DataFrame,
    config: Dict[str, Any]
) -> pd.DataFrame:
    """
    Create augmented examples by concatenating *n* consecutive sentences.

    Args:
        df:      DataFrame with at least the columns ['en', 'ka', 'id'].
        config:  {
            'percentage': float,   # Probability to start a concatenation at a given row. (default 0.1)
            'separator':  str,     # Token inserted between sentences.          (default ' ')
            'min_n':      int,     # Minimum number of rows to concatenate.     (default 2)
            'max_n':      int      # Maximum number of rows to concatenate.     (default = min_n)
        }

    Returns:
        DataFrame containing the new, concatenated examples.
    """
    import random
    percentage = config.get('percentage', 0.1)
    separator  = config.get('separator', ' ')
    min_n      = config.get('min_n', 2)
    max_n      = config.get('max_n', min_n)

    assert min_n >= 2 and max_n >= min_n, \
        "`min_n` must be ≥ 2 and `max_n` must be ≥ min_n."

    augmented_data = []
    total_len = len(df)

    for idx, row in tqdm(df.iterrows(),
                         total=total_len,
                         desc="Concatenating sentences"):
        # decide whether to build an augmented example that starts at this row
        if random.random() < percentage:
            n = random.randint(min_n, max_n)
            n = min(n, total_len - idx)          # keep slice inside the frame

            rows_slice = df.iloc[idx: idx + n]   # the n consecutive rows

            new_row = row.copy()
            new_row['en'] = separator.join(rows_slice['en'].astype(str))
            new_row['ka'] = separator.join(rows_slice['ka'].astype(str))
            new_row['id'] = f"{row['id']}_concat{n}"

            augmented_data.append(new_row)

    return pd.DataFrame(augmented_data)


@register_augmenter("number_copying", "Generate number copying examples")
def number_copying_augmentation(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """
    Generate examples where both English and Georgian contain the same numbers.
    Helps the model learn to copy numbers correctly.

    Args:
        df: Input DataFrame
        config: Configuration with 'num_examples', 'number_types'

    Returns:
        DataFrame with number copying examples
    """
    num_examples = config.get('num_examples', 100)
    number_types = config.get('number_types', ['integer', 'float', 'large', 'small'])

    augmented_data = []

    def generate_number(number_type: str) -> str:
        """Generate different types of numbers."""
        if number_type == 'integer':
            return str(random.randint(1, 10000))
        elif number_type == 'float':
            return f"{random.uniform(1, 1000):.2f}"
        elif number_type == 'large':
            return str(random.randint(1000000, 999999999))
        elif number_type == 'small':
            return f"{random.uniform(0.001, 0.999):.3f}"
        else:
            return str(random.randint(1, 100))

    # Paired templates - each tuple contains (English, Georgian) translations
    template_pairs = [
        ("The number is {number}.", "რიცხვი არის {number}."),
        ("Value: {number}", "მნიშვნელობა: {number}"),
        ("Result equals {number}", "შედეგი ტოლია {number}"),
        ("Count: {number}", "რაოდენობა: {number}"),
        ("Total: {number}", "სრული: {number}"),
        ("Amount: {number}", "თანხა: {number}"),
        ("Score: {number}", "ქულა: {number}"),
        ("Price: ${number}", "ფასი: ${number}"),
        ("Year: {number}", "წელი: {number}"),
        ("ID: {number}", "ID: {number}"),
        ("Temperature: {number}°C", "ტემპერატურა: {number}°C"),
        ("Distance: {number} km", "მანძილი: {number} კმ"),
        ("Weight: {number} kg", "წონა: {number} კგ"),
        ("Speed: {number} km/h", "სიჩქარე: {number} კმ/სთ"),
        ("Percentage: {number}%", "პროცენტი: {number}%")
    ]

    for i in tqdm(range(num_examples), desc="Generating number copying examples"):
        number_type = random.choice(number_types)
        number = generate_number(number_type)

        # Select a random template pair (ensures EN and KA are translations)
        en_template, ka_template = random.choice(template_pairs)

        new_row = {
            'title': None,
            'en': en_template.format(number=number),
            'ka': ka_template.format(number=number),
            'domain': 'synthetic',
            'id': f"number_copy_{i}"
        }

        augmented_data.append(new_row)

    return pd.DataFrame(augmented_data)


@register_augmenter("georgian_text_copying", "Generate examples where English equals Georgian text")
def georgian_text_copying_augmentation(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """
    Generate examples where English and Georgian texts are identical.
    Always samples Georgian text from the current dataset to ensure relevance.
    Helps model learn to copy Georgian text when appropriate.

    Args:
        df: Input DataFrame
        config: Configuration with 'percentage' or 'num_examples'

    Returns:
        DataFrame with Georgian text copying examples
    """
    # Get percentage or number of examples
    percentage = config.get('percentage', 0.05)
    num_examples = config.get('num_examples', None)

    # Always sample Georgian texts from the current dataset
    georgian_texts = df['ka'].dropna().astype(str).tolist()

    if not georgian_texts:
        print("Warning: No Georgian texts found in dataset for copying")
        return pd.DataFrame()

    # Remove empty or very short texts
    georgian_texts = [text.strip() for text in georgian_texts if text.strip() and len(text.strip()) > 2]

    if not georgian_texts:
        print("Warning: No suitable Georgian texts found for copying")
        return pd.DataFrame()

    # Determine number of examples to create
    if num_examples is not None:
        target_examples = num_examples
    else:
        target_examples = int(len(df) * percentage)

    # Ensure we don't exceed available texts
    target_examples = min(target_examples, len(georgian_texts))

    if target_examples == 0:
        print("Warning: No examples will be generated (target_examples = 0)")
        return pd.DataFrame()

    augmented_data = []

    # Sample unique Georgian texts (without replacement if possible)
    if target_examples <= len(georgian_texts):
        sampled_texts = random.sample(georgian_texts, target_examples)
    else:
        # If we need more examples than unique texts, sample with replacement
        sampled_texts = [random.choice(georgian_texts) for _ in range(target_examples)]

    for i, georgian_text in enumerate(tqdm(sampled_texts, desc="Generating Georgian copying examples")):
        new_row = {
            'title': None,
            'en': georgian_text,  # English = Georgian text
            'ka': georgian_text,  # Georgian text stays the same
            'domain': 'synthetic_copying',
            'id': f"georgian_copy_{i}"
        }

        augmented_data.append(new_row)

    print(f"  Generated {len(augmented_data)} Georgian copying examples from dataset texts")
    return pd.DataFrame(augmented_data)


@register_augmenter("restructure_to_longer_texts", "DATASET RESTRUCTURING: Combine multiple rows into fewer longer texts")
def restructure_to_longer_texts_augmentation(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """
    DATASET RESTRUCTURING OPERATION: Combine multiple consecutive rows to create fewer, longer texts.
    This REDUCES the total number of rows while creating longer training examples.

    Purpose: Improve translation quality for long texts by better covering the positional embedding space.

    Note: This is a dataset restructuring operation, not traditional augmentation.
    It will reduce your dataset size while creating longer, more comprehensive examples.

    Args:
        df: Input DataFrame
        config: Configuration with 'concatenation_ratio', 'min_group_size', 'max_group_size', 'separator'

    Returns:
        DataFrame with concatenated longer texts (this replaces the original dataset structure)
    """
    concatenation_ratio = config.get('concatenation_ratio', 0.3)  # 30% of rows will be part of concatenations
    min_group_size = config.get('min_group_size', 2)  # Minimum rows to concatenate
    max_group_size = config.get('max_group_size', 4)  # Maximum rows to concatenate
    separator = config.get('separator', ' ')  # Separator between concatenated texts
    strategy = config.get('strategy', 'consecutive')  # 'consecutive' or 'random'

    if len(df) < min_group_size:
        print(f"    Warning: Dataset too small for concatenation (size: {len(df)})")
        return df.copy()

    df_copy = df.copy().reset_index(drop=True)

    # Calculate how many rows should be affected by concatenation
    total_rows = len(df_copy)
    rows_to_concatenate = int(total_rows * concatenation_ratio)

    if rows_to_concatenate < min_group_size:
        print(f"    Warning: Too few rows for concatenation (would affect {rows_to_concatenate} rows)")
        return df_copy

    # Track which rows have been used
    used_indices = set()
    concatenated_rows = []
    remaining_rows = []

    if strategy == 'consecutive':
        i = 0
        tqdm_desc = "Restructuring dataset with longer texts"

        with tqdm(total=total_rows, desc=tqdm_desc) as pbar:
            while i < total_rows:
                if i in used_indices:
                    i += 1
                    pbar.update(1)
                    continue

                # Decide if we should start a concatenation group here
                remaining_budget = rows_to_concatenate - len(used_indices)
                if remaining_budget >= min_group_size and random.random() < 0.5:

                    # Determine group size
                    max_possible = min(max_group_size, total_rows - i, remaining_budget)
                    if max_possible >= min_group_size:
                        group_size = random.randint(min_group_size, max_possible)

                        # Get consecutive rows
                        group_indices = list(range(i, min(i + group_size, total_rows)))
                        group_rows = [df_copy.iloc[idx] for idx in group_indices]

                        # Create concatenated row
                        concatenated_row = group_rows[0].copy()
                        concatenated_en = separator.join([str(row['en']) for row in group_rows])
                        concatenated_ka = separator.join([str(row['ka']) for row in group_rows])
                        concatenated_ids = [str(row['id']) for row in group_rows]

                        concatenated_row['en'] = concatenated_en
                        concatenated_row['ka'] = concatenated_ka
                        concatenated_row['id'] = f"restructured_{'_'.join(concatenated_ids[:3])}"  # Limit ID length

                        concatenated_rows.append(concatenated_row)
                        used_indices.update(group_indices)

                        i += group_size
                        pbar.update(group_size)
                    else:
                        remaining_rows.append(df_copy.iloc[i])
                        i += 1
                        pbar.update(1)
                else:
                    remaining_rows.append(df_copy.iloc[i])
                    i += 1
                    pbar.update(1)

    elif strategy == 'random':
        # Random grouping strategy
        available_indices = list(range(total_rows))
        random.shuffle(available_indices)

        i = 0
        with tqdm(total=rows_to_concatenate, desc="Restructuring with random grouping") as pbar:
            while i < len(available_indices) and len(used_indices) < rows_to_concatenate:
                remaining_budget = rows_to_concatenate - len(used_indices)

                if remaining_budget >= min_group_size:
                    # Determine group size
                    max_possible = min(max_group_size, remaining_budget, len(available_indices) - i)

                    if max_possible >= min_group_size:
                        group_size = random.randint(min_group_size, max_possible)

                        # Get random rows
                        group_indices = available_indices[i:i+group_size]
                        group_indices.sort()  # Sort for consistent ordering
                        group_rows = [df_copy.iloc[idx] for idx in group_indices]

                        # Create concatenated row
                        concatenated_row = group_rows[0].copy()
                        concatenated_en = separator.join([str(row['en']) for row in group_rows])
                        concatenated_ka = separator.join([str(row['ka']) for row in group_rows])
                        concatenated_ids = [str(row['id']) for row in group_rows]

                        concatenated_row['en'] = concatenated_en
                        concatenated_row['ka'] = concatenated_ka
                        concatenated_row['id'] = f"restructured_{'_'.join(concatenated_ids[:3])}"

                        concatenated_rows.append(concatenated_row)
                        used_indices.update(group_indices)

                        i += group_size
                        pbar.update(group_size)
                    else:
                        break
                else:
                    break

        # Add remaining unused rows
        for idx in available_indices:
            if idx not in used_indices:
                remaining_rows.append(df_copy.iloc[idx])

    # Combine concatenated and remaining rows
    all_rows = concatenated_rows + remaining_rows

    if not all_rows:
        print("    Warning: No rows produced by restructuring")
        return df_copy

    result_df = pd.DataFrame(all_rows).reset_index(drop=True)

    # Statistics
    num_concatenated_groups = len(concatenated_rows)
    num_original_rows_used = len(used_indices)
    compression_ratio = len(result_df) / len(df_copy) if len(df_copy) > 0 else 1

    print(f"    ✓ DATASET RESTRUCTURED: {num_original_rows_used} rows → {num_concatenated_groups} longer texts")
    print(f"    ✓ Total dataset: {len(df_copy)} → {len(result_df)} rows (compression: {compression_ratio:.2f}x)")

    return result_df


@register_augmenter("natural_writing_variations", "Simulate natural human writing variations and common errors")
def natural_writing_variations_augmentation(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """
    Simulate natural human writing variations and common errors to improve model robustness.
    This includes typical variations like word order changes, missing articles, etc.

    Args:
        df: Input DataFrame
        config: Configuration with 'percentage', 'variation_types'

    Returns:
        DataFrame with natural writing variations
    """
    percentage = config.get('percentage', 0.05)
    variation_types = config.get('variation_types', ['word_order', 'article_errors', 'informal_style'])

    augmented_data = []

    def introduce_word_order_changes(text: str) -> str:
        """Introduce word order changes typical in casual writing."""
        words = text.split()
        if len(words) > 2:
            # Randomly swap adjacent words
            idx = random.randint(0, len(words) - 2)
            words[idx], words[idx + 1] = words[idx + 1], words[idx]
        return ' '.join(words)

    def introduce_article_errors(text: str) -> str:
        """Remove articles, simulating casual or non-native writing."""
        # Remove articles randomly (as people sometimes do in casual writing)
        # Only remove some articles, not all
        words = text.split()
        result_words = []
        for word in words:
            if word.lower() in ['a', 'an', 'the'] and random.random() < 0.7:
                continue  # Skip this article
            result_words.append(word)
        return ' '.join(result_words)

    def introduce_informal_style(text: str) -> str:
        """Add informal writing patterns."""
        # Simple contractions and informal patterns
        informal_replacements = {
            'and': '&',
            'you are': "you're",
            'do not': "don't",
            'cannot': "can't",
            'will not': "won't",
            'it is': "it's",
            'that is': "that's",
            'I am': "I'm"
        }

        result = text
        for formal, informal in informal_replacements.items():
            # Case-insensitive replacement
            pattern = re.compile(r'\b' + re.escape(formal) + r'\b', re.IGNORECASE)
            result = pattern.sub(informal, result)

        return result

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Adding natural writing variations"):
        if random.random() < percentage:
            new_row = row.copy()

            variation_type = random.choice(variation_types)

            if variation_type == 'word_order':
                new_row['en'] = introduce_word_order_changes(str(row['en']))
            elif variation_type == 'article_errors':
                new_row['en'] = introduce_article_errors(str(row['en']))
            elif variation_type == 'informal_style':
                new_row['en'] = introduce_informal_style(str(row['en']))

            new_row['id'] = f"{row['id']}_natural_variation"
            augmented_data.append(new_row)

    return pd.DataFrame(augmented_data)