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
    char_ranges = config.get('char_ranges', ['a-z', 'A-Z', 'ა-ჰ'])
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


@register_augmenter("sentence_concatenation", "Concatenate sentence pairs")
def sentence_concatenation_augmentation(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """
    Create augmented examples by concatenating consecutive sentence pairs.

    Args:
        df: Input DataFrame
        config: Configuration with 'percentage', 'separator'

    Returns:
        DataFrame with concatenated sentence examples
    """
    percentage = config.get('percentage', 0.1)
    separator = config.get('separator', ' ')

    augmented_data = []

    for idx, row in tqdm(df.iterrows(), total=len(df) - 1, desc="Concatenating sentences"):
        if idx < len(df) - 1 and random.random() < percentage:
            next_row = df.iloc[idx + 1]

            new_row = row.copy()
            new_row['en'] = str(row['en']) + separator + str(next_row['en'])
            new_row['ka'] = str(row['ka']) + separator + str(next_row['ka'])
            new_row['id'] = f"{row['id']}_concatenation"

            augmented_data.append(new_row)

    return pd.DataFrame(augmented_data)





@register_augmenter("sentence_reversal", "Reverse word order in target sentences")
def sentence_reversal_augmentation(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """
    Create augmented examples by reversing word order in target sentences.

    Args:
        df: Input DataFrame
        config: Configuration with 'percentage', 'target_column'

    Returns:
        DataFrame with reversed sentence examples
    """
    percentage = config.get('percentage', 0.1)
    target_column = config.get('target_column', 'ka')

    def reverse_sentence(text: str) -> str:
        """Reverse word order in sentence."""
        return ' '.join(text.split()[::-1])

    augmented_data = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Reversing sentences"):
        if random.random() < percentage:
            new_row = row.copy()
            new_row[target_column] = reverse_sentence(str(row[target_column]))
            new_row['id'] = f"{row['id']}_reversed"

            augmented_data.append(new_row)

    return pd.DataFrame(augmented_data)


@register_augmenter("random_word_replacement", "Replace aligned words with random alternatives")
def random_word_replacement_augmentation(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """
    Create augmented examples by replacing aligned words with random alternatives.

    Args:
        df: Input DataFrame
        config: Configuration with 'percentage', 'lexicon'

    Returns:
        DataFrame with random word replacement examples
    """
    percentage = config.get('percentage', 0.1)
    lexicon = config.get('lexicon', {})

    if not lexicon:
        print("Warning: No lexicon provided for random word replacement")
        return pd.DataFrame()

    def replace_with_random_word(source: str, target: str, lexicon: Dict[str, str]) -> Tuple[str, str]:
        """Replace aligned words with random alternatives."""
        source_words = source.split()
        target_words = target.split()

        # Find aligned words using lexicon
        aligned_words = [
            (sw, lexicon[sw.lower()])
            for sw in source_words
            if sw.lower() in lexicon and lexicon[sw.lower()] in target_words
        ]

        if aligned_words:
            src_word, tgt_word = random.choice(aligned_words)
            random_src_word = random.choice(list(lexicon.keys()))
            random_tgt_word = lexicon[random_src_word]

            # Replace the words
            source = source.replace(src_word, random_src_word, 1)
            target = target.replace(tgt_word, random_tgt_word, 1)

        return source, target

    augmented_data = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Replacing random words"):
        if random.random() < percentage:
            new_en, new_ka = replace_with_random_word(
                str(row['en']), str(row['ka']), lexicon
            )

            new_row = row.copy()
            new_row['en'] = new_en
            new_row['ka'] = new_ka
            new_row['id'] = f"{row['id']}_random_replace"

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

    templates_en = [
        "The number is {number}.",
        "Value: {number}",
        "Result equals {number}",
        "Count: {number}",
        "Total: {number}",
        "Amount: {number}",
        "Score: {number}",
        "Price: ${number}",
        "Year: {number}",
        "ID: {number}"
    ]

    templates_ka = [
        "რიცხვი არის {number}.",
        "მნიშვნელობა: {number}",
        "შედეგი ტოლია {number}",
        "რაოდენობა: {number}",
        "სრული: {number}",
        "თანხა: {number}",
        "ქულა: {number}",
        "ფასი: ${number}",
        "წელი: {number}",
        "ID: {number}"
    ]

    for i in tqdm(range(num_examples), desc="Generating number copying examples"):
        number_type = random.choice(number_types)
        number = generate_number(number_type)

        en_template = random.choice(templates_en)
        ka_template = random.choice(templates_ka)

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


@register_augmenter("concatenate_long_texts", "Concatenate multiple rows to create longer texts")
def concatenate_long_texts_augmentation(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """
    Create a new dataset by concatenating multiple consecutive rows to create longer texts.
    This helps improve translation quality for long texts by better covering
    the positional embedding space. Returns the restructured dataset.

    Args:
        df: Input DataFrame
        config: Configuration with 'concatenation_ratio', 'min_group_size', 'max_group_size', 'separator'

    Returns:
        DataFrame with concatenated longer texts (this replaces the original dataset)
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
        tqdm_desc = "Concatenating consecutive rows"

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
                        concatenated_row['id'] = f"concat_{'_'.join(concatenated_ids[:3])}"  # Limit ID length

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
        with tqdm(total=rows_to_concatenate, desc="Concatenating random rows") as pbar:
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
                        concatenated_row['id'] = f"concat_{'_'.join(concatenated_ids[:3])}"

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
        print("    Warning: No rows produced by concatenation")
        return df_copy

    result_df = pd.DataFrame(all_rows).reset_index(drop=True)

    # Statistics
    num_concatenated_groups = len(concatenated_rows)
    num_original_rows_used = len(used_indices)
    compression_ratio = len(result_df) / len(df_copy) if len(df_copy) > 0 else 1

    print(f"    Created {num_concatenated_groups} concatenated groups from {num_original_rows_used} original rows")
    print(f"    Dataset size: {len(df_copy)} → {len(result_df)} (compression: {compression_ratio:.2f}x)")

    return result_df


@register_augmenter("simulate_translation_artifacts", "Simulate common machine translation errors and artifacts")
def simulate_translation_artifacts_augmentation(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """
    Simulate common machine translation artifacts and errors to improve model robustness.
    This includes typical errors like word order changes, missing articles, etc.

    Args:
        df: Input DataFrame
        config: Configuration with 'percentage', 'artifact_types'

    Returns:
        DataFrame with simulated translation artifacts
    """
    percentage = config.get('percentage', 0.05)
    artifact_types = config.get('artifact_types', ['word_order', 'article_errors', 'literal_translation'])

    augmented_data = []

    def introduce_word_order_changes(text: str) -> str:
        """Introduce word order changes typical in machine translation."""
        words = text.split()
        if len(words) > 2:
            # Randomly swap adjacent words
            idx = random.randint(0, len(words) - 2)
            words[idx], words[idx + 1] = words[idx + 1], words[idx]
        return ' '.join(words)

    def introduce_article_errors(text: str) -> str:
        """Remove articles incorrectly, simulating common MT errors."""
        # Remove articles
        text = re.sub(r'\b(a|an|the)\s+', '', text, flags=re.IGNORECASE)
        return text

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Simulating translation artifacts"):
        if random.random() < percentage:
            new_row = row.copy()

            artifact_type = random.choice(artifact_types)

            if artifact_type == 'word_order':
                new_row['en'] = introduce_word_order_changes(str(row['en']))
            elif artifact_type == 'article_errors':
                new_row['en'] = introduce_article_errors(str(row['en']))
            # Add more artifact types as needed

            new_row['id'] = f"{row['id']}_mt_artifacts"
            augmented_data.append(new_row)

    return pd.DataFrame(augmented_data)
