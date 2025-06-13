# Data Preprocessor Module

A flexible, YAML-configured data preprocessing and augmentation system for multilingual datasets. This module provides text normalization, cleaning functions, and various augmentation techniques with comprehensive logging and WandB integration.

## Overview

The preprocessing system uses a dual registry pattern to handle both:
1. **Preprocessing functions**: Modify existing data in-place (text normalization, cleaning)
2. **Augmentation functions**: Create new data rows (synthetic examples, data augmentation)

All functions are configurable via YAML and automatically log examples to Weights & Biases for quality control.

## File Structure

```
src/preprocessor/
├── README.md           # This file  
├── __init__.py        # Package initialization
├── runner.py          # CLI entry point
├── preprocessor.py    # Main DataPreprocessor class
├── preprocessors.py   # Text preprocessing functions
├── augmenters.py      # Data augmentation functions
└── registry.py        # Dual registry system

config/
└── preprocessing.yaml  # Preprocessing configuration
```

## Quick Start

### 1. Configure Preprocessing Pipeline

Edit `config/preprocessing.yaml`:

```yaml
preprocessors:
  - name: "normalize_characters"
    enabled: true
    params:
      columns: ["en", "ka"]
      
  - name: "convert_to_american"
    enabled: true
    params:
      columns: ["en"]

augmenters:
  - name: "number_copying"
    enabled: true
    params:
      num_examples: 200
```

### 2. Run the Preprocessor

```bash
python -m src.preprocessor.runner \
  --cleaned-artifact-version "latest" \
  --description "Preprocess and augment cleaned datasets" \
  --config config/preprocessing.yaml
```

## Available Functions

### Preprocessing Functions (In-Place Modification)

| Function | Description | Key Parameters |
|----------|-------------|----------------|
| `normalize_characters` | Standardize dashes, quotes, apostrophes | `columns` |
| `remove_extra_whitespaces` | Clean spacing and trim text | `columns` |
| `convert_to_american` | British to American English | `columns` |
| `lowercase_english` | Convert English to lowercase | `preserve_proper_nouns` |
| `sync_punctuation` | Match punctuation between EN/KA | `strategy`, `punctuation_marks` |
| `remove_brackets_content` | Remove content in brackets/parens | `bracket_types` |
| `fix_encoding_issues` | Fix common encoding problems | `columns` |

### Augmentation Functions (Create New Examples)

| Function | Description | Key Parameters |
|----------|-------------|----------------|
| `synthetic_noise` | Add character-level noise | `percentage`, `noise_types`, `target_column` |
| `sentence_concatenation` | Combine consecutive sentences | `percentage`, `separator` |
| `concatenate_long_texts` | Restructure dataset with longer texts | `concatenation_ratio`, `group_size`, `strategy` |
| `sentence_reversal` | Reverse word order | `target_column`, `percentage` |
| `random_word_replacement` | Replace with random words | `lexicon`, `percentage` |
| `number_copying` | Generate number copying examples | `num_examples`, `number_types` |
| `georgian_text_copying` | EN=KA for Georgian text | `percentage` or `num_examples` |
| `simulate_translation_artifacts` | Simulate MT errors/artifacts | `artifact_types`, `percentage` |

## Function Details

### Core Preprocessing Functions

#### normalize_characters
Standardizes various Unicode characters to ASCII equivalents:
- Dashes: `–`, `—`, `−` → `-`
- Apostrophes: `'`, `'`, `´`, `'` → `'`
- Quotes: `"`, `"`, `„`, `«` → `"`

```yaml
- name: "normalize_characters"
  params:
    columns: ["en", "ka"]
```

#### convert_to_american
Uses the `eng` library to convert British English to American English:
- "colour" → "color"
- "realise" → "realize"
- "centre" → "center"

**Note**: Required for inference consistency.

```yaml
- name: "convert_to_american"
  params:
    columns: ["en"]
```

#### sync_punctuation
Ensures punctuation consistency between English and Georgian:

```yaml
- name: "sync_punctuation"
  params:
    strategy: "add_missing"  # or "remove_all"
    punctuation_marks: ".!?;:"
```

Strategies:
- `add_missing`: If one text has punctuation, add to the other
- `remove_all`: Remove all ending punctuation

### Core Augmentation Functions

#### synthetic_noise
Adds character-level noise to improve model robustness:
- **Delete**: Remove random character
- **Insert**: Add random character
- **Duplicate**: Duplicate random character
- **Substitute**: Replace with random character
- **Swap**: Swap adjacent characters

```yaml
- name: "synthetic_noise"
  params:
    percentage: 0.05
    noise_types: ["delete", "insert", "substitute", "swap"]
    char_ranges: ["a-z", "A-Z", "ა-ჰ"]
    target_column: "en"
```

#### number_copying
Generates examples to teach number copying:

```yaml
- name: "number_copying"
  params:
    num_examples: 200
    number_types: ["integer", "float", "large", "small"]
```

Creates examples like:
- EN: "The number is 42." → KA: "რიცხვი არის 42."
- EN: "Value: 3.14" → KA: "მნიშვნელობა: 3.14"

#### concatenate_long_texts
**Dataset restructuring operation** - Concatenates multiple rows to create longer texts, improving translation quality for long texts by better covering the positional embedding space:

```yaml
- name: "concatenate_long_texts"
  params:
    concatenation_ratio: 0.2    # 20% of rows affected
    min_group_size: 2           # Minimum rows per group
    max_group_size: 4           # Maximum rows per group
    separator: " "              # Text separator
    strategy: "consecutive"     # 'consecutive' or 'random'
```

**Note**: This is a special augmenter that **restructures the dataset** (reduces row count) rather than adding new examples.

Strategies:
- `consecutive`: Concatenate adjacent rows in order
- `random`: Randomly group rows for concatenation

Example transformation:
- Input: 3 rows → "Hello." | "How are you?" | "Fine."  
- Output: 1 row → "Hello. How are you? Fine."

#### georgian_text_copying
Teaches the model to copy Georgian text when appropriate by sampling from the dataset:

```yaml
- name: "georgian_text_copying"
  params:
    percentage: 0.01        # 1% of dataset size
    # OR use num_examples: 100 for fixed number
```

**Always samples from current dataset** - no predefined texts. This ensures:
- Relevance to your specific domain
- Vocabulary consistency with training data
- Authentic Georgian text patterns

Creates examples like:
- Georgian text from dataset: "თბილისი საქართველოს დედაქალაქია"
- Generated example: EN = "თბილისი საქართველოს დედაქალაქია", KA = "თბილისი საქართველოს დედაქალაქია"

#### simulate_translation_artifacts
Simulates common machine translation errors and artifacts to improve model robustness:

```yaml
- name: "simulate_translation_artifacts"
  params:
    percentage: 0.02
    artifact_types: ["word_order", "article_errors"]
```

Artifact types:
- `word_order`: Introduces unnatural word ordering
- `article_errors`: Removes articles incorrectly
- Future: Could add preposition errors, tense errors, literal translations

This helps train models that are robust to imperfect input and common MT system errors.

## Configuration Reference

### Complete Configuration Example

```yaml
# Preprocessing functions (modify existing data)
preprocessors:
  - name: "normalize_characters"
    enabled: true
    description: "Standardize Unicode characters"
    params:
      columns: ["en", "ka"]

  - name: "convert_to_american"
    enabled: true
    description: "British to American English"
    params:
      columns: ["en"]

  - name: "sync_punctuation"
    enabled: true
    description: "Match punctuation between languages"
    params:
      strategy: "add_missing"

# Augmentation functions (create new data)
augmenters:
  - name: "concatenate_long_texts"
    enabled: true
    description: "Restructure dataset with longer texts"
    params:
      concatenation_ratio: 0.2
      strategy: "consecutive"

  - name: "synthetic_noise"
    enabled: true
    description: "Add robustness through noise"
    params:
      percentage: 0.05
      target_column: "en"

  - name: "number_copying"
    enabled: true
    description: "Teach number copying"
    params:
      num_examples: 200

  - name: "georgian_text_copying"
    enabled: true
    description: "Teach Georgian text copying from dataset"
    params:
      percentage: 0.01

# Global settings
settings:
  random_seed: 42
  log_samples: true
  max_log_samples: 10
```

## Adding New Functions

### Adding a Preprocessor

Add to `src/preprocessor/preprocessors.py`:

```python
@register_preprocessor("my_preprocessor", "Description of what it does")
def my_preprocessor(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """
    My custom preprocessing function.
    
    Args:
        df: Input DataFrame
        config: Parameters from YAML config
        
    Returns:
        Processed DataFrame (same number of rows)
    """
    columns = config.get('columns', ['en', 'ka'])
    
    def process_text(text: str) -> str:
        # Your processing logic here
        return text.upper()  # Example
    
    return _process_dataframe_columns(df, process_text, columns)
```

### Adding an Augmenter

Add to `src/preprocessor/augmenters.py`:

```python
@register_augmenter("my_augmenter", "Description of augmentation")
def my_augmenter(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """
    My custom augmentation function.
    
    Args:
        df: Input DataFrame
        config: Parameters from YAML config
        
    Returns:
        DataFrame with new augmented examples
    """
    percentage = config.get('percentage', 0.1)
    augmented_data = []
    
    for idx, row in tqdm(df.iterrows(), desc="My augmentation"):
        if random.random() < percentage:
            new_row = row.copy()
            new_row['en'] = modify_somehow(row['en'])
            new_row['id'] = f"{row['id']}_my_aug"
            augmented_data.append(new_row)
    
    return pd.DataFrame(augmented_data)
```

Then add to config:
```yaml
- name: "my_augmenter"
  enabled: true
  params:
    percentage: 0.05
```

## CLI Options

```bash
python -m src.preprocessor.runner [OPTIONS]

Options:
  --cleaned-artifact-version TEXT  Version of cleaned dataset artifact [required]
  --bucket TEXT                    S3 bucket name [default: personal-data-science-data]
  --project TEXT                   WandB project name [default: NMT_Training]
  --description TEXT               Description of preprocessing run [required]
  --config TEXT                    Path to config [default: config/preprocessing.yaml]
  --develop                        Run without WandB login [flag]
  --help                          Show help message
```

## Pipeline Integration

The preprocessor integrates seamlessly with the full pipeline:

```bash
# 1. Load raw data
python -m src.loader.runner --description "Load datasets"
# Creates: raw:latest

# 2. Clean the data  
python -m src.cleaner.runner \
  --raw-artifact-version "latest" \
  --description "Clean datasets"
# Creates: cleaned:latest

# 3. Preprocess and augment
python -m src.preprocessor.runner \
  --cleaned-artifact-version "latest" \
  --description "Preprocess and augment"
# Creates: preprocessed:latest

# 4. Train models on preprocessed:latest
```

## Quality Control & Logging

The system automatically logs:

### Per-Function Samples
- Random samples of preprocessed text for each function
- Before/after comparisons for validation
- Augmented examples to verify quality

### Statistics Tracking
- Processing time per function
- Number of examples generated by augmenters
- Augmentation ratios per dataset

### WandB Artifacts
- **Configuration**: YAML config and Python code
- **Samples**: Function-specific examples for quality control
- **Final samples**: Combined preprocessed and augmented data
- **Metadata**: Complete statistics and ratios

### Example Output

```
Processing dataset: flores_devtest
  Loaded 998 rows from flores_devtest_cleaned.parquet
  --- Preprocessing Steps ---
  Applying preprocessor: normalize_characters
    ✓ normalize_characters: Processed 998 rows in 0.8s
  Applying preprocessor: convert_to_american
    ✓ convert_to_american: Processed 998 rows in 2.1s
  --- Augmentation Steps ---
  Applying augmenter: synthetic_noise
    ✓ synthetic_noise: Generated 47 new examples in 1.2s
  Applying augmenter: number_copying
    ✓ number_copying: Generated 200 new examples in 0.3s
  Combined 247 augmented examples with 998 original
  ✓ Saved processed dataset: artifacts/preprocessed/.../flores_devtest_processed.parquet
  📊 Final: 998 → 1245 (24.75% augmentation)
```

## Best Practices

### Function Ordering
1. **Character normalization first**: Fix encoding and Unicode issues
2. **Text cleaning**: Remove extra spaces, fix formatting
3. **Language-specific**: British→American, case conversion
4. **Cross-language sync**: Punctuation matching
5. **Augmentation last**: Generate new examples from clean data

### Configuration Management
1. **Start conservative**: Enable basic preprocessing only
2. **Add augmentation gradually**: Monitor quality with samples
3. **Balance augmentation**: Don't over-augment (10-30% typical)
4. **Test inference functions**: Ensure preprocessing works in production

### Quality Monitoring
1. **Review WandB samples**: Check function outputs carefully
2. **Monitor statistics**: Watch processing times and ratios
3. **Validate manually**: Spot-check augmented examples
4. **A/B test configurations**: Compare different preprocessing setups

## Dependencies

Required packages:
```
pandas
numpy  
torch
tqdm
wandb
pyyaml
click
eng  # For British→American conversion
```

## Troubleshooting

### Common Issues

1. **Missing eng library**: Install with `pip install eng`
2. **Artifact not found**: Check cleaned artifact version exists
3. **Function not found**: Ensure function is registered with decorator
4. **Memory issues**: Reduce augmentation percentages
5. **Quality issues**: Review samples in WandB before full runs

### Debug Mode

```bash
python -m src.preprocessor.runner \
  --cleaned-artifact-version "latest" \
  --develop \
  --description "Debug run"
```

### Checking Available Functions

```python
from src.preprocessor.registry import preprocessor_registry

print("Preprocessors:")
for name, desc in preprocessor_registry.list_preprocessors().items():
    print(f"  {name}: {desc}")

print("Augmenters:")
for name, desc in preprocessor_registry.list_augmenters().items():
    print(f"  {name}: {desc}")
```

## Production Considerations

### Inference Pipeline
Some preprocessing functions must be applied during inference:
- `normalize_characters` - Always apply
- `convert_to_american` - Always apply  
- `remove_extra_whitespaces` - Always apply
- `fix_encoding_issues` - Always apply

### Model Training
The preprocessed dataset is ready for model training:
- Consistent text normalization
- Improved robustness through augmentation
- Better number handling
- Enhanced Georgian text copying

### Performance
- Preprocessing: ~1000 examples/second
- Augmentation: Varies by function (100-1000/second)
- Memory usage: ~2x dataset size during processing
- Disk usage: 1.2-2x original size after augmentation

This preprocessing pipeline ensures your data is clean, consistent, and augmented for optimal model training performance!