# Data Cleaner Module

A flexible, YAML-configured data cleaning system for multilingual datasets. This module provides various cleaning functions with comprehensive logging, statistics tracking, and WandB integration for reproducibility.

## Overview

The cleaning system uses a registry pattern with YAML configuration to make cleaning pipelines maintainable and configurable. All cleaning functions automatically log statistics and samples to Weights & Biases for analysis and debugging.

## File Structure

```
src/cleaner/
├── README.md           # This file  
├── __init__.py        # Package initialization
├── runner.py          # CLI entry point
├── cleaner.py         # Main DataCleaner class
├── cleaners.py        # Individual cleaning functions
└── registry.py        # Cleaner registry system

config/
└── cleaning.yaml      # Cleaning configuration
```

## Quick Start

### 1. Configure Cleaning Pipeline

Edit `config/cleaning.yaml`:

```yaml
cleaners:
  - name: "filter_empty_content"
    enabled: true
    description: "Remove empty rows"
    params: {}
    
  - name: "filter_georgian_purity" 
    enabled: true
    description: "Remove Georgian with too much Latin"
    params:
      threshold: 0.5
```

### 2. Run the Cleaner

```bash
python -m src.cleaner.runner \
  --raw-artifact-version "latest" \
  --description "Clean raw datasets" \
  --config config/cleaning.yaml
```

## Available Cleaning Functions

### Core Text Filters

| Function | Description | Parameters |
|----------|-------------|------------|
| `filter_empty_content` | Remove empty/whitespace-only text | None |
| `filter_minimum_length` | Remove very short texts | `min_en_length`, `min_ka_length` |
| `filter_length_ratio` | Remove extreme length differences | `min_ratio`, `max_ratio` |

### Georgian-Specific Filters

| Function | Description | Parameters |
|----------|-------------|------------|
| `filter_georgian_purity` | Remove Georgian with too many Latin chars | `threshold` |
| `filter_shared_latin_characters` | Ensure Latin chars in Georgian appear in English | None |

### Advanced Filters

| Function | Description | Parameters |
|----------|-------------|------------|
| `filter_similarity_threshold` | Remove semantically dissimilar pairs | `threshold`, `batch_size`, `model_name` |

## Adding New Cleaning Functions

### Step 1: Write the Cleaner Function

Add to `src/cleaner/cleaners.py`:

```python
@register_cleaner("my_custom_filter", "Description of what this filter does")
def my_custom_filter(df: pd.DataFrame, config: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    My custom cleaning function.
    
    Args:
        df: Input DataFrame
        config: Parameters from YAML config
        
    Returns:
        Tuple of (cleaned_df, dropped_df)
    """
    threshold = config.get('threshold', 0.5)
    
    # Your cleaning logic here
    def check_row(row):
        return some_condition(row, threshold)
    
    tqdm.pandas(desc="Applying my custom filter")
    mask = df.progress_apply(check_row, axis=1)
    
    cleaned_df = df[mask].copy()
    dropped_df = df[~mask].copy()
    
    return cleaned_df, dropped_df
```

### Step 2: Add to Configuration

Add to `config/cleaning.yaml`:

```yaml
cleaners:
  # ... existing cleaners ...
  - name: "my_custom_filter"
    enabled: true
    description: "My custom cleaning function"
    params:
      threshold: 0.7
      custom_param: "value"
```

That's it! The system will automatically:
- ✅ Validate input/output consistency
- ✅ Track and log statistics
- ✅ Save samples of dropped data
- ✅ Measure processing time
- ✅ Log everything to WandB

## Configuration Reference

### Cleaner Configuration

```yaml
cleaners:
  - name: "function_name"          # Must match registered function name
    enabled: true                  # Whether to apply this cleaner
    description: "What it does"    # Human-readable description  
    params:                        # Parameters passed to function
      param1: value1
      param2: value2
```

### Example Complete Configuration

```yaml
cleaners:
  # Basic content filtering
  - name: "filter_empty_content"
    enabled: true
    description: "Remove rows with empty English or Georgian text"
    params: {}

  - name: "filter_minimum_length"
    enabled: true
    description: "Remove texts that are too short"
    params:
      min_en_length: 5
      min_ka_length: 5

  # Georgian-specific cleaning
  - name: "filter_georgian_purity"
    enabled: true
    description: "Remove Georgian texts with too many Latin characters"
    params:
      threshold: 0.5

  - name: "filter_shared_latin_characters"
    enabled: true
    description: "Ensure Latin chars in Georgian also appear in English"
    params: {}

  # Advanced filtering
  - name: "filter_similarity_threshold"
    enabled: false  # Computationally expensive
    description: "Remove semantically dissimilar text pairs"
    params:
      threshold: 0.5
      batch_size: 32
      model_name: "jinaai/jina-embeddings-v3"

# Global settings
settings:
  log_intermediate: true
  random_seed: 42
  save_dropped_data: false
```

## CLI Options

```bash
python -m src.cleaner.runner [OPTIONS]

Options:
  --raw-artifact-version TEXT  Version of raw dataset artifact [required]
  --bucket TEXT                S3 bucket name [default: personal-data-science-data]
  --project TEXT               WandB project name [default: NMT_Training]
  --description TEXT           Description of cleaning run [required]
  --config TEXT                Path to cleaning config [default: config/cleaning.yaml]
  --develop                    Run without WandB login [flag]
  --help                       Show help message
```

## Function Details

### filter_empty_content

Removes rows where English or Georgian text is empty, contains only whitespace, or only punctuation.

```python
# Detects these as empty:
""
"   "  
"!@#$%"
"[]{};:"
```

### filter_minimum_length

Removes texts shorter than specified thresholds.

```yaml
params:
  min_en_length: 5    # Minimum English characters
  min_ka_length: 5    # Minimum Georgian characters
```

### filter_georgian_purity

Removes Georgian texts containing too many Latin characters. Uses Unicode ranges to identify Georgian characters (ა-ჰ, Ⴀ-Ⴥ).

```yaml
params:
  threshold: 0.5  # Require 50% Georgian characters
```

### filter_shared_latin_characters

Ensures that any Latin characters appearing in Georgian text also appear in the corresponding English text. Handles:
- Roman numeral removal
- English suffix removal while preserving punctuation
- Case-insensitive matching

### filter_length_ratio

Removes pairs with extreme length differences between English and Georgian.

```yaml
params:
  min_ratio: 0.3   # EN/KA ratio must be >= 0.3
  max_ratio: 3.0   # EN/KA ratio must be <= 3.0
```

### filter_similarity_threshold

Uses Jina embeddings to compute semantic similarity between English and Georgian texts. Removes pairs below threshold.

```yaml
params:
  threshold: 0.5                           # Minimum cosine similarity
  batch_size: 32                          # Processing batch size
  model_name: "jinaai/jina-embeddings-v3" # Embedding model
```

**Note**: Requires `sentence-transformers` or `transformers` library. Computationally expensive for large datasets.

## Statistics and Logging

The cleaner automatically tracks and logs:

### Per-Cleaner Statistics
- Original dataset size
- Cleaned dataset size  
- Number of dropped rows
- Retention rate
- Processing time

### Per-Dataset Statistics
- Total retention rate across all cleaners
- Final dataset size
- Individual cleaner performance

### WandB Artifacts
- **Configuration**: YAML config and Python code
- **Samples**: Random samples of dropped data for each cleaner
- **Final samples**: Samples of final cleaned datasets
- **Metadata**: Complete statistics for analysis

### Example Output

```
Cleaning dataset: flores_devtest
  Loaded 1012 rows from flores_devtest.parquet
Applying cleaner: filter_empty_content
  ✓ filter_empty_content: 1012 → 1012 (100.00% retained, 0.1s)
Applying cleaner: filter_georgian_purity  
  ✓ filter_georgian_purity: 1012 → 1003 (99.11% retained, 2.3s)
Applying cleaner: filter_shared_latin_characters
  ✓ filter_shared_latin_characters: 1003 → 998 (99.50% retained, 1.8s)
  ✓ Saved cleaned dataset: artifacts/cleaned/.../flores_devtest_cleaned.parquet
  📊 Final: 1012 → 998 (98.62% retained)
```

## Registry System

The cleaner registry manages all available cleaning functions:

```python
from src.cleaner.registry import cleaner_registry

# List all available cleaners
print(cleaner_registry.list_cleaners())

# Get a specific cleaner
cleaner_func = cleaner_registry.get_cleaner("filter_empty_content")
```

## Error Handling and Validation

The system includes comprehensive validation:

- **Input validation**: Checks for required columns
- **Output validation**: Ensures cleaned + dropped = original
- **Column preservation**: Validates columns aren't modified
- **Error recovery**: Graceful handling of processing errors

## Best Practices

### Configuration Management
1. **Start conservative**: Enable basic cleaners first
2. **Test incrementally**: Add one cleaner at a time
3. **Monitor retention**: Watch retention rates carefully
4. **Document decisions**: Use descriptive descriptions

### Performance Optimization  
1. **Order matters**: Put fast filters first
2. **Batch processing**: Use appropriate batch sizes for similarity
3. **Development mode**: Use `--develop` for testing
4. **Sample first**: Test on small datasets initially

### Debugging and Analysis
1. **Review samples**: Check dropped data samples in WandB
2. **Track statistics**: Monitor retention rates per cleaner
3. **Compare configs**: Use different YAML files for experiments
4. **Iterate carefully**: Make incremental changes

## Integration with Loading Pipeline

The cleaner works seamlessly with the loader pipeline using WandB artifacts:

```bash
# 1. Load raw data
python -m src.loader.runner --description "Load raw datasets"
# This creates artifact: raw:latest

# 2. Clean the data using the artifact
python -m src.cleaner.runner \
  --raw-artifact-version "latest" \
  --description "Clean loaded datasets"
# This creates artifact: cleaned:latest

# 3. Use cleaned data for training
# The cleaned parquet files are ready for model training
```

You can also specify specific versions:
```bash
# Use a specific version
python -m src.cleaner.runner \
  --raw-artifact-version "v1" \
  --description "Clean v1 datasets"

# Use latest
python -m src.cleaner.runner \
  --raw-artifact-version "latest" \
  --description "Clean latest datasets"
```

## Troubleshooting

### Common Issues

1. **Artifact not found**: Check that the raw artifact version exists in WandB
2. **No parquet files found**: Check that the raw artifact contains parquet files
3. **Cleaner not found**: Ensure function is registered with `@register_cleaner`  
4. **Memory errors**: Reduce batch size for similarity filtering
5. **Import errors**: Install required dependencies (`sentence-transformers`)

### Debug Mode

```bash
python -m src.cleaner.runner \
  --raw-artifact-version "latest" \
  --develop \
  --description "Debug run"
```

### Checking Available Cleaners

```python
from src.cleaner.registry import cleaner_registry
print("Available cleaners:")
for name, desc in cleaner_registry.list_cleaners().items():
    print(f"  {name}: {desc}")
```

## Dependencies

Required packages:
```
pandas
numpy  
torch
tqdm
wandb
pyyaml
boto3
```

Optional packages:
```
sentence-transformers  # For similarity filtering
transformers          # Alternative for similarity filtering  
```