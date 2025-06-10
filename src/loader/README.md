# Data Loader Module

A flexible, YAML-configured data loading system for multilingual datasets. This module loads various datasets, ensures they conform to a standard format, and logs everything to Weights & Biases for reproducibility.

## Overview

The loader system uses a registry pattern with YAML configuration to make adding new datasets simple and maintainable. All datasets are automatically validated to ensure they have the required columns for the `MultilingualDataset` class.

## File Structure

```
src/loader/
├── README.md           # This file
├── __init__.py        # Package initialization
├── runner.py          # CLI entry point
├── loader.py          # Main DataLoader class
├── loaders.py         # Dataset loading functions
└── registry.py        # Loader registry system
```

## Required Output Format

All loader functions must return a pandas DataFrame with these columns:
- `title`: Document/sentence title (can be None)
- `ka`: Georgian text
- `en`: English text  
- `domain`: Content domain/category
- `id`: Unique identifier

## Quick Start

### 1. Configure Datasets

Edit `config/datasets.yaml`:

```yaml
datasets:
  - name: "my_dataset"
    source_type: "huggingface"
    source: "username/dataset_name"
    loader_function: "load_my_dataset"
    split: "train"
    enabled: true
    description: "My awesome dataset"
```

### 2. Run the Loader

```bash
python -m src.loader.runner --description "Loading my datasets" --config config/datasets.yaml
```

## Adding New Datasets

### Step 1: Write the Loader Function

Add to `src/loader/loaders.py`:

```python
@register_loader("load_my_dataset", "Description of what this loader does")
def load_my_dataset(config: Dict[str, Any]) -> pd.DataFrame:
    """
    Load my custom dataset.
    
    Args:
        config: Dataset configuration from YAML
        
    Returns:
        DataFrame with required columns: ['title', 'ka', 'en', 'domain', 'id']
    """
    # Your loading logic here
    df = load_dataset(config["source"])
    
    # Transform to required format
    result_df = pd.DataFrame({
        'title': df['title_column'],  # Map your columns
        'ka': df['georgian_column'],
        'en': df['english_column'], 
        'domain': df['category_column'],
        'id': df.index.map(lambda x: f"mydataset_{x}")
    })
    
    return result_df
```

### Step 2: Add to Configuration

Add to `config/datasets.yaml`:

```yaml
datasets:
  # ... existing datasets ...
  - name: "my_dataset"
    source_type: "local"  # or "huggingface", "csv", etc.
    source: "/path/to/data"
    loader_function: "load_my_dataset"
    params:  # Optional parameters
      custom_param: "value"
    enabled: true
    description: "My custom dataset description"
```

That's it! The system will automatically:
- ✅ Validate your output format
- ✅ Save as parquet file
- ✅ Log samples to wandb
- ✅ Track metadata
- ✅ Log configuration and code for reproducibility

## Configuration Options

### Dataset Configuration Fields

| Field | Required | Description |
|-------|----------|-------------|
| `name` | ✅ | Unique dataset name (used for output filename) |
| `source` | ✅ | Data source (HuggingFace dataset, file path, etc.) |
| `loader_function` | ✅ | Name of the registered loader function |
| `source_type` | ❌ | Type hint for the data source |
| `split` | ❌ | Dataset split to use (for HuggingFace datasets) |
| `params` | ❌ | Additional parameters passed to loader function |
| `enabled` | ❌ | Whether to load this dataset (default: true) |
| `description` | ❌ | Human-readable description |

### Example Configuration

```yaml
datasets:
  # HuggingFace dataset
  - name: "english_georgian_corpora"
    source_type: "huggingface"
    source: "Darsala/english_georgian_corpora"
    loader_function: "load_en_ka_corpora"
    split: "train"
    enabled: true
    description: "English-Georgian parallel corpus"
    
  # Complex dataset with parameters
  - name: "flores_devtest" 
    source_type: "huggingface"
    source: "openlanguagedata/flores_plus"
    loader_function: "load_flores_devtest"
    params:
      ka_lang: "kat_Geor"
      en_lang: "eng_Latn"
      split: "devtest"
    enabled: true
    description: "FLORES development and test sets"

  # Local CSV file
  - name: "my_csv_data"
    source_type: "local"
    source: "data/my_data.csv"
    loader_function: "load_custom_csv"
    params:
      separator: ","
      encoding: "utf-8"
    enabled: false
    description: "My local CSV dataset"
```

## CLI Options

```bash
python -m src.loader.runner [OPTIONS]

Options:
  --bucket TEXT        S3 bucket name [default: personal-data-science-data]
  --project TEXT       WandB project name [default: NMT_Training]
  --description TEXT   Experiment description [required]
  --config TEXT        Path to datasets config [default: config/datasets.yaml]
  --develop           Run without WandB login [flag]
  --help              Show this message and exit
```

## Registry System

The loader registry ensures all loader functions are available and provides introspection:

```python
from src.loader.registry import loader_registry

# List all available loaders
print(loader_registry.list_loaders())

# Get a specific loader
loader_func = loader_registry.get_loader("load_my_dataset")
```

## Validation

All loader outputs are automatically validated to ensure they have the required columns. If validation fails, you'll get a clear error message:

```
ValueError: Loader 'my_dataset' output missing required columns: ['ka', 'en']. 
Required columns: ['title', 'ka', 'en', 'domain', 'id']
```

## Common Loader Patterns

### Loading from HuggingFace

```python
@register_loader("load_hf_dataset", "Load from HuggingFace Hub")
def load_hf_dataset(config: Dict[str, Any]) -> pd.DataFrame:
    ds = load_dataset(config["source"])
    df = ds[config.get("split", "train")].to_pandas()
    
    # Map columns to required format
    return df.rename(columns={
        'georgian': 'ka',
        'english': 'en',
        'category': 'domain'
    })
```

### Loading from CSV

```python
@register_loader("load_csv", "Load from CSV file")
def load_csv(config: Dict[str, Any]) -> pd.DataFrame:
    params = config.get("params", {})
    df = pd.read_csv(
        config["source"],
        sep=params.get("separator", ","),
        encoding=params.get("encoding", "utf-8")
    )
    
    # Transform to required format
    return pd.DataFrame({
        'title': df.get('title', None),
        'ka': df['georgian_text'],
        'en': df['english_text'],
        'domain': df.get('category', 'general'),
        'id': df.index.map(lambda x: f"{config['name']}_{x}")
    })
```

### Loading from JSON

```python
@register_loader("load_json", "Load from JSON file")
def load_json(config: Dict[str, Any]) -> pd.DataFrame:
    import json
    
    with open(config["source"], 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Assuming data is a list of dictionaries
    df = pd.DataFrame(data)
    
    # Transform to required format
    return pd.DataFrame({
        'title': df.get('title', None),
        'ka': df['ka_text'],
        'en': df['en_text'],
        'domain': df.get('domain', 'general'),
        'id': df.index.map(lambda x: f"{config['name']}_{x}")
    })
```

## WandB Integration

The system automatically logs:
- **Configuration file**: The YAML config used for the run
- **Loader code**: All Python files in the loader module
- **Data samples**: Random samples from each dataset
- **Metadata**: Dataset sizes, descriptions, and sources
- **Artifacts**: References to S3 storage locations

This ensures complete reproducibility - you can always see exactly how your data was loaded.

## Troubleshooting

### Common Issues

1. **Missing columns error**: Ensure your loader returns all required columns
2. **Loader not found**: Make sure your function is decorated with `@register_loader`
3. **Import errors**: Check that all dependencies are installed
4. **Config file not found**: Verify the path to your YAML file

### Debug Mode

Run with `--develop` flag to skip WandB authentication:

```bash
python -m src.loader.runner --develop --description "Debug run"
```

### Checking Available Loaders

```python
from src.loader.registry import loader_registry
print("Available loaders:")
for name, desc in loader_registry.list_loaders().items():
    print(f"  {name}: {desc}")
```

## Best Practices

1. **Always validate your data**: Check for missing values, encoding issues, etc.
2. **Use descriptive names**: Make dataset names and descriptions clear
3. **Handle errors gracefully**: Add try/catch blocks for external API calls
4. **Document parameters**: Use docstrings to explain config parameters
5. **Test with small samples**: Use `enabled: false` while developing
6. **Version your configs**: Keep different YAML files for different experiments