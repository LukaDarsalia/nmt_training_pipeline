# Data Splitter Module

A flexible, YAML-configured data splitting system for creating train/valid/test sets from various pipeline artifacts with advanced contamination checking and comprehensive logging.

## Overview

The splitter module allows you to:
- Create train/valid/test splits from any combination of pipeline artifacts
- Mix different percentages from different sources for each split
- Automatically detect and remove contamination between splits
- Maintain stratification by domain
- Log comprehensive statistics and visualizations to WandB

## File Structure

```
src/splitter/
├── README.md           # This file  
├── __init__.py        # Package initialization
├── runner.py          # CLI entry point
├── splitter.py        # Main DataSplitter class
└── contamination.py   # Contamination detection algorithms

config/
└── splitting.yaml     # Splitting configuration
```

## Quick Start

### 1. Configure Your Splits

Edit `config/splitting.yaml`:

```yaml
splits:
  train:
    sources:
      - artifact: "preprocessed:latest"
        percentage: 90
        seed: 42
    
  valid:
    sources:
      - artifact: "raw:v4"
        percentage: 100
      - artifact: "preprocessed:latest"
        percentage: 5
        seed: 42
    
  test:
    sources:
      - artifact: "raw:v3"
        percentage: 100

contamination_check:
  enabled: true
```

### 2. Run the Splitter

```bash
python -m src.splitter.runner \
  --description "Create train/valid/test splits" \
  --config config/splitting.yaml
```

## Configuration Options

### Split Configuration

Each split (train/valid/test) can have multiple sources:

```yaml
splits:
  <split_name>:
    sources:
      - artifact: "artifact_name:version"
        percentage: <0-100>
        seed: <optional_seed>
```

#### Examples

**Simple split from single source:**
```yaml
train:
  sources:
    - artifact: "preprocessed:latest"
      percentage: 80
```

**Mixed sources for robust validation:**
```yaml
valid:
  sources:
    - artifact: "raw:v2"        # 100% of raw v2
      percentage: 100
    - artifact: "cleaned:v1"    # 10% of cleaned v1
      percentage: 10
      seed: 42
```

**Complex multi-source configuration:**
```yaml
train:
  sources:
    - artifact: "preprocessed:v3"
      percentage: 70
      seed: 42
    - artifact: "augmented:latest"
      percentage: 100
    - artifact: "cleaned:v2"
      percentage: 30
      seed: 42
```

### Contamination Checking

Contamination checking prevents data leakage between splits:

```yaml
contamination_check:
  enabled: true
  
  check_pairs:
    - source: "test"
      target: "train"
      action: "remove_from_target"  # or "warn_only"
    - source: "valid"
      target: "train"
      action: "remove_from_target"
  
  methods:
    exact_match:
      enabled: true
      columns: ["en", "ka"]
    
    near_duplicate:
      enabled: true
      threshold: 0.95
      algorithm: "minhash"  # Options: "minhash", "simhash", "embeddings"
      minhash_params:
        num_perm: 128
        threshold: 0.9
```

### Global Settings

```yaml
settings:
  random_seed: 42
  shuffle_before_split: true
  stratify_by_domain: true
  min_samples_per_domain: 10
  log_distributions: true
  log_contamination_report: true
```

## Features

### 1. Flexible Source Mixing

Create splits from any combination of artifacts:
- Use 100% of one artifact for test set (ensuring it's never seen during training)
- Mix multiple versions for robustness
- Sample different percentages with different seeds

### 2. Contamination Detection

Multiple algorithms for detecting contamination:

#### Exact Match
- Detects identical text pairs
- Uses MD5 hashing for efficiency
- Checks specified columns (default: ["en", "ka"])

#### Near-Duplicate Detection
- **MinHash LSH**: Fast approximate matching for large datasets
- **SimHash**: (Placeholder for implementation)
- **Embeddings**: (Placeholder for semantic similarity)

### 3. Stratification

Maintains domain distribution across splits:
- Ensures each domain has minimum samples in each split
- Preserves relative proportions
- Configurable minimum samples per domain

### 4. Comprehensive Logging

#### Statistics Logged
- Split sizes and percentages
- Domain distributions
- Source artifact contributions
- Contamination statistics

#### Visualizations
- Split size comparison
- Domain distribution across splits
- Source artifact distribution
- Contamination summary

#### WandB Artifacts
- Configuration file
- Sample data from each split
- Contamination report
- Contaminated sample examples

## CLI Options

```bash
python -m src.splitter.runner [OPTIONS]

Options:
  --bucket TEXT        S3 bucket name [default: personal-data-science-data]
  --project TEXT       WandB project name [default: NMT_Training]
  --description TEXT   Description of splitting run [required]
  --config TEXT        Path to config [default: config/splitting.yaml]
  --develop           Run without WandB login [flag]
  --help             Show help message
```

## Example Configurations

### Basic Train/Valid/Test Split

```yaml
splits:
  train:
    sources:
      - artifact: "preprocessed:latest"
        percentage: 80
        
  valid:
    sources:
      - artifact: "preprocessed:latest"
        percentage: 10
        seed: 42  # Different from train
        
  test:
    sources:
      - artifact: "preprocessed:latest"
        percentage: 10
        seed: 123  # Different from train/valid

contamination_check:
  enabled: true
```

### Production Configuration

```yaml
splits:
  train:
    sources:
      # Main training data
      - artifact: "preprocessed:v2"
        percentage: 90
        seed: 42
      # Add all augmented data
      - artifact: "augmented:v1"
        percentage: 100
        
  valid:
    sources:
      # Held-out raw data
      - artifact: "raw:flores_v1"
        percentage: 100
      # Small sample from preprocessed
      - artifact: "preprocessed:v2"
        percentage: 5
        seed: 99
        
  test:
    sources:
      # Completely separate test sets
      - artifact: "raw:flores_v2"
        percentage: 100
      - artifact: "raw:benchmark_v1"
        percentage: 100

contamination_check:
  enabled: true
  check_pairs:
    - source: "test"
      target: "train"
      action: "remove_from_target"
    - source: "test"
      target: "valid"
      action: "warn_only"
    - source: "valid"
      target: "train"
      action: "remove_from_target"
```

### Research Configuration with Multiple Versions

```yaml
splits:
  train:
    sources:
      # Mix different preprocessing versions
      - artifact: "preprocessed:v1"
        percentage: 30
        seed: 1
      - artifact: "preprocessed:v2"
        percentage: 30
        seed: 2
      - artifact: "preprocessed:v3"
        percentage: 30
        seed: 3
        
  valid:
    sources:
      # Use older raw version for validation
      - artifact: "raw:v1"
        percentage: 100
        
  test:
    sources:
      # Latest raw data for testing
      - artifact: "raw:latest"
        percentage: 100

contamination_check:
  enabled: true
  methods:
    exact_match:
      enabled: true
    near_duplicate:
      enabled: true
      algorithm: "minhash"
      minhash_params:
        num_perm: 256  # Higher for better accuracy
        threshold: 0.85  # Lower threshold for stricter checking
```

## Integration with Pipeline

The splitter integrates seamlessly with the full pipeline:

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

# 4. Create splits
python -m src.splitter.runner \
  --description "Create train/valid/test splits"
# Creates: splits:latest with train.parquet, valid.parquet, test.parquet

# 5. Train models on splits:latest
```

## Best Practices

### 1. Source Selection
- Use completely separate data for test sets when possible
- Mix sources for validation to improve robustness
- Keep augmented data only in training set

### 2. Contamination Checking
- Always enable for production models
- Use stricter thresholds for competition/benchmark data
- Review contamination reports carefully

### 3. Seed Management
- Use different seeds for train/valid/test sampling
- Document seeds in configuration
- Keep seeds consistent across experiments

### 4. Validation Strategy
- Include some raw/cleaned data in validation
- Don't use augmented data in validation/test
- Consider domain distribution when splitting

## Troubleshooting

### Common Issues

1. **Artifact not found**: Check artifact name and version exist in WandB
2. **Percentages don't add up**: Each artifact is sampled independently
3. **Memory errors**: Reduce MinHash permutations or process in batches
4. **Contamination too strict**: Adjust threshold parameters

### Debug Mode

```bash
python -m src.splitter.runner \
  --develop \
  --description "Debug splitting"
```

### Checking Available Artifacts

```python
import wandb
api = wandb.Api()
artifacts = api.artifacts("your-entity/your-project")
for artifact in artifacts:
    print(f"{artifact.name}:{artifact.version}")
```

## Output Format

The splitter creates three files:
- `train.parquet`: Training data
- `valid.parquet`: Validation data  
- `test.parquet`: Test data

Each file contains all original columns plus:
- `source_artifact`: Which artifact this sample came from
- `source_file`: Original filename within the artifact

## Performance Considerations

### Memory Usage
- Loads artifacts on demand
- Caches loaded artifacts to avoid re-downloading
- MinHash uses ~1KB per document with 128 permutations

### Speed
- Exact matching: O(n) with hash tables
- MinHash: O(n log n) with LSH
- Typical performance: ~10K samples/second

### Scalability
- Can handle millions of samples
- Use sampling for very large datasets
- Consider batch processing for 10M+ samples