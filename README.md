# Neural Machine Translation Training Pipeline

A comprehensive, production-ready neural machine translation training pipeline designed for English-Georgian language pairs. This modular system provides end-to-end functionality from data loading to model deployment, with extensive experimentation capabilities and comprehensive logging via Weights & Biases.

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Pipeline Modules](#pipeline-modules)
- [Configuration](#configuration)
- [Model Architectures](#model-architectures)
- [Evaluation Metrics](#evaluation-metrics)
- [Usage Examples](#usage-examples)
- [Development](#development)
- [Contributing](#contributing)
- [License](#license)

## Overview

This pipeline implements a flexible, registry-based approach to neural machine translation training, specifically optimized for English-Georgian translation tasks. The system is built on HuggingFace Transformers and provides extensive experimentation capabilities through YAML configuration files.

### Key Capabilities

- **Multiple Model Architectures**: Support for M2M100, Marian, Encoder-Decoder, and custom models
- **Advanced Data Processing**: Comprehensive cleaning, preprocessing, and augmentation pipeline
- **Contamination Detection**: Advanced algorithms to prevent data leakage between train/validation/test sets
- **Registry System**: Pluggable components for models, trainers, evaluators, and tokenizers
- **Production Ready**: Complete MLOps integration with S3, WandB, and comprehensive logging
- **Georgian Language Optimization**: Specialized preprocessing and evaluation for Georgian text

## Architecture

The pipeline follows a modular architecture with five main components:

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Loader    │───▶│   Cleaner   │───▶│Preprocessor │───▶│  Splitter   │───▶│  Training   │
│             │    │             │    │             │    │             │    │             │
│ HuggingFace │    │ Data        │    │ Text        │    │ Train/Valid/│    │ Multiple    │
│ Local Files │    │ Quality     │    │ Augment.    │    │ Test Splits │    │ Architectures│
│ CSV/Parquet │    │ Filtering   │    │ Normalize   │    │ Contamin.   │    │ Evaluation  │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
```

Each module is independently configurable and produces WandB artifacts for reproducibility and lineage tracking.

## Features

### Data Pipeline
- **Multi-source Loading**: HuggingFace datasets, local files, CSV, Parquet support
- **Quality Control**: Comprehensive data cleaning with statistical tracking
- **Text Preprocessing**: Character normalization, encoding fixes, language-specific processing
- **Data Augmentation**: Synthetic noise, concatenation, number copying, translation artifacts
- **Smart Splitting**: Contamination detection, stratified sampling, flexible source mixing

### Training System
- **Registry Architecture**: Pluggable models, trainers, evaluators, and tokenizers
- **Model Support**: M2M100, Marian, custom encoder-decoder architectures
- **Advanced Training**: Early stopping, learning rate scheduling, mixed precision
- **Comprehensive Evaluation**: BLEU, chrF++, COMET scores with detailed logging
- **Experiment Tracking**: Complete WandB integration with artifact management

### Production Features
- **Artifact Management**: S3 integration for large-scale data storage
- **Reproducibility**: Complete configuration and code versioning
- **Development Mode**: Offline development capabilities
- **Dry Run Support**: Configuration validation without resource consumption
- **Extensive Logging**: Per-component statistics and sample tracking

## Installation

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended)
- AWS credentials for S3 storage
- Weights & Biases account

### Setup

1. **Clone the repository:**
```bash
git clone https://github.com/LukaDarsalia/nmt_training_pipeline.git
cd nmt_training_pipeline
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Configure environment:**
```bash
# Create .env file with your credentials
echo "WANDB_API_KEY=your_wandb_key" > .env
echo "AWS_ACCESS_KEY_ID=your_aws_key" >> .env
echo "AWS_SECRET_ACCESS_KEY=your_aws_secret" >> .env
```

4. **Verify installation:**
```bash
python -m src.training.runner list --registry all
```

## Quick Start

Run the complete pipeline with default configurations:

```bash
# 1. Load datasets
python -m src.loader.runner --description "Load English-Georgian datasets"

# 2. Clean data
python -m src.cleaner.runner \
  --raw-artifact-version "latest" \
  --description "Clean raw datasets"

# 3. Preprocess and augment
python -m src.preprocessor.runner \
  --cleaned-artifact-version "latest" \
  --description "Preprocess and augment data"

# 4. Create splits
python -m src.splitter.runner \
  --description "Create train/valid/test splits"

# 5. Train model
python -m src.training.runner train \
  --config config/training/m2m100_pretrained.yaml \
  --splits-artifact-version "latest" \
  --description "Train M2M100 model"
```

## Pipeline Modules

### 1. Data Loader (`src/loader/`)
Loads datasets from various sources with automatic format standardization.

**Supported Sources:**
- HuggingFace datasets
- Local CSV/Parquet files
- Custom data sources via registry

**Key Features:**
- Automatic column mapping
- Data validation
- Sample logging
- Registry-based extensibility

### 2. Data Cleaner (`src/cleaner/`)
Comprehensive data quality control with statistical tracking.

**Available Cleaners:**
- Empty content filtering
- Minimum length thresholds
- Georgian text purity validation
- Latin character consistency
- Semantic similarity filtering
- Length ratio validation

### 3. Data Preprocessor (`src/preprocessor/`)
Text normalization and data augmentation pipeline.

**Preprocessing Functions:**
- Character normalization
- Encoding issue fixes
- American English conversion
- Whitespace cleaning
- Punctuation synchronization

**Augmentation Functions:**
- Synthetic noise injection
- Long text concatenation
- Number copying enhancement
- Translation artifact simulation

### 4. Data Splitter (`src/splitter/`)
Advanced data splitting with contamination detection.

**Features:**
- Multi-source flexible splitting
- Exact match contamination detection
- Near-duplicate detection (MinHash)
- Domain stratification
- Comprehensive reporting

### 5. Training System (`src/training/`)
Registry-based training pipeline with multiple model support.

**Supported Models:**
- M2M100 (multilingual pretrained)
- Marian (custom and pretrained)
- Encoder-Decoder architectures
- Custom model implementations

## Configuration

All pipeline components are configured via YAML files in the `config/` directory:

### Dataset Configuration (`config/datasets.yaml`)
```yaml
datasets:
  - name: "english_georgian_corpora"
    source_type: "huggingface"
    source: "Darsala/english_georgian_corpora"
    loader_function: "load_en_ka_corpora"
    enabled: true
```

### Training Configuration (`config/training/m2m100_pretrained.yaml`)
```yaml
model:
  type: "m2m100_pretrained"
  model_name: "facebook/m2m100_418M"

trainer:
  type: "seq2seq_with_metrics"
  training:
    num_epochs: 8
    learning_rate: 1e-5
    eval_steps: 0.2
```

## Model Architectures

### M2M100 Models
- **M2M100 Pretrained**: Fine-tune Facebook's multilingual M2M100 model
- **Configuration**: Automatic language detection and generation
- **Best for**: Quick setup with good baseline performance

### Marian Models
- **Marian Custom**: Build from scratch with Georgian tokenizer
- **Marian Pretrained**: Fine-tune existing Marian models
- **Best for**: Specialized Georgian language processing

### Encoder-Decoder Models
- **Flexible Architecture**: Mix any encoder/decoder combination
- **Examples**: BERT + GPT-2, multilingual-E5 + custom decoder
- **Best for**: Experimental architectures and research

## Evaluation Metrics

The pipeline supports comprehensive evaluation:

- **BLEU**: Standard machine translation metric
- **chrF++**: Character-level F-score with word order
- **Georgian COMET**: Specialized quality estimation for Georgian
- **Custom Metrics**: Extensible via registry system

All metrics are logged with detailed breakdown by dataset and domain.

## Usage Examples

### Basic Training Experiment
```bash
python -m src.training.runner train \
  --config config/training/m2m100_pretrained.yaml \
  --splits-artifact-version "latest" \
  --description "Baseline M2M100 experiment"
```

### Custom Preprocessing Pipeline
```bash
# Custom preprocessing configuration
python -m src.preprocessor.runner \
  --cleaned-artifact-version "v1" \
  --config config/preprocessing_custom.yaml \
  --description "Custom augmentation experiment"
```

### Development Mode (Offline)
```bash
# Run without WandB authentication
python -m src.training.runner train \
  --config config/training/m2m100_pretrained.yaml \
  --splits-artifact-version "latest" \
  --develop \
  --description "Local development run"
```

### Configuration Validation
```bash
# Validate configuration without training
python -m src.training.runner train \
  --config config/training/experiment.yaml \
  --splits-artifact-version "latest" \
  --dry-run \
  --description "Config validation"
```

### Exploratory Data Analysis
```bash
python -m src.eda.runner run \
  --artifact "raw:latest" \
  --analysis-type "complete" \
  --limit 10000
```

## Development

### Adding New Models
1. Create model function in `src/training/models/`
2. Register with `@register_model("model_name", "description")`
3. Add configuration to `config/training/`

### Adding New Preprocessing Functions
1. Implement function in `src/preprocessor/preprocessors.py`
2. Register with `@register_preprocessor("function_name", "description")`
3. Add to configuration YAML

### Running Tests
```bash
# Validate all configurations
python -m src.training.runner list --registry all

# Test data pipeline
python -m src.loader.runner --develop --description "Test run"
```

### Performance Optimization
- Use `--develop` flag for offline development
- Reduce batch sizes for memory-constrained environments
- Enable mixed precision training (`fp16: true`)
- Adjust evaluation frequency for faster iteration

## Project Structure

```
nmt_training_pipeline/
├── README.md                    # This file
├── requirements.txt            # Python dependencies
├── LICENSE                     # MIT License
├── config/                     # Configuration files
│   ├── datasets.yaml          # Dataset loading config
│   ├── cleaning.yaml           # Data cleaning config
│   ├── preprocessing.yaml      # Preprocessing config
│   ├── splitting.yaml          # Data splitting config
│   └── training/               # Training configurations
├── src/                        # Source code
│   ├── loader/                 # Data loading module
│   ├── cleaner/                # Data cleaning module
│   ├── preprocessor/           # Preprocessing module
│   ├── splitter/               # Data splitting module
│   ├── training/               # Training pipeline
│   ├── eda/                    # Exploratory data analysis
│   └── utils/                  # Shared utilities
└── artifacts/                  # Output artifacts (created during execution)
```

## Contributing

Contributions are welcome! Please follow these guidelines:

1. **Fork the repository** and create a feature branch
2. **Follow the registry pattern** for new components
3. **Add comprehensive documentation** for new features
4. **Include configuration examples** for new functionality
5. **Test with development mode** before submitting
6. **Update README** if adding new modules or features

### Development Setup
```bash
# Create development environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# Run in development mode
export WANDB_MODE=offline  # Optional: disable WandB
python -m src.training.runner train --develop --dry-run [options]
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author

**Luka Darsalia**
- GitHub: [@LukaDarsalia](https://github.com/LukaDarsalia)
- Project: [nmt_training_pipeline](https://github.com/LukaDarsalia/nmt_training_pipeline)

## Acknowledgments

- Built on [HuggingFace Transformers](https://huggingface.co/transformers/)
- Evaluation metrics from [COMET](https://unbabel.github.io/COMET/html/index.html)
- Georgian language resources from various open-source projects
- MLOps infrastructure powered by [Weights & Biases](https://wandb.ai/)

---

For detailed documentation on specific modules, refer to the README files in each module directory:
- [Data Loader Documentation](src/loader/README.md)
- [Data Cleaner Documentation](src/cleaner/README.md)
- [Preprocessor Documentation](src/preprocessor/README.md)
- [Splitter Documentation](src/splitter/README.md)
- [Training Documentation](src/training/README.md)
