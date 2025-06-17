# Neural Machine Translation Training Pipeline

A flexible, registry-based training system for neural machine translation models. This pipeline enables easy experimentation with different model architectures, training strategies, and evaluation metrics through YAML configuration files.

## Overview

The training pipeline uses a registry system to dynamically create and configure:
- **Models**: Various architectures (Marian, Encoder-Decoder, Custom)
- **Trainers**: Different training strategies and configurations
- **Evaluators**: Multiple evaluation metrics including custom Georgian COMET

## File Structure

```
src/training/
├── README.md                 # This file
├── __init__.py              # Package initialization
├── runner.py                # CLI entry point
├── trainer.py               # Main NMTTrainer class
├── registry/                # Registry system
│   ├── __init__.py
│   ├── base.py             # Base registry class
│   ├── model_registry.py   # Model component registry
│   ├── trainer_registry.py # Trainer component registry
│   └── evaluator_registry.py # Evaluator component registry
├── models/                  # Model implementations
│   ├── __init__.py
│   ├── marian_models.py    # Marian-based models
│   ├── encoder_decoder_models.py # Encoder-decoder combinations
│   └── custom_models.py    # Custom model architectures
├── trainers/               # Trainer implementations
│   ├── __init__.py
│   ├── seq2seq_trainers.py # Seq2Seq training strategies
│   └── custom_trainers.py  # Custom training strategies
├── evaluators/             # Evaluation metrics
│   ├── __init__.py
│   ├── standard_metrics.py # Standard MT metrics
│   └── custom_metrics.py   # Custom metrics (Georgian COMET)
└── utils/                  # Utilities and helpers
    ├── __init__.py
    ├── callbacks.py        # Training callbacks
    └── data_utils.py       # Data loading utilities
```

## Quick Start

### 1. Basic Training Run

```bash
python -m src.training.runner train \
  --config config/training/baseline_marian.yaml \
  --tokenizer-artifact-version "latest" \
  --splits-artifact-version "latest" \
  --description "Baseline Marian model experiment"
```

### 2. Fine-tuning Pretrained Model

```bash
python -m src.training.runner train \
  --config config/training/pretrained_marian.yaml \
  --tokenizer-artifact-version "latest" \
  --splits-artifact-version "latest" \
  --description "Fine-tuning pretrained Marian model"
```

### 3. Encoder-Decoder Experiment

```bash
python -m src.training.runner train \
  --config config/training/encoder_decoder.yaml \
  --tokenizer-artifact-version "latest" \
  --splits-artifact-version "latest" \
  --description "BERT encoder + GPT-2 decoder experiment"
```

## Configuration System

All experiments are configured through YAML files. Here's the configuration structure:

### Model Configuration

```yaml
model:
  type: "marian_custom"  # Registered model type
  
  # Architecture parameters (for custom models)
  architecture:
    d_model: 512
    encoder_layers: 6
    decoder_layers: 6
    encoder_attention_heads: 8
    decoder_attention_heads: 8
    dropout: 0.1
  
  # Generation settings
  generation_config:
    max_length: 128
    num_beams: 1
    early_stopping: true
```

### Training Configuration

```yaml
trainer:
  type: "seq2seq_with_metrics"  # Registered trainer type
  
  training:
    num_epochs: 10
    train_batch_size: 32
    learning_rate: 5e-4
    lr_scheduler_type: "cosine"
    warmup_ratio: 0.1
    
    # Early stopping
    early_stopping:
      enabled: true
      patience: 3
```

### Evaluation Configuration

```yaml
trainer:
  evaluation:
    evaluators:
      - name: "sacrebleu"
        config: {}
      - name: "chrf"
        config:
          word_order: 2
      - name: "georgian_comet"
        config:
          batch_size: 8
```

## Available Components

### Models

| Component | Description | Use Case |
|-----------|-------------|----------|
| `marian_pretrained` | Load pretrained Marian from HF | Fine-tuning existing models |
| `marian_custom` | Custom Marian architecture | Training from scratch |
| `marian_finetuned` | Load local finetuned model | Continue training |
| `encoder_decoder_pretrained` | Combine encoder + decoder | Experimental architectures |
| `encoder_decoder_random` | Random init encoder-decoder | Clean slate training |
| `encoder_decoder_mixed` | Mixed initialization | Partial pretraining |

### Trainers

| Component | Description | Features |
|-----------|-------------|----------|
| `standard_seq2seq` | Basic Seq2Seq trainer | Standard training loop |
| `seq2seq_with_metrics` | Enhanced trainer | Custom evaluation metrics |

### Evaluators

| Component | Description | Notes |
|-----------|-------------|-------|
| `bleu` | BLEU score | Standard MT metric |
| `sacrebleu` | SacreBLEU score | Standardized BLEU |
| `chrf` | CHRF score | Character-level metric |
| `rouge` | ROUGE scores | Summarization metric |
| `meteor` | METEOR score | Semantic similarity |
| `bertscore` | BERTScore | Contextual embeddings |
| `georgian_comet` | Georgian COMET | Fine-tuned for Georgian |
| `length_ratio` | Length ratio | Simple baseline |
| `exact_match` | Exact match | String matching |

## CLI Commands

### Training Command

```bash
python -m src.training.runner train [OPTIONS]

Options:
  --config TEXT                    Path to training configuration YAML [required]
  --tokenizer-artifact-version TEXT  Tokenizer artifact version [required]
  --splits-artifact-version TEXT     Splits artifact version [required]
  --model-artifact-version TEXT      Model artifact version (for fine-tuning)
  --bucket TEXT                    S3 bucket name [default: personal-data-science-data]
  --project TEXT                   WandB project name [default: NMT_Training]
  --run-name TEXT                  Optional WandB run name
  --description TEXT               Experiment description [required]
  --develop                        Development mode flag
  --dry-run                        Validate config without training
```

### List Components

```bash
python -m src.training.runner list [--registry models|trainers|evaluators|all]
```

## Adding New Components

### Adding a New Model

1. Create model function in appropriate file (e.g., `models/custom_models.py`):

```python
@register_model("my_custom_model", "Description of my model")
def create_my_custom_model(config: Dict[str, Any], 
                          tokenizer: PreTrainedTokenizer) -> Tuple[nn.Module, GenerationConfig, DataCollator]:
    # Your implementation here
    model = MyCustomModel(config)
    generation_config = GenerationConfig(...)
    data_collator = DataCollatorForSeq2Seq(...)
    
    return model, generation_config, data_collator
```

2. Use in configuration:

```yaml
model:
  type: "my_custom_model"
  # Your custom parameters
```

### Adding a New Trainer

1. Create trainer function in `trainers/custom_trainers.py`:

```python
@register_trainer("my_custom_trainer", "Description of my trainer")
def create_my_custom_trainer(config: Dict[str, Any],
                           model: nn.Module,
                           tokenizer: PreTrainedTokenizer,
                           train_dataset: Dataset,
                           eval_dataset: Dataset,
                           data_collator: DataCollator) -> Trainer:
    # Your implementation here
    return trainer
```

2. Use in configuration:

```yaml
trainer:
  type: "my_custom_trainer"
  # Your custom parameters
```

### Adding a New Evaluator

1. Create evaluator function in `evaluators/custom_metrics.py`:

```python
@register_evaluator("my_custom_metric", "Description of my metric")
def create_my_custom_metric(config: Dict[str, Any]) -> Callable:
    def evaluate_metric(predictions: List[str], references: List[str]) -> Dict[str, float]:
        # Your implementation here
        score = compute_my_metric(predictions, references)
        return {"my_metric": score}
    
    return evaluate_metric
```

2. Use in configuration:

```yaml
trainer:
  evaluation:
    evaluators:
      - name: "my_custom_metric"
        config:
          param1: value1
```

## Integration with Pipeline

The training pipeline integrates seamlessly with the full data pipeline:

```bash
# 1. Load data
python -m src.loader.runner --description "Load datasets"
# Creates: raw:latest

# 2. Clean data
python -m src.cleaner.runner \
  --raw-artifact-version "latest" \
  --description "Clean datasets"
# Creates: cleaned:latest

# 3. Preprocess data
python -m src.preprocessor.runner \
  --cleaned-artifact-version "latest" \
  --description "Preprocess and augment"
# Creates: preprocessed:latest

# 4. Create tokenizer
python -m src.tokenizer.runner \
  --preprocessed-artifact-version "latest" \
  --description "Train tokenizer"
# Creates: tokenizer:latest

# 5. Split data
python -m src.splitter.runner \
  --description "Create train/valid/test splits"
# Creates: splits:latest

# 6. Train model
python -m src.training.runner train \
  --config config/training/baseline_marian.yaml \
  --tokenizer-artifact-version "latest" \
  --splits-artifact-version "latest" \
  --description "Train baseline model"
# Creates: training:latest
```

## Best Practices

### Configuration Management

1. **Use descriptive names**: Name your config files clearly
2. **Version control**: Keep configs in git for reproducibility
3. **Document experiments**: Use detailed descriptions
4. **Parameter sweeps**: Create config variants for hyperparameter tuning

### Experiment Organization

1. **Consistent naming**: Use systematic naming conventions
2. **Tag experiments**: Use meaningful tags in configs
3. **Compare results**: Use WandB for experiment comparison
4. **Save configs**: Always log configs to WandB artifacts

### Model Development

1. **Start simple**: Begin with baseline configurations
2. **Validate quickly**: Use dry-run mode to test configs
3. **Monitor training**: Use prediction logging callbacks
4. **Early stopping**: Configure appropriate early stopping

### Evaluation Strategy

1. **Multiple metrics**: Use diverse evaluation metrics
2. **Georgian COMET**: Include for Georgian-specific evaluation
3. **Custom metrics**: Add domain-specific evaluators
4. **Validation split**: Use consistent validation methodology

## Troubleshooting

### Common Issues

1. **Component not found**: Check registry imports and decorators
2. **Configuration errors**: Use dry-run mode for validation
3. **Memory issues**: Reduce batch sizes or sequence length
4. **Tokenizer mismatches**: Ensure consistent tokenizer usage

### Debug Mode

```bash
python -m src.training.runner train \
  --config config/training/baseline_marian.yaml \
  --tokenizer-artifact-version "latest" \
  --splits-artifact-version "latest" \
  --description "Debug run" \
  --develop \
  --dry-run
```

### Checking Components

```bash
python -m src.training.runner list --registry all
```

## Performance Considerations

### Memory Optimization

- Use gradient accumulation for larger effective batch sizes
- Enable FP16 training when possible
- Adjust sequence length based on GPU memory

### Training Speed

- Use appropriate number of workers for data loading
- Consider model parallelism for large models
- Use efficient evaluation frequencies

### Hyperparameter Tuning

- Start with proven baselines
- Use learning rate schedulers (cosine recommended)
- Tune batch size and accumulation steps together
- Consider warmup ratios for stable training

## Example Experiments

### Baseline Comparison

```bash
# Train baseline Marian
python -m src.training.runner train \
  --config config/training/baseline_marian.yaml \
  --tokenizer-artifact-version "latest" \
  --splits-artifact-version "latest" \
  --description "Baseline Marian model"

# Fine-tune pretrained model
python -m src.training.runner train \
  --config config/training/pretrained_marian.yaml \
  --tokenizer-artifact-version "latest" \
  --splits-artifact-version "latest" \
  --description "Pretrained Marian fine-tuning"
```

### Architecture Exploration

```bash
# Encoder-decoder experiment
python -m src.training.runner train \
  --config config/training/encoder_decoder.yaml \
  --tokenizer-artifact-version "latest" \
  --splits-artifact-version "latest" \
  --description "BERT-GPT2 encoder-decoder"
```

This training pipeline provides a flexible foundation for neural machine translation experiments while maintaining reproducibility and ease of use.