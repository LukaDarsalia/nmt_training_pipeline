# Neural Machine Translation Training Pipeline

A flexible, registry-based training system for neural machine translation models. This pipeline enables easy experimentation with different model architectures, training strategies, and evaluation metrics through YAML configuration files.

## Quick Start

### 1. Basic Training Run (Custom Model from Scratch)

```bash
python -m src.training.runner train \
  --config config/training/baseline_marian.yaml \
  --splits-artifact-version "latest" \
  --description "Baseline Marian model experiment"
```

### 2. Fine-tuning Pretrained Model

```bash
python -m src.training.runner train \
  --config config/training/pretrained_marian.yaml \
  --splits-artifact-version "latest" \
  --description "Fine-tuning pretrained Marian model"
```

### 3. Continue Training from Checkpoint

```bash
python -m src.training.runner train \
  --config config/training/baseline_marian.yaml \
  --splits-artifact-version "latest" \
  --model-artifact-version "v3" \
  --description "Continue training from checkpoint"
```

## Tokenizer Handling

**The training pipeline automatically handles tokenizers based on model type:**

### For Custom Models (marian_custom, custom_transformer)
- Uses `RichNachos/georgian-corpus-tokenizer-test` by default
- Optimized for Georgian language processing
- No additional configuration needed

### For Pretrained Models (marian_pretrained, m2m100_multilingual)
- Uses the model's own tokenizer automatically
- Examples: M2M100, mBART, other HuggingFace models
- Preserves model-specific tokenization behavior

### For Encoder-Decoder Models
- Uses the decoder model's tokenizer
- Example: BERT encoder + GPT-2 decoder uses GPT-2 tokenizer
- Handles special token alignment automatically

### For Fine-tuned Models (marian_finetuned)
- Uses saved tokenizer from model directory if available
- Falls back to Georgian corpus tokenizer if not found

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

# 4. Split data
python -m src.splitter.runner \
  --description "Create train/valid/test splits"
# Creates: splits:latest

# 5. Train model
python -m src.training.runner train \
  --config config/training/baseline_marian.yaml \
  --splits-artifact-version "latest" \
  --description "Train baseline model"
# Creates: training:latest
```

## CLI Commands

### Training Command

```bash
python -m src.training.runner train [OPTIONS]

Options:
  --config TEXT                    Path to training configuration YAML [required]
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

## Example Experiments

### Baseline Comparison

```bash
# Train baseline Marian from scratch
python -m src.training.runner train \
  --config config/training/baseline_marian.yaml \
  --splits-artifact-version "latest" \
  --description "Baseline Marian model"

# Fine-tune pretrained model
python -m src.training.runner train \
  --config config/training/pretrained_marian.yaml \
  --splits-artifact-version "latest" \
  --description "Pretrained Marian fine-tuning"
```

### Architecture Exploration

```bash
# Encoder-decoder experiment
python -m src.training.runner train \
  --config config/training/encoder_decoder.yaml \
  --splits-artifact-version "latest" \
  --description "BERT-GPT2 encoder-decoder"
```

## Configuration Examples

### Custom Model Configuration (Uses Georgian Tokenizer)
```yaml
model:
  type: "marian_custom"
  architecture:
    d_model: 512
    encoder_layers: 6
    # ... other params
```

### Pretrained Model Configuration (Uses Model's Tokenizer)
```yaml
model:
  type: "marian_pretrained"
  model_name: "facebook/m2m100_418M"
  target_lang: "ka"
```

### Encoder-Decoder Configuration (Uses Decoder's Tokenizer)
```yaml
model:
  type: "encoder_decoder_pretrained"
  encoder_model: "bert-base-uncased"
  decoder_model: "gpt2"
```

## Troubleshooting

### Common Issues

1. **Splits artifact not found**: Ensure you've run the splitter pipeline
2. **Model configuration errors**: Use dry-run mode for validation
3. **Tokenizer issues**: The pipeline handles this automatically
4. **Memory issues**: Reduce batch sizes or sequence length

### Debug Mode

```bash
python -m src.training.runner train \
  --config config/training/baseline_marian.yaml \
  --splits-artifact-version "latest" \
  --description "Debug run" \
  --develop \
  --dry-run
```

This training pipeline provides a flexible foundation for neural machine translation experiments while automatically handling tokenizer selection and maintaining reproducibility.