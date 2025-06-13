# Training Module

A registry-driven training pipeline built on HuggingFace Transformers. The goal
is to make experimenting with new models and architectures as easy as editing a
YAML file.

## Key Features

- **Model Registry** – Register different model creation functions. Default
  options include loading any seq2seq model via the `auto` architecture,
  a `marian_custom` option for Marian models built from scratch with custom
  tokenizers and an `e5_m2m` encoder-decoder using multilingual-e5-large
  weights.
- **Trainer Registry** – Swap out the training loop implementation. By default
  the pipeline uses `Seq2SeqTrainer` with an AdamW optimizer and a cosine
-  scheduler.
- **Metric Registry** – Register evaluation metrics such as BLEU, chrF++
  or COMET and reference them from the config.
- **Early Stopping** – Built-in support via configuration options for
  patience and threshold.
- **YAML Configuration** – Hyperparameters, model choice and trainer settings are
  all controlled from `config/training.yaml` and logged to Weights & Biases.

## Quick Start

```bash
python -m src.trainer.runner \
  --splits-artifact-version latest \
  --description "Train baseline model" \
  --config config/marian_baseline.yaml
```

The command downloads the specified `splits` artifact from WandB, creates the
model defined in the config, trains it and logs the resulting model as a new
artifact.

For the advanced experiment using `multilingual-e5-large` run:

```bash
python -m src.trainer.runner \
  --splits-artifact-version latest \
  --description "Train e5 encoder-decoder" \
  --config config/e5_encoder_decoder.yaml
```

## Adding New Architectures

1. Implement a function in `src/trainer/models.py` and decorate it with
   `@register_model("my_arch")`.
2. Update `config/training.yaml` with `architecture: "my_arch"` and any custom
   parameters your function expects.

## Changing the Training Loop

Implement a trainer creation function and register it with
`@register_trainer("my_trainer")`. Reference it in the config under
`trainer.trainer_type`.

## Custom Metrics

Metrics are defined in `src/trainer/evaluation.py` and registered with
`@register_metric`. List the desired metrics in `trainer.metrics` within the
config file. During validation the scores will be logged to wandb.
