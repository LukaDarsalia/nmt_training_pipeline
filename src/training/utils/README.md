# Training Utilities - Enhanced WandB Evaluation Logging

This module provides enhanced WandB logging capabilities for neural machine translation training, allowing you to track detailed evaluation results including source texts, translations, references, and individual metric scores.

## Features

### WandbEvaluationCallback

The `WandbEvaluationCallback` is an enhanced version of the original prediction logging callback that provides comprehensive evaluation result logging to WandB.

#### Features:

1. **Comprehensive Data Logging**:
   - Source sentences (input text)
   - Reference translations (ground truth)
   - Model predictions (generated translations)
   - Tokenized inputs (for debugging)

2. **Individual Metric Scores**:
   - BLEU scores for each sample
   - chrF++ scores for each sample
   - Georgian COMET scores for each sample (when sources are available)

3. **Summary Statistics**:
   - Average text lengths (source, reference, prediction)
   - Metric statistics (mean, std, min, max)
   - Sample count tracking

4. **Flexible Configuration**:
   - Configurable sample count and logging frequency
   - Support for different tokenizer types (including EncoderDecoder)
   - Customizable source/target column names

## Configuration

### Basic Configuration

```yaml
training:
  prediction_logging:
    enabled: true
    num_samples: 30        # Number of samples to log
    frequency: 1           # Log every N evaluation steps
```

### Advanced Configuration

```yaml
data:
  source_column: "en"      # Source language column name
  target_column: "ka"      # Target language column name

training:
  prediction_logging:
    enabled: true
    num_samples: 50        # More samples for detailed analysis
    frequency: 1           # Log every evaluation for complete tracking

generation:
  max_length: 256
  eval_num_beams: 5        # Number of beams for evaluation generation
```

## WandB Tables

The callback creates detailed WandB tables with the following columns:

| Column | Description |
|--------|-------------|
| step | Training step number |
| source | Source sentence (input text) |
| reference | Reference translation (ground truth) |
| prediction | Model prediction (generated translation) |
| input_tokens | Tokenized input (for debugging) |
| bleu_score | Individual BLEU score for this sample |
| chrf_score | Individual chrF++ score for this sample |
| comet_score | Individual Georgian COMET score for this sample |

## WandB Metrics

Additional metrics logged to WandB:

### Summary Statistics
- `eval_samples_count`: Number of samples in the evaluation batch
- `avg_prediction_length`: Average length of predictions
- `avg_reference_length`: Average length of references
- `avg_source_length`: Average length of source sentences

### Individual Metric Statistics
- `eval_bleu_mean`: Mean BLEU score across samples
- `eval_bleu_std`: Standard deviation of BLEU scores
- `eval_bleu_min`: Minimum BLEU score
- `eval_bleu_max`: Maximum BLEU score

Similar statistics are logged for chrF++ and COMET scores.

## Usage Examples

### In Training Configuration

```yaml
trainer:
  type: "seq2seq_with_metrics"
  
  training:
    # ... other training parameters ...
    
    prediction_logging:
      enabled: true
      num_samples: 30
      frequency: 1

  evaluation:
    evaluators:
      - name: "sacrebleu"
        config: {}
      - name: "chrf"
        config:
          word_order: 2
      - name: "georgian_comet"
        config:
          model_name: "Darsala/georgian_comet"
          batch_size: 16
          device: "cuda"
          gpus: 1
```

### Programmatic Usage

```python
from src.training.utils.callbacks import WandbEvaluationCallback

callback = WandbEvaluationCallback(
    tokenizer=tokenizer,
    eval_dataset=eval_dataset,
    compute_metrics_fn=compute_metrics_fn,
    num_samples=50,
    log_frequency=1,
    max_length=256,
    num_beams=5,
    source_column='en',
    target_column='ka'
)
```

## Benefits

1. **Detailed Analysis**: See exactly how your model performs on individual examples
2. **Error Analysis**: Identify patterns in translation errors
3. **Progress Tracking**: Monitor improvement on specific examples over time
4. **Model Comparison**: Compare different models' outputs side-by-side
5. **Debugging**: Use tokenized inputs to debug tokenization issues

## Best Practices

1. **Sample Size**: Use 20-50 samples for regular monitoring, more for detailed analysis
2. **Frequency**: Log every evaluation step for experiments, every 2-3 steps for long training
3. **Metric Selection**: Include both automatic metrics (BLEU, chrF, COMET) for comprehensive evaluation
4. **Source Availability**: Ensure source sentences are available for COMET evaluation

## Troubleshooting

### Common Issues

1. **Missing Source Sentences**: 
   - Check that `source_column` matches your dataset
   - Verify dataset has the correct structure

2. **COMET Scoring Errors**:
   - Ensure CUDA is available if using GPU
   - Check that the Georgian COMET model is accessible

3. **Memory Issues**:
   - Reduce `num_samples` if running out of memory
   - Use smaller batch sizes for individual metric computation

### Performance Tips

1. Use smaller sample sizes during development
2. Set frequency > 1 for very frequent evaluations
3. Consider disabling individual metric computation for very large evaluations

## Backward Compatibility

The `WandbPredictionProgressCallback` class is maintained as an alias to `WandbEvaluationCallback` for backward compatibility with existing configurations. 