"""
Marian Model Implementations

Provides various Marian-based model configurations including pretrained models,
custom architectures, and randomly initialized models.
"""

from typing import Dict, Any, Tuple
import torch
from transformers import (
    MarianConfig,
    MarianMTModel,
    MarianTokenizer,
    GenerationConfig,
    DataCollatorForSeq2Seq,
    PreTrainedTokenizer
)
from ..registry.model_registry import register_model


@register_model("marian_pretrained", "Load pretrained Marian NMT model from HuggingFace")
def create_marian_pretrained(config: Dict[str, Any],
                             tokenizer: PreTrainedTokenizer) -> Tuple[
    MarianMTModel, GenerationConfig, DataCollatorForSeq2Seq]:
    """
    Create a pretrained Marian model.

    Args:
        config: Model configuration containing 'model_name' key
        tokenizer: Tokenizer for the model

    Returns:
        Tuple of (model, generation_config, data_collator)
    """
    model_name = config.get('model_name', 'Helsinki-NLP/opus-mt-en-ka')

    # Load pretrained model
    model = MarianMTModel.from_pretrained(model_name)

    # Create generation config
    generation_config = GenerationConfig.from_model_config(model.config)

    # Update generation config with custom parameters
    generation_config.update(**config.get('generation_config', {}))

    # Create data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        padding=True
    )

    return model, generation_config, data_collator


@register_model("marian_custom", "Create custom Marian model with specified architecture")
def create_marian_custom(config: Dict[str, Any],
                         tokenizer: PreTrainedTokenizer) -> Tuple[
    MarianMTModel, GenerationConfig, DataCollatorForSeq2Seq]:
    """
    Create a custom Marian model with specified architecture.

    Args:
        config: Model configuration with architecture parameters
        tokenizer: Tokenizer for the model

    Returns:
        Tuple of (model, generation_config, data_collator)
    """
    # Get model architecture parameters
    model_params = config.get('architecture', {})

    # Create model configuration
    marian_config = MarianConfig(
        vocab_size=tokenizer.vocab_size,
        max_position_embeddings=config.get('max_length', 512),
        max_length=config.get('max_length', 512),
        decoder_start_token_id=tokenizer.bos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
        forced_eos_token_id=tokenizer.eos_token_id,

        # Architecture parameters
        d_model=model_params.get('d_model', 512),
        encoder_layers=model_params.get('encoder_layers', 6),
        decoder_layers=model_params.get('decoder_layers', 6),
        encoder_ffn_dim=model_params.get('encoder_ffn_dim', 2048),
        decoder_ffn_dim=model_params.get('decoder_ffn_dim', 2048),
        encoder_attention_heads=model_params.get('encoder_attention_heads', 8),
        decoder_attention_heads=model_params.get('decoder_attention_heads', 8),
        dropout=model_params.get('dropout', 0.1),

        # Additional parameters
        add_bias_logits=False,
        add_final_layer_norm=False,
        static_position_embeddings=True,
        scale_embedding=True,
        activation_function=model_params.get('activation_function', 'swish'),
    )

    # Create model
    model = MarianMTModel(marian_config)

    # Move to device if specified
    device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Create generation config
    generation_config = GenerationConfig.from_model_config(marian_config)
    generation_config.update(**config.get('generation_config', {}))

    # Create data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        padding=True
    )

    return model, generation_config, data_collator


@register_model("marian_finetuned", "Load finetuned Marian model from local path")
def create_marian_finetuned(config: Dict[str, Any],
                            tokenizer: PreTrainedTokenizer) -> Tuple[
    MarianMTModel, GenerationConfig, DataCollatorForSeq2Seq]:
    """
    Load a finetuned Marian model from local path.

    Args:
        config: Model configuration containing 'model_path' key
        tokenizer: Tokenizer for the model

    Returns:
        Tuple of (model, generation_config, data_collator)
    """
    model_path = config.get('model_path')
    if not model_path:
        raise ValueError("model_path must be specified for marian_finetuned")

    # Load configuration and model
    marian_config = MarianConfig.from_pretrained(model_path)
    model = MarianMTModel.from_pretrained(model_path, config=marian_config)

    # Move to device if specified
    device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Create generation config
    generation_config = GenerationConfig.from_model_config(marian_config)
    generation_config.update(**config.get('generation_config', {}))

    # Create data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        padding=True
    )

    return model, generation_config, data_collator