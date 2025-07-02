"""
Marian Model Implementation

Provides clean Marian-based model implementation for translation tasks.
"""

from typing import Dict, Any, Tuple

import torch
from transformers import (
    MarianConfig,
    MarianMTModel,
)
from transformers.generation.configuration_utils import GenerationConfig
from transformers.data.data_collator import DataCollatorForSeq2Seq
from src.training.tokenizers.encoder_decoder_tokenizers import EncoderDecoderTokenizer

from ..registry.model_registry import register_model


@register_model("marian_custom", "Create custom Marian model with specified architecture")
def create_marian_custom(config: Dict[str, Any],
                         tokenizer: EncoderDecoderTokenizer) -> Tuple[
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
    
    # Validate tokenizer properties
    if not hasattr(tokenizer, 'bos_token_id') or tokenizer.bos_token_id is None:
        raise ValueError("Tokenizer must have bos_token_id")
    if not hasattr(tokenizer, 'pad_token_id') or tokenizer.pad_token_id is None:
        raise ValueError("Tokenizer must have pad_token_id")
    if not hasattr(tokenizer, 'eos_token_id') or tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer must have eos_token_id")
    print("max(tokenizer.encoder.vocab_size, tokenizer.decoder.vocab_size)", max(tokenizer.encoder.vocab_size, tokenizer.decoder.vocab_size))
    # Create model configuration
    marian_config = MarianConfig(
        vocab_size=max(tokenizer.decoder.vocab_size, tokenizer.encoder.vocab_size),
        max_position_embeddings=config.get('max_length', 512),
        max_length=config.get('max_length', 512),
        pad_token_id=int(tokenizer.encoder.pad_token_id),
        eos_token_id=int(tokenizer.decoder.eos_token_id),
        bos_token_id=int(tokenizer.encoder.bos_token_id),
        decoder_start_token_id=int(tokenizer.decoder.bos_token_id),
        forced_eos_token_id=int(tokenizer.encoder.eos_token_id),
        share_encoder_decoder_embeddings=False,

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

    model.model.decoder.padding_idx = int(tokenizer.decoder.pad_token_id)

    print("config", config)
    # Move to device if specified
    device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Moving model to device: {device}")
    model = model.to(device)

    # Create generation config
    generation_config = GenerationConfig.from_model_config(marian_config)
    generation_config.max_length = config.get('generation_config', {}).get('max_length', 128)
    generation_config.num_beams = config.get('generation_config', {}).get('num_beams', 1)
    generation_config.early_stopping = config.get('generation_config', {}).get('early_stopping', False)
    generation_config.do_sample = config.get('generation_config', {}).get('do_sample', False)
    
    generation_config.pad_token_id = int(tokenizer.decoder.pad_token_id)
    generation_config.bos_token_id = int(tokenizer.decoder.bos_token_id)
    generation_config.eos_token_id = int(tokenizer.decoder.eos_token_id)

    # Create data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        padding=True
    )

    print(f"Created Marian model:")
    print(f"  Architecture: d_model={marian_config.d_model}, layers={marian_config.encoder_layers}/{marian_config.decoder_layers}")
    print(f"  Vocab size: {marian_config.vocab_size}")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    return model, generation_config, data_collator
