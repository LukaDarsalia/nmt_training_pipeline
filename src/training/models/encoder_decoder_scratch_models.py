"""
Encoder-Decoder Scratch Model Implementation

Provides encoder-decoder models built from scratch for translation tasks.
"""

from math import e
from typing import Dict, Any, Tuple

import torch
from transformers import (
    EncoderDecoderConfig,
    EncoderDecoderModel,
    AutoConfig,
)
from transformers.generation.configuration_utils import GenerationConfig
from transformers.data.data_collator import DataCollatorForSeq2Seq
from src.training.tokenizers.encoder_decoder_tokenizers import EncoderDecoderTokenizer

from ..registry.model_registry import register_model


@register_model("encoder_decoder_scratch", "Create encoder-decoder model from scratch with custom architecture")
def create_encoder_decoder_scratch(config: Dict[str, Any],
                                   tokenizer: EncoderDecoderTokenizer) -> Tuple[
    EncoderDecoderModel, GenerationConfig, DataCollatorForSeq2Seq]:
    """
    Create an encoder-decoder model from scratch with specified architecture.

    Args:
        config: Model configuration with architecture parameters
        tokenizer: EncoderDecoderTokenizer for the model

    Returns:
        Tuple of (model, generation_config, data_collator)
    """
    # Get model architecture parameters
    model_params = config.get('architecture', {})
    
    # Validate tokenizer properties
    if not hasattr(tokenizer, 'encoder') or not hasattr(tokenizer, 'decoder'):
        raise ValueError("Tokenizer must have encoder and decoder tokenizers")
    
    print(f"Creating EncoderDecoder model from scratch:")
    print(f"  Encoder vocab size: {tokenizer.encoder.vocab_size}")
    print(f"  Decoder vocab size: {tokenizer.decoder.vocab_size}")
    
    # Get encoder and decoder base models for architecture
    encoder_model_name = config.get('encoder_model', 'openai-community/gpt2')
    decoder_model_name = config.get('decoder_model', 'openai-community/gpt2')
    
    # Load base configurations
    encoder_config = AutoConfig.from_pretrained(encoder_model_name)
    decoder_config = AutoConfig.from_pretrained(decoder_model_name)
    
    
    # Update configurations with custom parameters
    encoder_config.vocab_size = tokenizer.encoder.vocab_size
    decoder_config.vocab_size = tokenizer.decoder.vocab_size
    
    # Apply architecture parameters
    if 'd_model' in model_params:
        if hasattr(encoder_config, 'hidden_size'):
            encoder_config.hidden_size = model_params['d_model']
        elif hasattr(encoder_config, 'd_model'):
            encoder_config.d_model = model_params['d_model']
        elif hasattr(encoder_config, 'n_embd'):
            encoder_config.n_embd = model_params['d_model']

        if hasattr(decoder_config, 'hidden_size'):
            decoder_config.hidden_size = model_params['d_model']
        elif hasattr(decoder_config, 'd_model'):
            decoder_config.d_model = model_params['d_model']
        elif hasattr(decoder_config, 'n_embd'):
            decoder_config.n_embd = model_params['d_model']
    
    if 'encoder_layers' in model_params:
        if hasattr(encoder_config, 'num_hidden_layers'):
            encoder_config.num_hidden_layers = model_params['encoder_layers']
        elif hasattr(encoder_config, 'n_layer'):
            encoder_config.n_layer = model_params['encoder_layers']
    
    if 'decoder_layers' in model_params:
        if hasattr(decoder_config, 'num_hidden_layers'):
            decoder_config.num_hidden_layers = model_params['decoder_layers']
        elif hasattr(decoder_config, 'n_layer'):
            decoder_config.n_layer = model_params['decoder_layers']
    
    if 'encoder_attention_heads' in model_params:
        if hasattr(encoder_config, 'num_attention_heads'):
            encoder_config.num_attention_heads = model_params['encoder_attention_heads']
        elif hasattr(encoder_config, 'n_head'):
            encoder_config.n_head = model_params['encoder_attention_heads']
    
    if 'decoder_attention_heads' in model_params:
        if hasattr(decoder_config, 'num_attention_heads'):
            decoder_config.num_attention_heads = model_params['decoder_attention_heads']
        elif hasattr(decoder_config, 'n_head'):
            decoder_config.n_head = model_params['decoder_attention_heads']
    
    if 'dropout' in model_params:
        if hasattr(encoder_config, 'hidden_dropout_prob'):
            encoder_config.hidden_dropout_prob = model_params['dropout']
        if hasattr(decoder_config, 'hidden_dropout_prob'):
            decoder_config.hidden_dropout_prob = model_params['dropout']
        if hasattr(encoder_config, 'resid_pdrop'):
            encoder_config.resid_pdrop = model_params['dropout']
        if hasattr(decoder_config, 'resid_pdrop'):
            decoder_config.resid_pdrop = model_params['dropout']
    
    # Set token IDs
    if hasattr(tokenizer, 'pad_token_id') and tokenizer.pad_token_id is not None:
        encoder_config.pad_token_id = tokenizer.pad_token_id
        decoder_config.pad_token_id = tokenizer.decoder.pad_token_id
    
    if hasattr(tokenizer, 'eos_token_id') and tokenizer.eos_token_id is not None:
        encoder_config.eos_token_id = tokenizer.eos_token_id
        decoder_config.eos_token_id = tokenizer.decoder.eos_token_id
    
    if hasattr(tokenizer, 'bos_token_id') and tokenizer.bos_token_id is not None:
        encoder_config.bos_token_id = tokenizer.bos_token_id
        decoder_config.bos_token_id = tokenizer.decoder.bos_token_id
    
    # Create encoder-decoder configuration
    encoder_decoder_config = EncoderDecoderConfig.from_encoder_decoder_configs(
        encoder_config, decoder_config
    )
    
    # Set special tokens in the main config
    if hasattr(tokenizer, 'pad_token_id') and tokenizer.pad_token_id is not None:
        encoder_decoder_config.pad_token_id = tokenizer.pad_token_id
    if hasattr(tokenizer, 'eos_token_id') and tokenizer.eos_token_id is not None:
        encoder_decoder_config.eos_token_id = tokenizer.decoder.eos_token_id
    if hasattr(tokenizer, 'bos_token_id') and tokenizer.bos_token_id is not None:
        encoder_decoder_config.decoder_start_token_id = tokenizer.decoder.bos_token_id
        encoder_decoder_config.bos_token_id = tokenizer.decoder.bos_token_id
    
    # Create model from scratch (randomly initialized)
    model = EncoderDecoderModel(encoder_decoder_config)
    
    print("Created EncoderDecoder model from scratch")
    
    # Move to device if specified
    device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Create generation config
    generation_config = GenerationConfig.from_model_config(encoder_decoder_config)
    generation_config.update(**config.get('generation_config', {}))

    # Create data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        padding=True
    )

    print(f"Created EncoderDecoder model from scratch:")
    print(f"  Architecture: encoder_layers={encoder_config.num_hidden_layers if hasattr(encoder_config, 'num_hidden_layers') else encoder_config.n_layer}, decoder_layers={decoder_config.num_hidden_layers if hasattr(decoder_config, 'num_hidden_layers') else decoder_config.n_layer}")
    print(f"  Encoder vocab size: {encoder_config.vocab_size}")
    print(f"  Decoder vocab size: {decoder_config.vocab_size}")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    return model, generation_config, data_collator 