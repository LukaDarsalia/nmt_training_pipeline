"""
Encoder-Decoder Model Implementations

Provides encoder-decoder model combinations using various pretrained models
from HuggingFace transformers library.
"""

from typing import Dict, Any, Tuple
import torch
from transformers import (
    EncoderDecoderModel,
    AutoTokenizer,
    GenerationConfig,
    DataCollatorForSeq2Seq,
    PreTrainedTokenizer
)
from ..registry.model_registry import register_model


@register_model("encoder_decoder_pretrained", "Create EncoderDecoder model from pretrained encoder and decoder")
def create_encoder_decoder_pretrained(config: Dict[str, Any],
                                      tokenizer: PreTrainedTokenizer) -> Tuple[
    EncoderDecoderModel, GenerationConfig, DataCollatorForSeq2Seq]:
    """
    Create an encoder-decoder model from pretrained encoder and decoder models.

    Args:
        config: Model configuration containing 'encoder_model' and 'decoder_model' keys
        tokenizer: Tokenizer for the model

    Returns:
        Tuple of (model, generation_config, data_collator)
    """
    encoder_model = config.get('encoder_model', 'bert-base-uncased')
    decoder_model = config.get('decoder_model', 'bert-base-uncased')

    # Create encoder-decoder model
    model = EncoderDecoderModel.from_encoder_decoder_pretrained(
        encoder_model,
        decoder_model
    )

    # Configure model tokens
    model.config.decoder_start_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id

    # Update decoder config
    model.config.decoder.bos_token_id = tokenizer.bos_token_id
    model.config.decoder.pad_token_id = tokenizer.pad_token_id
    model.config.decoder.eos_token_id = tokenizer.eos_token_id

    # Move to device if specified
    device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Create generation config
    generation_config = GenerationConfig.from_model_config(model.config)
    generation_config.update(**config.get('generation_config', {}))

    # Create data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        padding=True
    )

    return model, generation_config, data_collator


@register_model("encoder_decoder_random", "Create randomly initialized EncoderDecoder model")
def create_encoder_decoder_random(config: Dict[str, Any],
                                  tokenizer: PreTrainedTokenizer) -> Tuple[
    EncoderDecoderModel, GenerationConfig, DataCollatorForSeq2Seq]:
    """
    Create a randomly initialized encoder-decoder model using specified architectures.

    Args:
        config: Model configuration with encoder/decoder architecture specifications
        tokenizer: Tokenizer for the model

    Returns:
        Tuple of (model, generation_config, data_collator)
    """
    encoder_model = config.get('encoder_model', 'bert-base-uncased')
    decoder_model = config.get('decoder_model', 'bert-base-uncased')

    # Create encoder-decoder model with random initialization
    model = EncoderDecoderModel.from_encoder_decoder_pretrained(
        encoder_model,
        decoder_model
    )

    # Randomly initialize weights if specified
    if config.get('random_init', True):
        model.apply(model._init_weights)

    # Configure model tokens
    model.config.decoder_start_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id

    # Update decoder config
    model.config.decoder.bos_token_id = tokenizer.bos_token_id
    model.config.decoder.pad_token_id = tokenizer.pad_token_id
    model.config.decoder.eos_token_id = tokenizer.eos_token_id

    # Move to device if specified
    device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Create generation config
    generation_config = GenerationConfig.from_model_config(model.config)
    generation_config.update(**config.get('generation_config', {}))

    # Create data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        padding=True
    )

    return model, generation_config, data_collator


@register_model("encoder_decoder_mixed", "Create EncoderDecoder with different encoder/decoder initialization")
def create_encoder_decoder_mixed(config: Dict[str, Any],
                                 tokenizer: PreTrainedTokenizer) -> Tuple[
    EncoderDecoderModel, GenerationConfig, DataCollatorForSeq2Seq]:
    """
    Create an encoder-decoder model with mixed initialization strategies.

    Args:
        config: Model configuration with fine-grained control over initialization
        tokenizer: Tokenizer for the model

    Returns:
        Tuple of (model, generation_config, data_collator)
    """
    encoder_model = config.get('encoder_model', 'bert-base-uncased')
    decoder_model = config.get('decoder_model', 'bert-base-uncased')

    # Create encoder-decoder model
    model = EncoderDecoderModel.from_encoder_decoder_pretrained(
        encoder_model,
        decoder_model
    )

    # Apply different initialization strategies
    if config.get('random_init_encoder', False):
        model.encoder.apply(model.encoder._init_weights)

    if config.get('random_init_decoder', False):
        model.decoder.apply(model.decoder._init_weights)

    # Configure model tokens
    model.config.decoder_start_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id

    # Update decoder config
    model.config.decoder.bos_token_id = tokenizer.bos_token_id
    model.config.decoder.pad_token_id = tokenizer.pad_token_id
    model.config.decoder.eos_token_id = tokenizer.eos_token_id

    # Move to device if specified
    device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Create generation config
    generation_config = GenerationConfig.from_model_config(model.config)
    generation_config.update(**config.get('generation_config', {}))

    # Create data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        padding=True
    )

    return model, generation_config, data_collator