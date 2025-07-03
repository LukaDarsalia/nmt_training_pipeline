"""
Encoder-Decoder Model Implementation

Provides clean encoder-decoder model implementation using various pretrained models.
"""

from typing import Dict, Any, Tuple
import torch
from transformers import (
    EncoderDecoderModel,
    AutoModel,
    BertConfig,
    BertLMHeadModel,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.generation.configuration_utils import GenerationConfig
from transformers.data.data_collator import DataCollatorForSeq2Seq
from src.training.tokenizers.encoder_decoder_tokenizers import EncoderDecoderTokenizer
from ..registry.model_registry import register_model


@register_model("encoder_decoder_pretrained", "Create EncoderDecoder model from pretrained encoder and decoder")
def create_encoder_decoder_pretrained(config: Dict[str, Any],
                                      tokenizer: EncoderDecoderTokenizer) -> Tuple[
    PreTrainedModel, GenerationConfig, DataCollatorForSeq2Seq]:
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

    print(f"Creating EncoderDecoder model:")
    print(f"  Encoder: {encoder_model}")
    print(f"  Decoder: {decoder_model}")

    # Create encoder-decoder model
    model = EncoderDecoderModel.from_encoder_decoder_pretrained(
        encoder_model,
        decoder_model
    )

    # Configure model tokens
    model.config.decoder_start_token_id = tokenizer.bos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.decoder.bos_token_id = tokenizer.bos_token_id
    model.config.encoder.bos_token_id = tokenizer.bos_token_id

    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.decoder.pad_token_id = tokenizer.pad_token_id
    model.config.encoder.pad_token_id = tokenizer.pad_token_id

    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.decoder.eos_token_id = tokenizer.eos_token_id
    model.config.encoder.eos_token_id = tokenizer.eos_token_id
    model.config.decoder.eos_token_id = tokenizer.eos_token_id

    # Move to device if specified
    device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Create generation config
    generation_config = GenerationConfig.from_model_config(model.config)
    print(config)
    print(generation_config)
    generation_config.max_length = config.get('generation_config', {}).get('max_length', 128)
    generation_config.num_beams = config.get('generation_config', {}).get('num_beams', 1)
    generation_config.early_stopping = config.get('generation_config', {}).get('early_stopping', False)
    generation_config.do_sample = config.get('generation_config', {}).get('do_sample', False)
    print(generation_config)
    generation_config.pad_token_id = int(tokenizer.decoder.pad_token_id)
    generation_config.bos_token_id = int(tokenizer.decoder.bos_token_id)
    generation_config.eos_token_id = int(tokenizer.decoder.eos_token_id)

    # Create data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        padding=True
    )
    print(f"Generation config: {generation_config}")
    print(f"Model config: {model.config}")

    print(f"Created EncoderDecoder model:")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    return model, generation_config, data_collator

@register_model("encoder_decoder_pretrained_encoder_custom_decoder", "Create EncoderDecoder model from pretrained encoder and custom decoder config")
def create_encoder_decoder_pretrained_encoder_custom_decoder(config: Dict[str, Any],
                                      tokenizer: EncoderDecoderTokenizer) -> Tuple[
    PreTrainedModel, GenerationConfig, DataCollatorForSeq2Seq]:
    """
    Create an encoder-decoder model with a pretrained encoder and a randomly initialized decoder with custom config.

    Args:
        config: Model configuration containing 'encoder_model' and 'decoder_config' keys
        tokenizer: Tokenizer for the model

    Returns:
        Tuple of (model, generation_config, data_collator)
    """
    encoder_model_name = config.get('encoder_model', 'bert-base-uncased')
    decoder_config_dict = config.get('decoder_config', {})
    decoder_config = BertConfig(**decoder_config_dict)
    decoder_config.vocab_size = tokenizer.decoder.vocab_size

    print(f"Creating EncoderDecoder model with pretrained encoder and custom decoder config:")
    print(f"  Encoder: {encoder_model_name}")
    print(f"  Decoder config: {decoder_config}")

    # Load pretrained encoder
    encoder = AutoModel.from_pretrained(encoder_model_name)
    # Create decoder from config (randomly initialized)
    decoder = BertLMHeadModel(decoder_config)
    print(f"decoder config: {decoder.config}")
    
    # Create encoder-decoder model
    model = EncoderDecoderModel(encoder=encoder, decoder=decoder)

    # Configure model tokens
    model.config.decoder_start_token_id = tokenizer.decoder.bos_token_id
    model.config.decoder.bos_token_id = tokenizer.decoder.bos_token_id
    model.config.eos_token_id = tokenizer.decoder.eos_token_id
    model.config.decoder.eos_token_id = tokenizer.decoder.eos_token_id

    model.config.pad_token_id = tokenizer.pad_token_id

    # Move to device if specified
    device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Create generation config
    generation_config = GenerationConfig.from_model_config(model.config)
    generation_config.max_length = config.get('generation_config', {}).get('max_length', 128)
    generation_config.num_beams = config.get('generation_config', {}).get('num_beams', 1)
    generation_config.early_stopping = config.get('generation_config', {}).get('early_stopping', False)
    generation_config.do_sample = config.get('generation_config', {}).get('do_sample', False)
    generation_config.pad_token_id = int(tokenizer.pad_token_id)
    generation_config.bos_token_id = int(tokenizer.decoder.bos_token_id)
    generation_config.eos_token_id = int(tokenizer.decoder.eos_token_id)

    # Create data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        padding=True
    )
    print(f"Generation config: {generation_config}")
    print(f"Model config: {model.config}")

    print(f"Created EncoderDecoder model:")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    return model, generation_config, data_collator
