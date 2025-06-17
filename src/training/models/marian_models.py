"""
Marian Model Implementations

Provides various Marian-based model configurations including pretrained models,
custom architectures, and multilingual models.
"""

from typing import Dict, Any, Tuple

import torch
from transformers import (
    MarianConfig,
    MarianMTModel,
    M2M100ForConditionalGeneration,
    MBartForConditionalGeneration,
    GenerationConfig,
    DataCollatorForSeq2Seq,
    PreTrainedTokenizer
)

from ..registry.model_registry import register_model


@register_model("marian_pretrained", "Load pretrained Marian/multilingual NMT model from HuggingFace")
def create_marian_pretrained(config: Dict[str, Any],
                             tokenizer: PreTrainedTokenizer) -> Tuple[
    torch.nn.Module, GenerationConfig, DataCollatorForSeq2Seq]:
    """
    Create a pretrained Marian or multilingual model.

    Args:
        config: Model configuration containing 'model_name' key
        tokenizer: Tokenizer for the model

    Returns:
        Tuple of (model, generation_config, data_collator)
    """
    model_name = config.get('model_name', 'facebook/m2m100_418M')
    target_lang = config.get('target_lang', 'ka')  # Georgian

    # Determine model type based on model name
    if 'm2m100' in model_name.lower():
        # M2M100 multilingual model
        model = M2M100ForConditionalGeneration.from_pretrained(model_name)

        # Create generation config for M2M100
        generation_config = GenerationConfig.from_model_config(model.config)

        # Set forced_bos_token_id for target language (Georgian)
        if hasattr(tokenizer, 'get_lang_id'):
            try:
                target_lang_id = tokenizer.get_lang_id(target_lang)
                generation_config.forced_bos_token_id = target_lang_id
                print(f"Set forced_bos_token_id to {target_lang_id} for language '{target_lang}'")
            except Exception as e:
                print(f"Warning: Could not set target language ID for '{target_lang}': {e}")

    elif 'mbart' in model_name.lower():
        # mBART multilingual model
        model = MBartForConditionalGeneration.from_pretrained(model_name)

        # Create generation config for mBART
        generation_config = GenerationConfig.from_model_config(model.config)

        # Set forced_bos_token_id for target language
        if hasattr(tokenizer, 'lang_code_to_id'):
            target_lang_code = f"{target_lang}_GE"  # Georgian: ka_GE
            if target_lang_code in tokenizer.lang_code_to_id:
                target_lang_id = tokenizer.lang_code_to_id[target_lang_code]
                generation_config.forced_bos_token_id = target_lang_id
                print(f"Set forced_bos_token_id to {target_lang_id} for language '{target_lang_code}'")
            else:
                print(f"Warning: Language code '{target_lang_code}' not found in tokenizer")

    else:
        # Standard Marian model
        model = MarianMTModel.from_pretrained(model_name)
        generation_config = GenerationConfig.from_model_config(model.config)

    # Update generation config with custom parameters
    generation_config.update(**config.get('generation_config', {}))

    # Move to device if specified
    device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

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


@register_model("m2m100_multilingual", "M2M100 multilingual translation model")
def create_m2m100_multilingual(config: Dict[str, Any],
                               tokenizer: PreTrainedTokenizer) -> Tuple[
    M2M100ForConditionalGeneration, GenerationConfig, DataCollatorForSeq2Seq]:
    """
    Create M2M100 multilingual model specifically configured for translation.

    Args:
        config: Model configuration
        tokenizer: M2M100Tokenizer

    Returns:
        Tuple of (model, generation_config, data_collator)
    """
    model_name = config.get('model_name', 'facebook/m2m100_418M')
    source_lang = config.get('source_lang', 'en')
    target_lang = config.get('target_lang', 'ka')

    # Load model
    model = M2M100ForConditionalGeneration.from_pretrained(model_name)

    # Set source language in tokenizer
    if hasattr(tokenizer, 'src_lang'):
        tokenizer.src_lang = source_lang

    # Create generation config
    generation_config = GenerationConfig.from_model_config(model.config)

    # Set forced_bos_token_id for target language
    if hasattr(tokenizer, 'get_lang_id'):
        try:
            target_lang_id = tokenizer.get_lang_id(target_lang)
            generation_config.forced_bos_token_id = target_lang_id
            print(f"M2M100: Set target language to '{target_lang}' (ID: {target_lang_id})")
        except Exception as e:
            print(f"Warning: Could not set target language '{target_lang}': {e}")

    # Update with custom generation parameters
    generation_config.update(**config.get('generation_config', {}))

    # Move to device if specified
    device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Create data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        padding=True
    )

    return model, generation_config, data_collator