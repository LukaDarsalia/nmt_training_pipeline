"""
M2M100 Model Implementation

Provides clean M2M100-based model implementation for multilingual translation.
"""

from typing import Dict, Any, Tuple

import torch
from transformers import (
    M2M100ForConditionalGeneration,
    M2M100Tokenizer,
)
from transformers.generation.configuration_utils import GenerationConfig
from transformers.data.data_collator import DataCollatorForSeq2Seq

from ..registry.model_registry import register_model


@register_model("m2m100_pretrained", "M2M100 pretrained multilingual translation model")
def create_m2m100_pretrained(config: Dict[str, Any],
                             tokenizer: M2M100Tokenizer) -> Tuple[
    M2M100ForConditionalGeneration, GenerationConfig, DataCollatorForSeq2Seq]:
    """
    Create M2M100 pretrained multilingual model.

    Args:
        config: Model configuration
        tokenizer: M2M100Tokenizer

    Returns:
        Tuple of (model, generation_config, data_collator)
    """
    model_name = config.get('model_name', 'facebook/m2m100_418M')
    source_lang = config.get('source_lang', 'en')
    target_lang = config.get('target_lang', 'ka')

    print(f"Loading M2M100 model: {model_name}")
    print(f"Translation direction: {source_lang} → {target_lang}")

    # Load model
    model = M2M100ForConditionalGeneration.from_pretrained(model_name)

    # Set source language in tokenizer
    if isinstance(tokenizer, M2M100Tokenizer):
        tokenizer.src_lang = source_lang

    # Create generation config
    generation_config = GenerationConfig.from_model_config(model.config)

    # Set forced_bos_token_id for target language
    if isinstance(tokenizer, M2M100Tokenizer):
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

    print(f"Created M2M100 model:")
    print(f"  Model: {model_name}")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    return model, generation_config, data_collator 