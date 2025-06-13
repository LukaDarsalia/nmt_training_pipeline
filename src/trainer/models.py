"""Model creation functions registered for the training pipeline."""

from typing import Any, Dict, Tuple
from pathlib import Path

from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    EncoderDecoderModel,
    MarianConfig,
    MarianModel,
)

from .registry import register_model


@register_model("auto", "Load a standard seq2seq model via Auto classes")
def load_auto_model(config: Dict[str, Any]) -> Tuple[AutoModelForSeq2SeqLM, AutoTokenizer]:
    """Load a Seq2Seq model and tokenizer based on a single model name."""
    model_name = config.get("name")
    pretrained = config.get("pretrained", True)
    if pretrained:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    else:
        model_cfg = AutoConfig.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_config(model_cfg)
    tokenizer_name = config.get("tokenizer_name", model_name)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    return model, tokenizer


@register_model("encoder_decoder", "Combine separate encoder and decoder models")
def load_encoder_decoder_model(config: Dict[str, Any]) -> Tuple[EncoderDecoderModel, AutoTokenizer]:
    """Create an EncoderDecoderModel from separate encoder and decoder."""
    encoder_name = config.get("encoder_name")
    decoder_name = config.get("decoder_name")
    pretrained = config.get("pretrained", True)

    if pretrained:
        encoder = AutoModel.from_pretrained(encoder_name)
        decoder = AutoModelForCausalLM.from_pretrained(decoder_name)
    else:
        encoder_cfg = AutoConfig.from_pretrained(encoder_name)
        decoder_cfg = AutoConfig.from_pretrained(decoder_name)
        encoder = AutoModel.from_config(encoder_cfg)
        decoder = AutoModelForCausalLM.from_config(decoder_cfg)

    model = EncoderDecoderModel(encoder=encoder, decoder=decoder)

    tokenizer_name = config.get("tokenizer_name", encoder_name)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    if model.config.decoder_start_token_id is None:
        model.config.decoder_start_token_id = tokenizer.cls_token_id
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    return model, tokenizer


class DualTokenizer:
    """Wrapper combining separate encoder and decoder tokenizers."""

    def __init__(self, encoder_tokenizer: Any, decoder_tokenizer: Any) -> None:
        self.encoder_tokenizer = encoder_tokenizer
        self.decoder_tokenizer = decoder_tokenizer

    def __getattr__(self, item: str) -> Any:
        return getattr(self.encoder_tokenizer, item)

    def as_target_tokenizer(self) -> Any:
        return self.decoder_tokenizer

    def batch_decode(self, *args: Any, **kwargs: Any) -> Any:
        return self.decoder_tokenizer.batch_decode(*args, **kwargs)

    def save_pretrained(self, save_directory: str) -> None:
        self.encoder_tokenizer.save_pretrained(Path(save_directory) / "encoder")
        self.decoder_tokenizer.save_pretrained(Path(save_directory) / "decoder")


@register_model(
    "marian_custom",
    "Marian encoder-decoder from scratch with separate tokenizers",
)
def load_marian_custom_model(config: Dict[str, Any]) -> Tuple[EncoderDecoderModel, DualTokenizer]:
    """Create a Marian encoder-decoder model initialized from scratch."""

    encoder_name = config.get("encoder_name", "Helsinki-NLP/opus-mt-en-de")
    decoder_name = config.get("decoder_name", "Helsinki-NLP/opus-mt-en-de")
    enc_tokenizer_name = config.get("encoder_tokenizer")
    dec_tokenizer_name = config.get("decoder_tokenizer")

    enc_tokenizer = AutoTokenizer.from_pretrained(enc_tokenizer_name)
    dec_tokenizer = AutoTokenizer.from_pretrained(dec_tokenizer_name)
    tokenizer = DualTokenizer(enc_tokenizer, dec_tokenizer)

    encoder_cfg = MarianConfig.from_pretrained(encoder_name)
    decoder_cfg = MarianConfig.from_pretrained(decoder_name)
    decoder_cfg.is_decoder = True
    decoder_cfg.add_cross_attention = True

    encoder = MarianModel(encoder_cfg)
    decoder = MarianModel(decoder_cfg)

    model = EncoderDecoderModel(encoder=encoder, decoder=decoder)

    if model.config.decoder_start_token_id is None:
        model.config.decoder_start_token_id = dec_tokenizer.cls_token_id
    if model.config.pad_token_id is None:
        model.config.pad_token_id = dec_tokenizer.pad_token_id

    return model, tokenizer


@register_model(
    "e5_m2m",
    "Encoder-decoder using multilingual-e5-large weights",
)
def load_e5_encoder_decoder(config: Dict[str, Any]) -> Tuple[EncoderDecoderModel, AutoTokenizer]:
    """Create an encoder-decoder model using multilingual-e5-large for both sides."""

    model_name = "intfloat/multilingual-e5-large"
    tokenizer_name = config.get("tokenizer_name", model_name)

    encoder = AutoModel.from_pretrained(model_name)
    decoder_cfg = AutoConfig.from_pretrained(model_name)
    decoder_cfg.is_decoder = True
    decoder_cfg.add_cross_attention = True
    decoder = AutoModelForCausalLM.from_config(decoder_cfg)

    model = EncoderDecoderModel(encoder=encoder, decoder=decoder)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    if model.config.decoder_start_token_id is None:
        model.config.decoder_start_token_id = tokenizer.cls_token_id
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    return model, tokenizer
