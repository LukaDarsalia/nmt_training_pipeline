"""
Encoder-Decoder Tokenizer Implementations

Provides tokenizer implementations for encoder-decoder models.
"""
import os
import numpy as np
import torch
import tensorflow as tf
from pathlib import Path
from overrides import overrides
from typing import Dict, Any, Tuple, Union, List, Optional, overload
from datasets import Dataset, DatasetDict
from transformers.tokenization_utils_base import (
    AddedToken, # type: ignore
    BatchEncoding,
    EncodedInput,
    EncodedInputPair,
    PreTokenizedInput,
    PreTokenizedInputPair,
    TextInput,
    TextInputPair,
    TruncationStrategy,
)
from transformers.utils import logging
from transformers import AutoTokenizer
from transformers.utils.generic import PaddingStrategy, TensorType
from transformers.tokenization_utils import PreTrainedTokenizer
from ..registry.tokenizer_registry import BaseTokenizer, register_tokenizer


logger = logging.get_logger(__name__)

SPIECE_UNDERLINE = "▁"


class EncoderDecoderTokenizer(PreTrainedTokenizer):
    def __init__(self, encoder_tokenizer_path, decoder_tokenizer_path, **kwargs):
        self.encoder: PreTrainedTokenizer = AutoTokenizer.from_pretrained(encoder_tokenizer_path)
        self.decoder: PreTrainedTokenizer = AutoTokenizer.from_pretrained(decoder_tokenizer_path)
        self.current_tokenizer = self.encoder
        self._decode_use_source_tokenizer = False

        if self.decoder.eos_token is None:
            self.decoder.eos_token = self.decoder.sep_token
        if self.decoder.eos_token is None:
            self.decoder.eos_token = self.decoder.sep_token

        if self.encoder.pad_token is None:
            self.encoder.pad_token = self.encoder.eos_token
        if self.decoder.pad_token is None:
            self.decoder.pad_token = self.decoder.eos_token

        if self.encoder.bos_token is None:
            self.encoder.bos_token = self.encoder.cls_token
        if self.decoder.bos_token is None:
            self.decoder.bos_token = self.decoder.cls_token

        self._pad_token = self.encoder.pad_token
        self._unk_token = self.encoder.unk_token
        self._bos_token = self.encoder.bos_token
        self._eos_token = self.encoder.eos_token
        self._sep_token = self.encoder.sep_token
        self._cls_token = self.encoder.cls_token
        self._mask_token = self.encoder.mask_token
        self.decoder_pad_token = self.decoder.pad_token
        self.decoder_unk_token = self.decoder.unk_token
        self.decoder_bos_token = self.decoder.bos_token
        self.decoder_eos_token = self.decoder.eos_token
        self.decoder_sep_token = self.decoder.sep_token
        self.decoder_cls_token = self.decoder.cls_token
        self.decoder_mas_token = self.decoder.mask_token

        self.decoder_pad_token_id = self.decoder.pad_token_id
        self.decoder_unk_token_id = self.decoder.unk_token_id
        self.decoder_bos_token_id = self.decoder.bos_token_id
        self.decoder_eos_token_id = self.decoder.eos_token_id
        self.decoder_sep_token_id = self.decoder.sep_token_id
        self.decoder_cls_token_id = self.decoder.cls_token_id
        self.decoder_mas_token_id = self.decoder.mask_token_id
        self._additional_special_tokens = []

    @property
    def is_fast(self) -> bool:
        return self.current_tokenizer.is_fast

    @property
    def vocab_size(self) -> int:
        """
        `int`: Size of the base vocabulary (without the added tokens).
        """
        return self.current_tokenizer.vocab_size

    @property
    def added_tokens_encoder(self) -> Dict[str, int]:
        """
        Returns the sorted mapping from string to index. The added tokens encoder is cached for performance
        optimisation in `self._added_tokens_encoder` for the slow tokenizers.
        """
        return self.current_tokenizer.added_tokens_encoder

    @property
    def added_tokens_decoder(self) -> Dict[int, AddedToken]:
        """
        Returns the added tokens in the vocabulary as a dictionary of index to AddedToken.

        Returns:
            `Dict[str, int]`: The added tokens.
        """
        return self.current_tokenizer.added_tokens_decoder

    @added_tokens_decoder.setter
    def added_tokens_decoder(self, value: Dict[int, Union[AddedToken, str]]) -> None:
        self.current_tokenizer.added_tokens_decoder = value

    def get_added_vocab(self) -> Dict[str, int]:
        """
        Returns the added tokens in the vocabulary as a dictionary of token to index. Results might be different from
        the fast call because for now we always add the tokens even if they are already in the vocabulary. This is
        something we should change.

        Returns:
            `Dict[str, int]`: The added tokens.
        """
        return self._added_tokens_encoder

    def __len__(self):
        """
        Size of the full vocabulary with the added tokens. Counts the `keys` and not the `values` because otherwise if
        there is a hole in the vocab, we will add tokenizers at a wrong index.
        """
        return len(set(self.get_vocab().keys()))

    def num_special_tokens_to_add(self, pair: bool = False) -> int:
        """
        Returns the number of added tokens when encoding a sequence with special tokens.

        <Tip>

        This encodes a dummy input and checks the number of added tokens, and is therefore not efficient. Do not put
        this inside your training loop.

        </Tip>

        Args:
            pair (`bool`, *optional*, defaults to `False`):
                Whether the number of added tokens should be computed in the case of a sequence pair or a single
                sequence.

        Returns:
            `int`: Number of special tokens added to sequences.
        """
        return self.current_tokenizer.num_special_tokens_to_add(pair)

    def tokenize(self, text: TextInput, **kwargs):
        """
        Converts a string in a sequence of tokens, using the tokenizer.

        Split in words for word-based vocabulary or sub-words for sub-word-based vocabularies
        (BPE/SentencePieces/WordPieces). Takes care of added tokens.

        Args:
            text (`str`):
                The sequence to be encoded.
            **kwargs (additional keyword arguments):
                Passed along to the model-specific `prepare_for_tokenization` preprocessing method.

        Returns:
            `List[str]`: The list of tokens.
        """
        return self.decoder.tokenize(text, **kwargs)

    def _tokenize(self, text, **kwargs):
        """
        Converts a string in a sequence of tokens (string), using the tokenizer. Split in words for word-based
        vocabulary or sub-words for sub-word-based vocabularies (BPE/SentencePieces/WordPieces).

        Do NOT take care of added tokens.
        """
        raise self.decoder._tokenize(text, **kwargs)

    def convert_tokens_to_ids(self, tokens: Union[str, List[str]]) -> Union[int, List[int]]:
        """
        Converts a token string (or a sequence of tokens) in a single integer id (or a sequence of ids), using the
        vocabulary.

        Args:
            tokens (`str` or `List[str]`): One or several token(s) to convert to token id(s).

        Returns:
            `int` or `List[int]`: The token id or list of token ids.
        """
        return self.current_tokenizer.convert_tokens_to_ids(tokens)

    def _convert_token_to_id_with_added_voc(self, token):
        return self.current_tokenizer._convert_token_to_id_with_added_voc(token)

    def _convert_token_to_id(self, token):
        return self.current_tokenizer._convert_token_to_id(token)

    def encode(self, *args, **kwargs):
        return self.current_tokenizer.encode(*args, **kwargs)

    def _batch_encode_plus(
            self,
            batch_text_or_text_pairs: Union[
                List[TextInput],
                List[TextInputPair],
                List[PreTokenizedInput],
                List[PreTokenizedInputPair],
                List[EncodedInput],
                List[EncodedInputPair],
            ],
            add_special_tokens: bool = True,
            padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
            truncation_strategy: TruncationStrategy = TruncationStrategy.DO_NOT_TRUNCATE,
            max_length: Optional[int] = None,
            stride: int = 0,
            is_split_into_words: bool = False,
            pad_to_multiple_of: Optional[int] = None,
            return_tensors: Optional[Union[str, TensorType]] = None,
            return_token_type_ids: Optional[bool] = None,
            return_attention_mask: Optional[bool] = None,
            return_overflowing_tokens: bool = False,
            return_special_tokens_mask: bool = False,
            return_offsets_mapping: bool = False,
            return_length: bool = False,
            verbose: bool = True,
            **kwargs,
    ) -> BatchEncoding:
        return self.current_tokenizer._batch_encode_plus(batch_text_or_text_pairs=batch_text_or_text_pairs,
                                                         add_special_tokens=add_special_tokens,
                                                         padding_strategy=padding_strategy,
                                                         truncation_strategy=truncation_strategy,
                                                         max_length=max_length,
                                                         stride=stride,
                                                         is_split_into_words=is_split_into_words,
                                                         pad_to_multiple_of=pad_to_multiple_of,
                                                         return_tensors=return_tensors,
                                                         return_token_type_ids=return_token_type_ids,
                                                         return_attention_mask=return_attention_mask,
                                                         return_overflowing_tokens=return_overflowing_tokens,
                                                         return_special_tokens_mask=return_special_tokens_mask,
                                                         return_offsets_mapping=return_offsets_mapping,
                                                         return_length=return_length,
                                                         verbose=verbose,
                                                         **kwargs,
                                                         )

    def prepare_for_tokenization(
            self, text: str, is_split_into_words: bool = False, **kwargs
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Performs any necessary transformations before tokenization.

        This method should pop the arguments from kwargs and return the remaining `kwargs` as well. We test the
        `kwargs` at the end of the encoding process to be sure all the arguments have been used.

        Args:
            text (`str`):
                The text to prepare.
            is_split_into_words (`bool`, *optional*, defaults to `False`):
                Whether or not the input is already pre-tokenized (e.g., split into words). If set to `True`, the
                tokenizer assumes the input is already split into words (for instance, by splitting it on whitespace)
                which it will tokenize. This is useful for NER or token classification.
            kwargs (`Dict[str, Any]`, *optional*):
                Keyword arguments to use for the tokenization.

        Returns:
            `Tuple[str, Dict[str, Any]]`: The prepared text and the unused kwargs.
        """
        return self.current_tokenizer.prepare_for_tokenization(text, is_split_into_words, **kwargs)

    def get_special_tokens_mask(
            self, token_ids_0: List, token_ids_1: Optional[List] = None, already_has_special_tokens: bool = False
    ) -> List[int]:
        """
        Retrieves sequence ids from a token list that has no special tokens added. This method is called when adding
        special tokens using the tokenizer `prepare_for_model` or `encode_plus` methods.

        Args:
            token_ids_0 (`List[int]`):
                List of ids of the first sequence.
            token_ids_1 (`List[int]`, *optional*):
                List of ids of the second sequence.
            already_has_special_tokens (`bool`, *optional*, defaults to `False`):
                Whether or not the token list is already formatted with special tokens for the model.

        Returns:
            A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.
        """

        return self.current_tokenizer.get_special_tokens_mask(token_ids_0, token_ids_1, already_has_special_tokens)

    @overload
    def convert_ids_to_tokens(self, ids: int, skip_special_tokens: bool = False) -> str:
        return self.current_tokenizer.convert_ids_to_tokens(ids, skip_special_tokens)

    @overload
    def convert_ids_to_tokens(self, ids: List[int], skip_special_tokens: bool = False) -> List[str]:
        return self.current_tokenizer.convert_ids_to_tokens(ids, skip_special_tokens)

    def convert_ids_to_tokens(
            self, ids: Union[int, List[int]], skip_special_tokens: bool = False
    ) -> Union[str, List[str]]:
        """
        Converts a single index or a sequence of indices in a token or a sequence of tokens, using the vocabulary and
        added tokens.

        Args:
            ids (`int` or `List[int]`):
                The token id (or token ids) to convert to tokens.
            skip_special_tokens (`bool`, *optional*, defaults to `False`):
                Whether or not to remove special tokens in the decoding.

        Returns:
            `str` or `List[str]`: The decoded token(s).
        """
        return self.current_tokenizer.convert_ids_to_tokens(ids, skip_special_tokens)

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        return self.current_tokenizer.convert_tokens_to_string(tokens)

    def decode(
        self,
        token_ids: Union[int, List[int], "np.ndarray", "torch.Tensor", "tf.Tensor"],
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: Optional[bool] = None,
        **kwargs,
    ) -> str:
        return self.decoder.decode(token_ids, skip_special_tokens, clean_up_tokenization_spaces, **kwargs)

    @overrides
    def __call__(self, text, text_target=None, *args, **kwargs):
        results = self.encoder(text, *args, **kwargs)
        if text_target:
            results['labels'] = self.decoder(text_target, *args, **kwargs)['input_ids']
        return results

    def _decode(
            self,
            token_ids: List[int],
            skip_special_tokens: bool = False,
            clean_up_tokenization_spaces: Optional[bool] = None,
            spaces_between_special_tokens: bool = True,
            **kwargs,
    ) -> str:
        return self.decoder._decode(token_ids,
                                    skip_special_tokens,
                                    clean_up_tokenization_spaces,
                                    spaces_between_special_tokens)

    def save_pretrained(
        self,
        save_directory: Union[str, os.PathLike],
        legacy_format: Optional[bool] = None,
        filename_prefix: Optional[str] = None,
        push_to_hub: bool = False,
        **kwargs,
    ) -> None:
        encoder_path=Path(save_directory) / Path("encoder")
        decoder_path = Path(save_directory) / Path("decoder")
        self.encoder.save_pretrained(encoder_path, legacy_format, filename_prefix, push_to_hub, **kwargs)
        self.decoder.save_pretrained(decoder_path, legacy_format, filename_prefix, push_to_hub, **kwargs)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Union[str, os.PathLike],
        *init_inputs,
        cache_dir: Optional[Union[str, os.PathLike]] = None,
        force_download: bool = False,
        local_files_only: bool = False,
        token: Optional[Union[str, bool]] = None,
        revision: str = "main",
        **kwargs,
    ):
        encoder_path = Path(pretrained_model_name_or_path) / Path("encoder")
        decoder_path = Path(pretrained_model_name_or_path) / Path("decoder")

        return EncoderDecoderTokenizer(encoder_path, decoder_path)


    def _switch_to_target_mode(self):
        self.current_encoder = self.decoder

    def _switch_to_input_mode(self):
        self.current_tokenizer = self.encoder

    @property
    def pad_token_id(self) -> Any:
        """Return pad token ID from current tokenizer."""
        return self.current_tokenizer.pad_token_id

    @property
    def unk_token_id(self) -> Any:
        """Return unk token ID from current tokenizer."""
        return self.current_tokenizer.unk_token_id

    @property
    def bos_token_id(self) -> Any:
        """Return bos token ID from current tokenizer."""
        return self.current_tokenizer.bos_token_id

    @property
    def eos_token_id(self) -> Any:
        """Return eos token ID from current tokenizer."""
        return self.current_tokenizer.eos_token_id

    @property
    def sep_token_id(self) -> Any:
        """Return sep token ID from current tokenizer."""
        return self.current_tokenizer.sep_token_id

    @property
    def cls_token_id(self) -> Any:
        """Return cls token ID from current tokenizer."""
        return self.current_tokenizer.cls_token_id

    @property
    def mask_token_id(self) -> Any:
        """Return mask token ID from current tokenizer."""
        return self.current_tokenizer.mask_token_id

    def get_vocab(self) -> Dict[str, int]:
        """
        Returns the vocabulary as a dictionary of token to indices.
        """
        return self.current_tokenizer.get_vocab()

    @property
    def pad_token(self) -> Any:
        """Return pad token from current tokenizer."""
        return self.current_tokenizer.pad_token

    @property
    def unk_token(self) -> Any:
        """Return unk token from current tokenizer."""
        return self.current_tokenizer.unk_token

    @property
    def bos_token(self) -> Any:
        """Return bos token from current tokenizer."""
        return self.current_tokenizer.bos_token

    @property
    def eos_token(self) -> Any:
        """Return eos token from current tokenizer."""
        return self.current_tokenizer.eos_token

    @property
    def sep_token(self) -> Any:
        """Return sep token from current tokenizer."""
        return self.current_tokenizer.sep_token

    @property
    def cls_token(self) -> Any:
        """Return cls token from current tokenizer."""
        return self.current_tokenizer.cls_token

    @property
    def mask_token(self) -> Any:
        """Return mask token from current tokenizer."""
        return self.current_tokenizer.mask_token


@register_tokenizer("encoder_decoder", "Encoder-decoder tokenizer for encoder-decoder models")
class EncoderDecoderTokenizerImpl(BaseTokenizer):
    """Encoder-decoder tokenizer implementation."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the encoder-decoder tokenizer."""
        super().__init__(config)
        self.tokenizer: Optional[EncoderDecoderTokenizer] = None
    
    def load_tokenizer(self) -> EncoderDecoderTokenizer:
        """
        Load tokenizer for encoder-decoder model.
        
        Returns:
            Configured EncoderDecoderTokenizer
        """
        encoder_model = self.config.get('encoder_model', 'xlm-roberta-base')
        decoder_model = self.config.get('decoder_model', 'gpt2')
        
        print(f"Loading encoder-decoder tokenizer:")
        print(f"  Encoder model: {encoder_model}")
        print(f"  Decoder model: {decoder_model}")
        
        # Create the combined encoder-decoder tokenizer
        tokenizer = EncoderDecoderTokenizer(encoder_model, decoder_model)
        
        print(f"Loaded encoder-decoder tokenizer successfully")
        print(f"Encoder vocab size: {tokenizer.encoder.vocab_size}")
        print(f"Decoder vocab size: {tokenizer.decoder.vocab_size}")
        
        return tokenizer
    
    def tokenize_datasets(self, 
                         datasets: Tuple[Dataset, Dataset, Dataset],
                         tokenization_config: Dict[str, Any]) -> Tuple[Dataset, Dataset, Dataset]:
        """
        Tokenize datasets for encoder-decoder model.
        
        Args:
            datasets: Tuple of (train, valid, test) datasets
            tokenization_config: Tokenization configuration
            
        Returns:
            Tuple of tokenized datasets
        """
        if self.tokenizer is None:
            self.tokenizer = self.load_tokenizer()
        
        # Check if encoder-decoder preprocessing is enabled
        encoder_decoder_preprocessing = tokenization_config.get('encoder_decoder_preprocessing', True)
        
        if encoder_decoder_preprocessing:
            return self._encoder_decoder_tokenize_datasets(datasets, self.tokenizer, tokenization_config, encoder_decoder_preprocessing)
        else:
            return self._default_tokenize_datasets(datasets, self.tokenizer, tokenization_config)
    
    def _encoder_decoder_tokenize_datasets(self,
                                          datasets: Tuple[Dataset, Dataset, Dataset],
                                          tokenizer: EncoderDecoderTokenizer,
                                          config: Dict[str, Any],
                                          encoder_decoder_preprocessing: bool) -> Tuple[Dataset, Dataset, Dataset]:
        """
        Special tokenization for encoder-decoder models with preprocessing.
        
        Args:
            datasets: Tuple of (train, valid, test) datasets
            tokenizer: EncoderDecoderTokenizer to use
            config: Tokenization configuration
            encoder_decoder_preprocessing: Whether encoder-decoder preprocessing is enabled
            
        Returns:
            Tuple of tokenized datasets
        """
        train_dataset, valid_dataset, test_dataset = datasets
        
        # Get configuration parameters
        max_length = config.get('max_length', 128)
        source_column = config.get('source_column', 'en')
        target_column = config.get('target_column', 'ka')
        
        def preprocess_encoder_decoder(examples):
            """Preprocess function for encoder-decoder tokenization."""
            # Get source and target texts
            inputs = [str(ex) for ex in examples[source_column]]
            targets = [str(ex) for ex in examples[target_column]]
            
            # Use the custom encoder-decoder tokenizer
            model_inputs = tokenizer(
                inputs,
                text_target=targets,
                max_length=max_length,
                truncation=True,
                padding='max_length'
            )
            
            return model_inputs
        
        # Create dataset dict for easier processing
        dataset_dict = DatasetDict({
            'train': train_dataset,
            'valid': valid_dataset,
            'test': test_dataset
        })
        
        # Tokenize all datasets
        tokenized_datasets = dataset_dict.map(
            preprocess_encoder_decoder,
            batched=True,
            remove_columns=dataset_dict["train"].column_names,
            desc="Tokenizing encoder-decoder datasets"
        )
        
        print(f"Encoder-decoder tokenization complete:")
        print(f"  Max length: {max_length}")
        print(f"  Source column: {source_column}")
        print(f"  Target column: {target_column}")
        print(f"  Encoder-decoder preprocessing: {encoder_decoder_preprocessing}")
        
        return (
            tokenized_datasets['train'],
            tokenized_datasets['valid'],
            tokenized_datasets['test']
        ) 