"""vectorscry-train: PyTorch training pipeline for vectorscry-embedding.

Public API:

- ``model``: PyTorch port of the inference architecture
  (:class:`ByteTextEncoder`, :class:`BpeTextEncoder`,
  :class:`MlpProjector`, :class:`ComposedEmbedder`,
  :func:`tokenize_batch`, :func:`bpe_tokenize_batch`).
- ``bpe_tokenizer``: Python port of the byte-level BPE tokenizer
  (:class:`BpeTokenizer`, :func:`whitespace_pre_tokenize`).
- ``bpe_format``: writer for the .bpe v1 tokenizer format
  (:func:`write_bpe`, :class:`Merge`).
- ``vse``: writer for the .vse v1 weight format
  (:func:`save_composed`, :func:`write_vse`). Supports both
  :class:`ByteTextEncoder` and :class:`BpeTextEncoder` models;
  ``save_composed`` bridges a trained :class:`ComposedEmbedder` to
  disk including the embedded tokenizer for BPE models.
- ``train``: distillation loop (:func:`distill`, :class:`StudentArch`,
  :class:`RunOptions`, :class:`RunResult`). Pass ``tokenizer`` to
  :class:`StudentArch` to train a BPE student instead of byte.
"""

from .model import (
    AttentionPoolMlpProjector,
    BYTE_VOCAB_SIZE,
    BpeTextEncoder,
    ByteTextEncoder,
    ComposedEmbedder,
    ConvBlock,
    KERNEL_SIZE,
    MlpProjector,
    NUM_BLOCKS,
    bpe_tokenize_batch,
    tokenize_batch,
)
from .augment import (
    Augmenter,
    CharAugmenter,
    build_paired_corpus,
)
from .bpe_format import (
    Merge,
    PRE_TOKENIZER_KIND_WHITESPACE,
    write_bpe,
)
from .bpe_tokenizer import (
    BpeTokenizer,
    NUM_BASE_BYTE_TOKENS,
    whitespace_pre_tokenize,
)
from .bpe_train import train_bpe
from .train import (
    RunOptions,
    RunResult,
    StudentArch,
    distill,
)
from .vse import (
    AttentionPoolMlpProjectorSpec,
    save_composed,
    write_vse,
)

__all__ = [
    "AttentionPoolMlpProjector",
    "AttentionPoolMlpProjectorSpec",
    "Augmenter",
    "BYTE_VOCAB_SIZE",
    "BpeTextEncoder",
    "BpeTokenizer",
    "ByteTextEncoder",
    "CharAugmenter",
    "ComposedEmbedder",
    "ConvBlock",
    "KERNEL_SIZE",
    "Merge",
    "MlpProjector",
    "NUM_BASE_BYTE_TOKENS",
    "NUM_BLOCKS",
    "PRE_TOKENIZER_KIND_WHITESPACE",
    "RunOptions",
    "RunResult",
    "StudentArch",
    "bpe_tokenize_batch",
    "train_bpe",
    "build_paired_corpus",
    "distill",
    "save_composed",
    "tokenize_batch",
    "whitespace_pre_tokenize",
    "write_bpe",
    "write_vse",
]