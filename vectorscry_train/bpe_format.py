"""Writer for the .bpe v1 tokenizer format ("VSBP").

Mirrors ``BpeTokenizer::to_bytes`` from ``bpe_tokenizer.rs``. Pure
stdlib — no numpy, no torch — so this module is cheap to import and
testable without the heavy deps.

The .bpe payload is what gets stuffed into a .vse file's
``tokenizer_payload`` field when the encoder kind is ``BpeText`` and
``tokenizer_kind`` is ``EmbeddedBpeV1`` (0x01). It can also be written
to its own file for inspection / round-trip testing against the Rust
reader.

Spec, copied from ``bpe_tokenizer.rs::from_bytes`` for reference:

    magic                "VSBP"  (4 bytes)
    format_version       u16     (= 1)
    pre_tokenizer_kind   u8      (= P::KIND, currently 0x01 for
                                   WhitespacePreTokenizer)
    reserved             u8      (= 0x00)
    vocab_size           u32 LE
    num_merges           u32 LE

    vocab table:
      for token_id in 0..vocab_size:
        byte_len   u16 LE
        bytes      [u8; byte_len]

    merge table:
      for merge_index in 0..num_merges:
        left       u32 LE
        right      u32 LE
        result     u32 LE

All multi-byte fields are little-endian. No padding anywhere — the
file is byte-aligned, so the data block does not have the 4-byte
alignment rule that .vse imposes.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass


# Magic bytes from the spec: ASCII "VSBP".
BPE_MAGIC = b"VSBP"
BPE_FORMAT_VERSION = 1

# Pre-tokenizer kind tags. The Rust side uses ``P::KIND`` as a
# trait-associated constant; we mirror the only current value.
# ``WhitespacePreTokenizer::KIND == 0x01`` (confirmed by tests in
# ``bpe_tokenizer.rs`` that corrupt this byte to 0xFE and expect
# ``PreTokenizerMismatch { expected_kind: 0x01 }``).
PRE_TOKENIZER_KIND_WHITESPACE = 0x01

# Token byte-length is u16-prefixed in the file format. Real BPE tokens
# are well under this limit (typical max < 30 bytes), but we surface
# the bound here so the writer's contract is honest.
MAX_TOKEN_BYTE_LEN = 0xFFFF


@dataclass(frozen=True)
class Merge:
    """A single BPE merge: ``(left, right) -> result`` token IDs.

    Direct mirror of ``bpe_tokenizer::Merge``. ``left`` and ``right``
    must be valid token IDs in the vocab; ``result`` must be a valid
    non-base-byte token ID (>= 256) whose vocab bytes equal
    ``vocab[left] ++ vocab[right]``. The :class:`BpeTokenizer`
    constructor enforces these invariants — the writer assumes its
    inputs were already validated.
    """
    left: int
    right: int
    result: int


def write_bpe(
    vocab: list[bytes],
    merges: list[Merge],
    pre_tokenizer_kind: int = PRE_TOKENIZER_KIND_WHITESPACE,
) -> bytes:
    """Serialize a tokenizer to .bpe v1 bytes.

    The caller is responsible for vocabulary validation (base-byte
    invariant, merge consistency, etc.) — this writer does only what
    the Rust ``to_bytes`` does, namely the per-field length and range
    checks the wire format itself imposes. Use :class:`BpeTokenizer`
    if you want the full structural validation.

    Args:
        vocab: per-token byte sequences in token-ID order. Index ``i``
            is the bytes for token ID ``i``. Must have length >= 256
            (the base-byte tokens fill IDs 0..255), but this writer
            does not enforce that — see the note above.
        merges: priority-ordered merges. Index 0 is highest priority.
        pre_tokenizer_kind: the ``P::KIND`` byte. Default is
            ``WhitespacePreTokenizer`` (0x01), which is the only kind
            v1 supports.

    Returns:
        The .bpe file bytes. Round-trips through
        ``BpeTokenizer::from_bytes`` on the Rust side.

    Raises:
        ValueError: if a vocab token exceeds ``u16`` length, if
            ``pre_tokenizer_kind`` is out of u8 range, or if
            ``vocab_size`` / ``num_merges`` overflow u32.
    """
    if not 0 <= pre_tokenizer_kind <= 0xFF:
        raise ValueError(
            f"pre_tokenizer_kind must fit in u8 (got {pre_tokenizer_kind})"
        )
    if len(vocab) > 0xFFFF_FFFF:
        raise ValueError(f"vocab_size {len(vocab)} exceeds u32")
    if len(merges) > 0xFFFF_FFFF:
        raise ValueError(f"num_merges {len(merges)} exceeds u32")

    parts: list[bytes] = []

    # Header: 16 bytes total.
    parts.append(BPE_MAGIC)
    parts.append(struct.pack("<H", BPE_FORMAT_VERSION))
    parts.append(struct.pack("<B", pre_tokenizer_kind))
    parts.append(b"\x00")  # reserved
    parts.append(struct.pack("<I", len(vocab)))
    parts.append(struct.pack("<I", len(merges)))

    # Vocab table.
    for token_id, token_bytes in enumerate(vocab):
        if len(token_bytes) > MAX_TOKEN_BYTE_LEN:
            raise ValueError(
                f"vocab token {token_id} is {len(token_bytes)} bytes long; "
                f"exceeds .bpe format u16 length prefix limit "
                f"({MAX_TOKEN_BYTE_LEN})"
            )
        parts.append(struct.pack("<H", len(token_bytes)))
        parts.append(token_bytes)

    # Merge table.
    for m in merges:
        if not 0 <= m.left <= 0xFFFF_FFFF:
            raise ValueError(f"merge left token id {m.left} out of u32 range")
        if not 0 <= m.right <= 0xFFFF_FFFF:
            raise ValueError(f"merge right token id {m.right} out of u32 range")
        if not 0 <= m.result <= 0xFFFF_FFFF:
            raise ValueError(f"merge result token id {m.result} out of u32 range")
        parts.append(struct.pack("<III", m.left, m.right, m.result))

    return b"".join(parts)