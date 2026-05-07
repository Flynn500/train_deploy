"""Python port of ``bpe_tokenizer.rs``.

Mirrors :class:`BpeTokenizer` from the Rust crate, including the
whitespace pre-tokenizer, the priority-ordered merge table, and the
tokenize algorithm. The encoding output is *bit-identical* to the
Rust side for the same inputs — that's what makes the trained model
deployable.

Two consumers:

- :class:`vectorscry_train.model.BpeTextEncoder` (added in a later
  chunk) holds a tokenizer and uses it to turn strings into token-ID
  tensors at training time.
- :func:`vectorscry_train.bpe_format.write_bpe` writes a tokenizer's
  vocab + merges to ``.bpe`` bytes, which then ride inside the
  ``.vse`` file as the ``tokenizer_payload``.

The class accepts a HuggingFace ``tokenizers`` instance (or any
source providing a vocab list and a merge list) — see the
``from_hf_tokenizer`` classmethod for the typical training-side
construction path.
"""

from __future__ import annotations

from dataclasses import dataclass

from .bpe_format import Merge, PRE_TOKENIZER_KIND_WHITESPACE, write_bpe


# Mirror of ``NUM_BASE_BYTE_TOKENS`` from ``bpe_tokenizer.rs``: the
# first 256 token IDs are reserved for raw byte values. Promoting
# this to a runtime knob would require coordinated changes across
# Rust, the .bpe format, and this file.
NUM_BASE_BYTE_TOKENS = 256


def whitespace_pre_tokenize(text: str) -> list[bytes]:
    """Split ``text`` into chunks per ``WhitespacePreTokenizer::split``.

    Each chunk has the shape ``[whitespace*][non_whitespace+]`` —
    leading whitespace clings to the following word. The final chunk
    may be whitespace-only if the input ends with trailing whitespace
    (e.g. ``"hello "`` → ``[b"hello", b" "]``). Empty input yields an
    empty list.

    "Whitespace" here means ASCII whitespace specifically — same as
    Rust's ``u8::is_ascii_whitespace``: tab, line feed, vertical tab,
    form feed, carriage return, space (0x09, 0x0A, 0x0B, 0x0C, 0x0D,
    0x20). Unicode whitespace bytes inside a UTF-8 sequence won't be
    matched, which is intentional — we operate on the byte stream,
    not on Unicode characters.
    """
    bytes_ = text.encode("utf-8")
    if not bytes_:
        return []

    chunks: list[bytes] = []
    chunk_start = 0
    i = 0
    n = len(bytes_)
    while i < n:
        # Leading whitespace run.
        while i < n and _is_ascii_whitespace(bytes_[i]):
            i += 1
        # Word body. May be empty if we're at trailing whitespace at
        # end-of-input — in which case the chunk we're about to push
        # is whitespace-only, matching the Rust behaviour.
        while i < n and not _is_ascii_whitespace(bytes_[i]):
            i += 1
        chunks.append(bytes_[chunk_start:i])
        chunk_start = i

    return chunks


# ASCII whitespace bytes, matching Rust's u8::is_ascii_whitespace.
_ASCII_WHITESPACE = frozenset((0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x20))


def _is_ascii_whitespace(b: int) -> bool:
    return b in _ASCII_WHITESPACE


class BpeTokenizer:
    """Byte-level BPE tokenizer, Python port of the Rust ``BpeTokenizer``.

    Holds a vocab table (token ID → byte sequence) and a priority-
    ordered merge table. Tokenization runs the whitespace pre-tokenizer
    and then applies merges per chunk in priority order.

    The tokenizer is hashable-but-not-mutable: vocab and merges are
    captured by reference at construction and not modified after. If
    you need a different vocab, build a new tokenizer.

    Construction validates the same invariants as ``BpeTokenizer::new``
    in Rust:

    - Vocab has at least 256 entries; entries 0..256 are the literal
      base bytes.
    - Every merge's ``left``, ``right``, and ``result`` are valid
      token IDs.
    - Every merge's ``result >= 256`` (base bytes are reserved as
      merge inputs only).
    - Every merge's result-token bytes equal the concatenation of its
      input tokens' bytes.
    - No two merges share the same ``(left, right)`` pair.

    Validation failures raise :class:`ValueError` with a message that
    names the offending index/field — same diagnostic intent as the
    Rust ``BpeTokenizerConstructionError`` variants, but as one error
    type since Python doesn't make multiple variants worth the noise.
    """

    def __init__(
        self,
        vocab: list[bytes],
        merges: list[Merge],
        pre_tokenizer_kind: int = PRE_TOKENIZER_KIND_WHITESPACE,
    ) -> None:
        if pre_tokenizer_kind != PRE_TOKENIZER_KIND_WHITESPACE:
            # Only kind defined in v1. Rust loaders reject unknown
            # kinds with PreTokenizerMismatch; we reject at
            # construction so the .vse round-trip can't smuggle
            # something through.
            raise ValueError(
                f"unsupported pre_tokenizer_kind {pre_tokenizer_kind:#x}; "
                f"only {PRE_TOKENIZER_KIND_WHITESPACE:#x} (Whitespace) "
                f"is defined in v1"
            )
        if len(vocab) < NUM_BASE_BYTE_TOKENS:
            raise ValueError(
                f"vocab too small: needs at least {NUM_BASE_BYTE_TOKENS} "
                f"base byte tokens, got {len(vocab)}"
            )
        # Base-byte invariant: vocab[b] == bytes([b]) for b in 0..256.
        for b in range(NUM_BASE_BYTE_TOKENS):
            if vocab[b] != bytes([b]):
                raise ValueError(
                    f"base byte token {b} must have byte sequence "
                    f"[{b}], got {vocab[b]!r}"
                )

        vocab_size = len(vocab)
        seen_pairs: dict[tuple[int, int], int] = {}
        for idx, m in enumerate(merges):
            if not 0 <= m.left < vocab_size:
                raise ValueError(
                    f"merge {idx} left token {m.left} out of range "
                    f"(vocab_size = {vocab_size})"
                )
            if not 0 <= m.right < vocab_size:
                raise ValueError(
                    f"merge {idx} right token {m.right} out of range "
                    f"(vocab_size = {vocab_size})"
                )
            if not 0 <= m.result < vocab_size:
                raise ValueError(
                    f"merge {idx} result token {m.result} out of range "
                    f"(vocab_size = {vocab_size})"
                )
            if m.result < NUM_BASE_BYTE_TOKENS:
                raise ValueError(
                    f"merge {idx} result token {m.result} is a base byte "
                    f"token (must be >= {NUM_BASE_BYTE_TOKENS})"
                )
            # Concatenation invariant.
            expected = vocab[m.left] + vocab[m.right]
            if vocab[m.result] != expected:
                raise ValueError(
                    f"merge {idx} result bytes {vocab[m.result]!r} don't "
                    f"match concatenation of inputs "
                    f"({vocab[m.left]!r} + {vocab[m.right]!r} = {expected!r})"
                )
            # Duplicate check.
            pair = (m.left, m.right)
            if pair in seen_pairs:
                raise ValueError(
                    f"merge {idx} duplicates earlier merge {seen_pairs[pair]} "
                    f"(same (left={m.left}, right={m.right}) pair)"
                )
            seen_pairs[pair] = idx

        # Store after validation.
        self._vocab: list[bytes] = list(vocab)
        self._merges: list[Merge] = list(merges)
        self._pre_tokenizer_kind = pre_tokenizer_kind

        # Pair-lookup index, mirror of the Rust ``merge_lookup`` BTreeMap.
        # Maps (left, right) to (priority, result). Lower priority wins.
        # Python dicts give O(1) lookup which is at least as good as
        # the Rust BTreeMap's O(log m); we don't need the ordered
        # property here.
        self._merge_lookup: dict[tuple[int, int], tuple[int, int]] = {
            (m.left, m.right): (idx, m.result)
            for idx, m in enumerate(self._merges)
        }

    # ----- Accessors -----

    @property
    def vocab(self) -> list[bytes]:
        """The vocab table. Index ``i`` is the bytes for token ID ``i``."""
        # Returning the live list — caller shouldn't mutate, but we
        # don't enforce that. Same contract as Rust's ``vocab()``
        # which returns a borrow.
        return self._vocab

    @property
    def merges(self) -> list[Merge]:
        """Priority-ordered merges. Index 0 is the highest priority."""
        return self._merges

    @property
    def pre_tokenizer_kind(self) -> int:
        return self._pre_tokenizer_kind

    def vocab_size(self) -> int:
        return len(self._vocab)

    # ----- Encoding -----

    def tokenize(self, text: str) -> list[int]:
        """Encode ``text`` to a list of token IDs.

        Pre-tokenizes via :func:`whitespace_pre_tokenize`, then applies
        BPE merges per chunk in priority order. Concatenates the
        per-chunk results.

        Determinism: same input → same output, always. The merge
        application is left-to-right and non-overlapping, with ties
        broken by priority (lower index wins).
        """
        out: list[int] = []
        for chunk in whitespace_pre_tokenize(text):
            self._encode_chunk(chunk, out)
        return out

    def _encode_chunk(self, chunk: bytes, out: list[int]) -> None:
        """Encode one pre-tokenized chunk, appending IDs to ``out``.

        Direct port of ``BpeTokenizer::encode_chunk`` from Rust:

        1. Lay the chunk out as base-byte token IDs.
        2. Find the highest-priority merge applicable to any adjacent
           pair (lowest priority *value*).
        3. Apply that merge to every non-overlapping occurrence,
           left-to-right.
        4. Repeat until no merge applies.

        Complexity is O(k² log m) per chunk; fine for the typical
        post-whitespace chunk sizes (well under 100 bytes).
        """
        if not chunk:
            return

        # Base-byte materialization. ``b`` for ``b in chunk`` already
        # iterates as int values — the byte's own token ID.
        seq: list[int] = list(chunk)

        while True:
            # Scan adjacent pairs for the best merge. ``best`` holds
            # ``(priority, result, pair)`` of the winner so far.
            best: tuple[int, int, tuple[int, int]] | None = None
            for i in range(len(seq) - 1):
                pair = (seq[i], seq[i + 1])
                record = self._merge_lookup.get(pair)
                if record is None:
                    continue
                priority, result = record
                if best is None or priority < best[0]:
                    best = (priority, result, pair)

            if best is None:
                break

            _priority, result, target_pair = best

            # Non-overlapping left-to-right replacement. Build a fresh
            # list — same big-O, and avoids the bookkeeping of in-place
            # mutation. Mirrors the Rust ``next`` Vec pattern.
            next_seq: list[int] = []
            i = 0
            n = len(seq)
            while i < n:
                if i + 1 < n and (seq[i], seq[i + 1]) == target_pair:
                    next_seq.append(result)
                    i += 2
                else:
                    next_seq.append(seq[i])
                    i += 1
            seq = next_seq

        out.extend(seq)

    # ----- Serialization -----

    def to_bytes(self) -> bytes:
        """Serialize to .bpe v1 bytes via :func:`write_bpe`.

        Round-trips through :meth:`from_bytes` (and through the Rust
        ``BpeTokenizer::from_bytes`` for the same ``P``).
        """
        return write_bpe(
            vocab=self._vocab,
            merges=self._merges,
            pre_tokenizer_kind=self._pre_tokenizer_kind,
        )