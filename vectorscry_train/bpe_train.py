"""BPE vocabulary training, compatible with :class:`BpeTokenizer`.

A pure-Python BPE trainer that produces a vocab + merge list in the
exact shape :class:`BpeTokenizer` expects: indices 0..255 are the
literal base bytes, indices 256.. are merged tokens added in priority
order, and every merge satisfies the concatenation invariant.

Pre-tokenization mirrors :func:`whitespace_pre_tokenize` exactly — the
trainer can't use HuggingFace's ``tokenizers`` library off the shelf
because:

- HF's ``ByteLevelBPETokenizer`` first remaps every byte through a
  GPT-2-style printable-Unicode bijection before BPE runs, so the
  resulting vocab is over remapped strings, not raw bytes. Undoing
  that remap on the way out is fragile and easy to get subtly wrong.
- HF's ``Whitespace`` pre-tokenizer splits on Unicode whitespace AND
  separates punctuation into its own tokens, which doesn't match our
  ASCII-whitespace-only splitter.

Writing the trainer ourselves is ~100 lines and produces a vocab
that's by-construction compatible with our :class:`BpeTokenizer`. The
speed penalty (pure Python vs Rust-backed) is real but tolerable: a
4K vocab over the SciFact corpus (~5K docs) trains in under a minute,
and vocab training happens once per deployment.

Algorithm: classic Sennrich-style BPE. Count pair frequencies across
the corpus, pick the most frequent adjacent pair, merge it everywhere,
add to vocab, repeat until target size or no eligible pairs remain.
Frequency-weighted by pre-tokenized chunk count — identical chunks
contribute their multiplicity, not their distinct count, to pair stats.
"""

from __future__ import annotations

from collections import Counter

from .bpe_format import Merge, PRE_TOKENIZER_KIND_WHITESPACE
from .bpe_tokenizer import (
    BpeTokenizer,
    NUM_BASE_BYTE_TOKENS,
    whitespace_pre_tokenize,
)


def train_bpe(
    corpus: list[str],
    vocab_size: int,
    *,
    min_pair_frequency: int = 2,
) -> BpeTokenizer:
    """Train a BPE tokenizer on a corpus of strings.

    The resulting tokenizer is compatible with :class:`BpeTokenizer`
    construction (passes its validation) and ready to embed in a
    ``.vse`` via :func:`save_composed`.

    Args:
        corpus: training texts. Each is whitespace-pre-tokenized via
            :func:`whitespace_pre_tokenize`; pair statistics are
            collected across the resulting chunks. An empty corpus is
            allowed and produces a tokenizer with only the base byte
            vocab (no merges).
        vocab_size: target vocab size. Must be at least
            :data:`NUM_BASE_BYTE_TOKENS` (256) since the base bytes
            occupy IDs 0..255 unconditionally. The trainer adds at
            most ``vocab_size - 256`` merges; if the corpus runs out
            of pairs above ``min_pair_frequency`` first, training
            terminates early and the returned tokenizer's vocab is
            smaller than the target.
        min_pair_frequency: minimum count for a pair to be eligible
            for merging. Pairs below this threshold are ignored even
            if no other pairs exist. Default 2 — singletons are noise.

    Returns:
        A trained :class:`BpeTokenizer` with whitespace pre-tokenizer.
    """
    if vocab_size < NUM_BASE_BYTE_TOKENS:
        raise ValueError(
            f"vocab_size must be >= {NUM_BASE_BYTE_TOKENS} (got {vocab_size})"
        )
    if min_pair_frequency < 1:
        raise ValueError(
            f"min_pair_frequency must be >= 1 (got {min_pair_frequency})"
        )

    # Vocab seeded with the 256 base byte tokens, in order. The
    # BpeTokenizer constructor enforces this invariant; we satisfy it
    # by construction here.
    vocab: list[bytes] = [bytes([b]) for b in range(NUM_BASE_BYTE_TOKENS)]
    merges: list[Merge] = []

    # Learn merges into the existing vocab/merges lists, in priority
    # order. The helper mutates them in place.
    _learn_merges(
        corpus=corpus,
        vocab=vocab,
        merges=merges,
        target_vocab_size=vocab_size,
        min_pair_frequency=min_pair_frequency,
    )

    return BpeTokenizer(
        vocab=vocab,
        merges=merges,
        pre_tokenizer_kind=PRE_TOKENIZER_KIND_WHITESPACE,
    )


def _learn_merges(
    corpus: list[str],
    vocab: list[bytes],
    merges: list[Merge],
    target_vocab_size: int,
    min_pair_frequency: int,
) -> None:
    """Iteratively learn merges, appending to ``vocab`` and ``merges``.

    Algorithm:

    1. Pre-tokenize the corpus into chunks via
       :func:`whitespace_pre_tokenize`. Group identical chunks and
       keep their multiplicity — pair statistics are weighted by
       chunk frequency, so a chunk appearing 1000 times contributes
       1000× as many pair counts as a chunk appearing once.

    2. Materialize each unique chunk as a list of base-byte token IDs
       (the byte values themselves, since vocab[i] == bytes([i]) for
       i < 256).

    3. Initialize a global pair-frequency Counter by walking every
       chunk's adjacent-pair list, weighted by chunk frequency.

    4. Loop until vocab_size reached or no eligible pair remains:
       a. Pick the highest-frequency pair (ties broken by pair tuple,
          deterministic).
       b. If its frequency is below ``min_pair_frequency``, stop.
       c. Allocate the new token ID and append to vocab/merges.
       d. For each chunk containing the pair, replace every
          non-overlapping occurrence and update the global pair counts
          incrementally — subtract counts for pairs that disappear,
          add counts for pairs that newly appear.

    The incremental update is the difference between "fast" and
    "infeasibly slow" — recounting from scratch each iteration is
    O(N × merges) where N is total chunk-byte-length, which on SciFact
    would be tens of seconds per iteration.
    """
    # Pre-tokenize and group identical chunks. ``Counter`` does the
    # frequency aggregation in one pass. Empty input yields an empty
    # counter and the main loop exits immediately on no-pairs.
    chunk_counts: Counter[bytes] = Counter()
    for text in corpus:
        for chunk in whitespace_pre_tokenize(text):
            if chunk:
                chunk_counts[chunk] += 1

    # Per-chunk state, indexed by chunk_id (a small int we assign here
    # to avoid hashing bytes objects in the hot loop). Three parallel
    # lists keep the chunk's current token sequence, its frequency,
    # and a fast inverted index from pair → set of chunk_ids that
    # currently contain that pair.
    chunk_tokens: list[list[int]] = []
    chunk_freqs: list[int] = []
    for chunk, freq in chunk_counts.items():
        chunk_tokens.append(list(chunk))  # list of byte values == base token IDs
        chunk_freqs.append(freq)

    # Global pair frequency and per-pair chunk-id index. The index
    # lets us skip chunks that don't contain the pair we're merging,
    # which is the other half of why this is fast.
    pair_freq: Counter[tuple[int, int]] = Counter()
    pair_to_chunks: dict[tuple[int, int], set[int]] = {}
    for cid, tokens in enumerate(chunk_tokens):
        freq = chunk_freqs[cid]
        for i in range(len(tokens) - 1):
            pair = (tokens[i], tokens[i + 1])
            pair_freq[pair] += freq
            pair_to_chunks.setdefault(pair, set()).add(cid)

    # Main loop. Stops when vocab is full, or when no pair beats the
    # frequency floor, or when the pair index is empty.
    while len(vocab) < target_vocab_size:
        if not pair_freq:
            break

        # Argmax with deterministic tie-break: highest frequency, then
        # smallest pair tuple. ``max`` over Counter.items() with a
        # composite key is O(num_distinct_pairs) — the dominant cost
        # per iteration alongside the chunk rewrite below.
        best_pair, best_freq = max(
            pair_freq.items(), key=lambda kv: (kv[1], -kv[0][0], -kv[0][1])
        )
        if best_freq < min_pair_frequency:
            break

        left, right = best_pair
        new_token_id = len(vocab)
        new_token_bytes = vocab[left] + vocab[right]
        vocab.append(new_token_bytes)
        merges.append(Merge(left=left, right=right, result=new_token_id))

        # Apply the merge to every chunk in the pair's index, updating
        # pair_freq and pair_to_chunks incrementally.
        affected_chunks = pair_to_chunks.pop(best_pair)
        # The target pair itself is fully consumed by the merge — every
        # occurrence becomes the new token. Drop it from pair_freq so
        # the next argmax doesn't pick it again. Per-chunk decrement
        # logic in _apply_merge_to_chunk only touches the *neighbouring*
        # pairs (the ones that change because a token id changes around
        # the merge site); the merge-target itself is removed here.
        del pair_freq[best_pair]
        for cid in affected_chunks:
            _apply_merge_to_chunk(
                cid=cid,
                target_pair=best_pair,
                new_token_id=new_token_id,
                chunk_tokens=chunk_tokens,
                chunk_freq=chunk_freqs[cid],
                pair_freq=pair_freq,
                pair_to_chunks=pair_to_chunks,
            )

        # Defensive: if the merged pair somehow lingers in pair_freq
        # with zero count (it shouldn't, given the incremental update),
        # drop it so the next argmax doesn't pick a stale entry.
        if best_pair in pair_freq and pair_freq[best_pair] <= 0:
            del pair_freq[best_pair]


def _apply_merge_to_chunk(
    cid: int,
    target_pair: tuple[int, int],
    new_token_id: int,
    chunk_tokens: list[list[int]],
    chunk_freq: int,
    pair_freq: "Counter[tuple[int, int]]",
    pair_to_chunks: dict[tuple[int, int], set[int]],
) -> None:
    """Rewrite one chunk to apply ``target_pair → new_token_id``.

    Updates ``pair_freq`` and ``pair_to_chunks`` so the global indices
    stay consistent. The trick: when we collapse ``... A target_l
    target_r B ...`` into ``... A new B ...``, three pairs change:

    - ``(target_l, target_r)``: one fewer occurrence (the merge itself)
    - ``(A, target_l)``: replaced by ``(A, new)``
    - ``(target_r, B)``: replaced by ``(new, B)``

    We subtract the old pairs and add the new ones, weighted by
    ``chunk_freq``. Adjacent occurrences of the target pair (e.g.
    ``target_l target_r target_l target_r``) are handled correctly by
    the left-to-right non-overlapping replacement: only the first and
    third positions become merge sites; the second pair `(target_r,
    target_l)` between them is a different pair and was counted
    independently.
    """
    tokens = chunk_tokens[cid]
    new_tokens: list[int] = []
    i = 0
    n = len(tokens)
    while i < n:
        if i + 1 < n and (tokens[i], tokens[i + 1]) == target_pair:
            # Pair to the left of this merge site disappears in its
            # old form, reappears with new_token_id on the right.
            if new_tokens:
                left_neighbor = new_tokens[-1]
                old_left_pair = (left_neighbor, tokens[i])
                _decrement_pair(old_left_pair, chunk_freq, cid, pair_freq, pair_to_chunks)
                new_left_pair = (left_neighbor, new_token_id)
                _increment_pair(new_left_pair, chunk_freq, cid, pair_freq, pair_to_chunks)

            # Pair to the right of this merge site, if any. We compute
            # it before consuming the two tokens. Note: the right
            # neighbor is tokens[i+2], from the *original* sequence —
            # not new_tokens, since new_tokens hasn't received it yet.
            if i + 2 < n:
                right_neighbor = tokens[i + 2]
                old_right_pair = (tokens[i + 1], right_neighbor)
                _decrement_pair(old_right_pair, chunk_freq, cid, pair_freq, pair_to_chunks)
                new_right_pair = (new_token_id, right_neighbor)
                _increment_pair(new_right_pair, chunk_freq, cid, pair_freq, pair_to_chunks)

            new_tokens.append(new_token_id)
            i += 2
        else:
            new_tokens.append(tokens[i])
            i += 1

    chunk_tokens[cid] = new_tokens


def _decrement_pair(
    pair: tuple[int, int],
    freq: int,
    cid: int,
    pair_freq: "Counter[tuple[int, int]]",
    pair_to_chunks: dict[tuple[int, int], set[int]],
) -> None:
    """Subtract ``freq`` from ``pair``'s count; clean up if it hits zero.

    The chunk-id index is conservatively *not* updated here — a chunk
    that loses one occurrence of a pair may still contain another, and
    distinguishing those cases requires re-scanning the chunk. Instead
    we let a stale chunk_id linger in pair_to_chunks; when the pair
    is eventually picked for merging, ``_apply_merge_to_chunk`` walks
    the chunk and naturally finds zero occurrences in stale entries
    (the ``while i < n`` loop falls through with no merges applied).

    The cost is a small amount of wasted iteration on stale chunks;
    the alternative (precise index maintenance) is a noticeable
    constant-factor slowdown for no algorithmic benefit.
    """
    pair_freq[pair] -= freq
    if pair_freq[pair] <= 0:
        del pair_freq[pair]
        pair_to_chunks.pop(pair, None)


def _increment_pair(
    pair: tuple[int, int],
    freq: int,
    cid: int,
    pair_freq: "Counter[tuple[int, int]]",
    pair_to_chunks: dict[tuple[int, int], set[int]],
) -> None:
    """Add ``freq`` to ``pair``'s count and register ``cid`` in the index."""
    pair_freq[pair] += freq
    pair_to_chunks.setdefault(pair, set()).add(cid)