"""PyTorch port of the vectorscry-embedding inference model.

Mirrors the Rust runtime defined in ``byte_text_encoder.rs`` and
``mlp.rs`` (the projector). This module is **parity-critical**: every
layer, shape, padding, and activation choice here must match what the
Rust loader expects from a ``.vse`` file. If you change anything here
without changing the format spec or runtime, you will silently break
deployment.

Conv-stack defaults (``NUM_BLOCKS``, ``KERNEL_SIZE``) below match the
alpha shape used by the byte/BPE text encoder tests in
``vectorscry-embedding``. Both are runtime-configurable on the Rust
side as well — the ``.vse`` descriptor carries them per encoder.

- Activation: GELU (the only ``activation_kind`` v1 supports)
- Byte vocab size: 256 (one row per byte value)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

if TYPE_CHECKING:
    from .bpe_tokenizer import BpeTokenizer


# Alpha defaults for the conv stack. The ``.vse`` descriptor carries
# ``kernel_size`` and a per-encoder block count, so callers can pass
# different values to the encoder constructors.
NUM_BLOCKS = 2
KERNEL_SIZE = 5
BYTE_VOCAB_SIZE = 256


class ConvBlock(nn.Module):
    """A single depthwise-separable conv block with a residual.

    Computes ``out = x + pointwise(GELU(depthwise(x)))``.

    Both depthwise and pointwise preserve channel count (``hidden_dim``),
    so the residual path is unprojected. SAME padding on the depthwise
    keeps the sequence length unchanged.

    Layout note: this module accepts and returns ``(batch, seq_len,
    hidden_dim)`` tensors. Internally we transpose for ``nn.Conv1d``
    (which wants channels-first) and back. The Rust runtime stores
    activations as ``[seq_len, hidden_dim]`` row-major, so this matches
    once you drop the batch dim.

    Parameter mapping to the .vse format:

    - ``self.depthwise.weight``  → ``block.{i}.depthwise.weight``,
      shape ``[hidden_dim, 1, kernel_size]`` in PyTorch but the .vse
      file wants ``[hidden_dim, kernel_size]``. The writer must squeeze
      the singleton channel dim.
    - ``self.depthwise.bias``    → ``block.{i}.depthwise.bias`` (when
      present), shape ``[hidden_dim]``.
    - ``self.pointwise.weight``  → ``block.{i}.pointwise.weight``,
      shape ``[hidden_dim, hidden_dim]`` in both PyTorch (out, in) and
      the .vse file. Direct mapping.
    - ``self.pointwise.bias``    → ``block.{i}.pointwise.bias`` (when
      present), shape ``[hidden_dim]``.
    """

    def __init__(
        self,
        hidden_dim: int,
        kernel_size: int = KERNEL_SIZE,
        depthwise_bias: bool = True,
        pointwise_bias: bool = True,
    ) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            # The Rust SAME-pad implementation handles even kernels by
            # putting an extra zero on the right, but the alpha runtime
            # hardcodes K=5 so we have never exercised that path. Reject
            # here to avoid silent disagreement.
            raise ValueError(
                f"kernel_size must be odd for symmetric SAME padding "
                f"(got {kernel_size}); the Rust runtime hardcodes 5"
            )

        # Depthwise: groups == channels means each output channel
        # depends only on its corresponding input channel — no channel
        # mixing, matching the Rust DepthwiseConv1d.
        self.depthwise = nn.Conv1d(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,  # SAME pad for odd K
            groups=hidden_dim,
            bias=depthwise_bias,
        )
        # Pointwise: the Rust runtime stores this as a Linear, not a
        # kernel-1 Conv1d. We use Linear here too so the parameter
        # shape and name layout match the .vse spec exactly.
        self.pointwise = nn.Linear(
            in_features=hidden_dim,
            out_features=hidden_dim,
            bias=pointwise_bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the block.

        Args:
            x: ``(batch, seq_len, hidden_dim)``.

        Returns:
            ``(batch, seq_len, hidden_dim)``, same shape as input.
        """
        # Conv1d wants (batch, channels, seq_len); we have
        # (batch, seq_len, channels). Transpose, conv, transpose back.
        h = x.transpose(1, 2)            # (B, C, T)
        h = self.depthwise(h)            # (B, C, T), SAME-padded
        h = h.transpose(1, 2)            # (B, T, C)
        h = F.gelu(h)
        h = self.pointwise(h)            # (B, T, C)
        return x + h


def _apply_conv_stack(
    x: torch.Tensor,
    attention_mask: torch.Tensor,
    blocks: nn.ModuleList,
) -> torch.Tensor:
    """Apply the masking dance + conv blocks to a batch of embeddings.

    Shared by :class:`ByteTextEncoder` and :class:`BpeTextEncoder`. The
    pre-block zero-out and per-block re-mask are parity-critical (see
    the class docstrings of either encoder); factoring this out keeps
    the two encoders' forward paths from drifting.

    Args:
        x: ``(batch, seq_len, hidden_dim)`` post-embedding tensor.
        attention_mask: ``(batch, seq_len)`` float tensor, 1.0 at real
            positions and 0.0 at pad positions.
        blocks: the encoder's :class:`ConvBlock` stack.

    Returns:
        ``(batch, seq_len, hidden_dim)`` post-stack tensor.
    """
    mask3 = attention_mask.unsqueeze(-1)     # (B, T, 1) for broadcasting
    x = x * mask3                            # zero pads before block 0
    for block in blocks:
        x = block(x)
        # Re-mask after each block. The block's residual + bias
        # produce nonzero values at pad positions even when the
        # input there was zero; those values would leak into the
        # next block's conv via SAME padding. Zeroing here keeps
        # padded positions inert through the whole stack, which is
        # what the Rust single-example path sees as "out-of-bounds".
        x = x * mask3
    return x


class ByteTextEncoder(nn.Module):
    """Byte-level text encoder: embedding → 2 conv blocks.

    Mirrors ``encoders::text::byte_text_encoder::ByteTextEncoder`` in the
    Rust runtime. Output is the post-conv ``[seq_len, hidden_dim]``
    sequence; pooling and projection are the projector's job.

    Parameter mapping to the .vse format:

    - ``self.embedding.weight`` → ``embedding``, shape ``[256, hidden_dim]``.
    - ``self.blocks[i].*``      → ``block.{i}.*`` (see ``ConvBlock`` docs).

    Notes on training-time inputs:

    The forward expects pre-tokenized, pre-padded ``input_ids`` of shape
    ``(batch, seq_len)`` and an ``attention_mask`` of the same shape with
    1.0 at real-token positions and 0.0 at pad positions. Use
    :func:`tokenize_batch` to produce these from a list of strings.

    The mask is applied **before** the conv stack, zeroing the
    embedding at padded positions. This matters for parity with the
    Rust runtime: Rust processes one string at a time, so its SAME-pad
    convolutions see literal zeros past the end of input. A naive
    batched forward without masking would instead let the conv reach
    into ``embedding[0]`` (the learned row for byte 0), producing
    different outputs at the trailing real-byte positions of a padded
    string vs. an unpadded one. Zeroing the embedding at masked
    positions makes those two paths agree byte-for-byte.

    Mean-pooling downstream then drops masked positions entirely from
    the average, which is also necessary — but the conv-time masking
    is the parity-critical piece.
    """

    def __init__(
        self,
        hidden_dim: int,
        max_seq_len: int,
        kernel_size: int = KERNEL_SIZE,
        num_blocks: int = NUM_BLOCKS,
        depthwise_bias: bool = True,
        pointwise_bias: bool = True,
    ) -> None:
        super().__init__()
        if max_seq_len < 1:
            raise ValueError(f"max_seq_len must be >= 1 (got {max_seq_len})")

        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len
        self.kernel_size = kernel_size
        self.num_blocks = num_blocks

        # Byte vocab is fixed at 256 (one row per byte value).
        self.embedding = nn.Embedding(BYTE_VOCAB_SIZE, hidden_dim)

        self.blocks = nn.ModuleList(
            ConvBlock(
                hidden_dim=hidden_dim,
                kernel_size=kernel_size,
                depthwise_bias=depthwise_bias,
                pointwise_bias=pointwise_bias,
            )
            for _ in range(num_blocks)
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Embed and apply the conv stack.

        Args:
            input_ids: ``(batch, seq_len)`` long tensor of byte values
                in ``[0, 256)``.
            attention_mask: ``(batch, seq_len)`` float tensor with 1.0
                at real-byte positions and 0.0 at pad positions.
                Required even for unpadded batches (use all-ones); it
                is the only signal the encoder has about where padding
                lives, and it must zero the embedding at pad positions
                for parity with the Rust runtime — see the class docs.

        Returns:
            ``(batch, seq_len, hidden_dim)`` float tensor.
        """
        x = self.embedding(input_ids)            # (B, T, C)
        return _apply_conv_stack(x, attention_mask, self.blocks)

    def tokenize_batch(
        self,
        texts: list[str],
        device: torch.device | str | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Convenience wrapper around the module-level :func:`tokenize_batch`.

        Defined on the encoder so callers can write
        ``encoder.tokenize_batch(...)`` and get the right behaviour
        regardless of whether the encoder is byte-level or BPE
        (:class:`BpeTextEncoder` overrides this with BPE encoding).
        """
        if device is None:
            device = self.embedding.weight.device
        return tokenize_batch(texts, max_seq_len=self.max_seq_len, device=device)


class BpeTextEncoder(nn.Module):
    """BPE-tokenized text encoder: embedding → 2 conv blocks.

    Mirrors ``encoders::text::bpe_text_encoder::BpeTextEncoder`` in the
    Rust runtime. Structurally identical to :class:`ByteTextEncoder`
    with two differences:

    1. The embedding table's row count is the BPE vocab size rather
       than fixed at 256.
    2. The encoder owns a :class:`BpeTokenizer` so callers can pass
       raw strings to :meth:`tokenize_batch` without managing
       tokenization separately.

    The conv stack is identical — same block count, same kernel size,
    same residual + GELU + pointwise structure. The shared
    :func:`_apply_conv_stack` helper handles the post-embedding path.

    Parameter mapping to the .vse format:

    - ``self.embedding.weight`` → ``embedding``,
      shape ``[vocab_size, hidden_dim]`` (NOT 256-row).
    - ``self.blocks[i].*``      → ``block.{i}.*`` (same as ByteText).

    The tokenizer is serialized separately into the .vse descriptor's
    ``tokenizer_payload`` field; see :func:`save_composed`. It is held
    as a plain Python attribute (not an ``nn.Module``) so it doesn't
    appear in ``state_dict()`` or ``parameters()`` — it has no
    learned state.

    Tokenization parity with the Rust runtime:

    The Rust ``BpeTextEncoder::encode`` calls
    ``tokenizer.tokenize(input)`` and then ``token_ids.truncate(max_seq_len)``.
    The Python :class:`BpeTokenizer` produces the same token ID
    sequence for the same input (verified against the Rust test suite
    in chunk 2). :meth:`tokenize_batch` runs the tokenizer per text
    and right-pads with token ID 0 to the longest sequence in the
    batch — matching the same masking convention as the byte encoder.

    Note on pad token ID: token 0 is the literal byte ``\\x00`` in
    the BPE vocab (the base-byte invariant). It's a valid token, not
    a special pad sentinel — but the attention_mask zeros its
    embedding contribution before any conv touches it, same as the
    byte encoder. The Rust runtime processes one input at a time so
    it never sees pad tokens at all; our masked-batch path produces
    bit-identical output to a single-example unpadded run.
    """

    def __init__(
        self,
        hidden_dim: int,
        max_seq_len: int,
        tokenizer: "BpeTokenizer",
        kernel_size: int = KERNEL_SIZE,
        num_blocks: int = NUM_BLOCKS,
        depthwise_bias: bool = True,
        pointwise_bias: bool = True,
    ) -> None:
        super().__init__()
        if max_seq_len < 1:
            raise ValueError(f"max_seq_len must be >= 1 (got {max_seq_len})")

        # Vocab size comes from the tokenizer (the authority, per the
        # Rust BpeTextEncoder::new contract). The tokenizer was already
        # validated at its own construction; we just read its size.
        vocab_size = tokenizer.vocab_size()
        if vocab_size < 1:
            # Should be unreachable — BpeTokenizer.__init__ enforces
            # vocab_size >= 256 — but mirror the Rust ZeroVocabSize
            # check for defence in depth.
            raise ValueError(
                f"tokenizer reports vocab_size == {vocab_size}; must be >= 1"
            )

        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len
        self.kernel_size = kernel_size
        self.num_blocks = num_blocks
        self.vocab_size = vocab_size

        # Plain attribute (not registered as a submodule). The
        # tokenizer carries no learned state — registering it would
        # have no upside and would put a non-Module object into the
        # _modules dict, which several PyTorch utilities don't expect.
        self.tokenizer = tokenizer

        self.embedding = nn.Embedding(vocab_size, hidden_dim)

        self.blocks = nn.ModuleList(
            ConvBlock(
                hidden_dim=hidden_dim,
                kernel_size=kernel_size,
                depthwise_bias=depthwise_bias,
                pointwise_bias=pointwise_bias,
            )
            for _ in range(num_blocks)
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Embed and apply the conv stack.

        Args:
            input_ids: ``(batch, seq_len)`` long tensor of BPE token
                IDs in ``[0, vocab_size)``.
            attention_mask: ``(batch, seq_len)`` float tensor, 1.0 at
                real-token positions, 0.0 at pad positions.

        Returns:
            ``(batch, seq_len, hidden_dim)`` float tensor.
        """
        x = self.embedding(input_ids)            # (B, T, C)
        return _apply_conv_stack(x, attention_mask, self.blocks)

    def tokenize_batch(
        self,
        texts: list[str],
        device: torch.device | str | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Tokenize a list of strings via the encoder's BPE tokenizer.

        Each string is passed through :meth:`BpeTokenizer.tokenize`,
        truncated to ``max_seq_len`` tokens (matching
        ``BpeTextEncoder::encode`` on the Rust side which calls
        ``token_ids.truncate(max_seq_len)`` after tokenization), and
        right-padded with token ID 0 to the longest sequence in the
        batch.

        See :func:`bpe_tokenize_batch` for the underlying free
        function — this method just plumbs in the encoder's tokenizer
        and ``max_seq_len``.
        """
        if device is None:
            device = self.embedding.weight.device
        return bpe_tokenize_batch(
            texts,
            tokenizer=self.tokenizer,
            max_seq_len=self.max_seq_len,
            device=device,
        )


def tokenize_batch(
    texts: list[str],
    max_seq_len: int,
    device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode a list of strings to ``(input_ids, attention_mask)`` tensors.

    Each string is UTF-8 encoded, truncated to ``max_seq_len`` bytes
    (matching ``ByteTextEncoder::encode`` on the Rust side which uses
    ``text.bytes().take(max_seq_len)``), then right-padded with byte
    value 0 to the longest sequence in the batch. The attention mask
    is 1.0 at real-byte positions and 0.0 at padded positions.

    Empty strings produce an all-zero row with an all-zero mask. The
    projector's masked-mean handles the empty case (it falls back to a
    zero vector); downstream loss should not crash on this but the
    resulting embedding is meaningless. Filter empty strings upstream
    if you care.

    Returns:
        ``(input_ids: (B, T) long, attention_mask: (B, T) float)`` where
        ``T`` is the maximum truncated byte length across the batch,
        capped at ``max_seq_len``.
    """
    if not texts:
        # Defining the zero-batch case avoids special-casing in the
        # caller. Empty input → empty (0, 0) tensors.
        return (
            torch.zeros((0, 0), dtype=torch.long, device=device),
            torch.zeros((0, 0), dtype=torch.float32, device=device),
        )

    byte_seqs = [t.encode("utf-8")[:max_seq_len] for t in texts]
    batch_seq_len = max((len(b) for b in byte_seqs), default=0)
    if batch_seq_len == 0:
        # Every input was empty after truncation. Return a (B, 0) shape
        # so downstream sees the right batch size.
        b = len(texts)
        return (
            torch.zeros((b, 0), dtype=torch.long, device=device),
            torch.zeros((b, 0), dtype=torch.float32, device=device),
        )

    input_ids = torch.zeros((len(texts), batch_seq_len), dtype=torch.long)
    attention_mask = torch.zeros((len(texts), batch_seq_len), dtype=torch.float32)
    for i, raw in enumerate(byte_seqs):
        n = len(raw)
        if n > 0:
            input_ids[i, :n] = torch.frombuffer(bytearray(raw), dtype=torch.uint8).long()
            attention_mask[i, :n] = 1.0

    return input_ids.to(device), attention_mask.to(device)


def bpe_tokenize_batch(
    texts: list[str],
    tokenizer: "BpeTokenizer",
    max_seq_len: int,
    device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode a list of strings via a BPE tokenizer.

    Each string is tokenized via :meth:`BpeTokenizer.tokenize`,
    truncated to ``max_seq_len`` tokens (matching the Rust
    ``BpeTextEncoder::encode`` which calls
    ``token_ids.truncate(max_seq_len)`` after tokenization), and
    right-padded with token ID 0 to the longest sequence in the
    batch. The attention mask is 1.0 at real-token positions and 0.0
    at padded positions — the same convention as the byte-level
    :func:`tokenize_batch`, so all the downstream masking logic
    (``_apply_conv_stack``, the projector's masked mean) just works.

    Pad token: 0. This is the literal byte ``\\x00`` in the BPE vocab
    — a real, valid token. We rely on the attention mask to suppress
    its contribution; the padded positions never reach the conv via
    :func:`_apply_conv_stack`'s mask3 zero-out. Choosing 0 over a
    sentinel ID keeps the embedding table dense (no extra row for a
    pad token that doesn't appear in any real input) and matches how
    the byte encoder pads.

    Empty strings (and strings the tokenizer maps to zero tokens, e.g.
    pure-whitespace input where the pre-tokenizer might produce
    whitespace-only chunks that BPE leaves as base-byte tokens —
    which is non-empty, but in principle a custom tokenizer could
    return empty) produce an all-zero row with an all-zero mask. The
    projector's masked-mean handles that case (zero divisor → falls
    back to zero vector), so the loss won't crash on empty input. As
    with the byte encoder, filter empties at the dataset layer if you
    care about distillation signal density.

    Returns:
        ``(input_ids: (B, T) long, attention_mask: (B, T) float)`` where
        ``T`` is the maximum truncated token length across the batch,
        capped at ``max_seq_len``.
    """
    if not texts:
        return (
            torch.zeros((0, 0), dtype=torch.long, device=device),
            torch.zeros((0, 0), dtype=torch.float32, device=device),
        )

    # Tokenize and truncate. We do both passes eagerly because we need
    # the per-row length to pick the batch's seq dim; nothing here is
    # hot enough to bother with lazy iteration.
    token_seqs: list[list[int]] = []
    for t in texts:
        ids = tokenizer.tokenize(t)
        if len(ids) > max_seq_len:
            ids = ids[:max_seq_len]
        token_seqs.append(ids)

    batch_seq_len = max((len(s) for s in token_seqs), default=0)
    if batch_seq_len == 0:
        b = len(texts)
        return (
            torch.zeros((b, 0), dtype=torch.long, device=device),
            torch.zeros((b, 0), dtype=torch.float32, device=device),
        )

    input_ids = torch.zeros((len(texts), batch_seq_len), dtype=torch.long)
    attention_mask = torch.zeros((len(texts), batch_seq_len), dtype=torch.float32)
    for i, ids in enumerate(token_seqs):
        n = len(ids)
        if n > 0:
            input_ids[i, :n] = torch.tensor(ids, dtype=torch.long)
            attention_mask[i, :n] = 1.0

    return input_ids.to(device), attention_mask.to(device)


class MlpProjector(nn.Module):
    """Mean-pool → MLP → L2 normalize.

    Mirrors ``projectors::mlp::MlpProjector`` in the Rust runtime. The
    projector takes the encoder's ``(batch, seq_len, hidden_dim)``
    output and produces a single ``(batch, output_dim)`` embedding per
    item.

    The MLP is an arbitrary-depth chain of ``nn.Linear`` layers with
    GELU **between** consecutive layers and no activation after the
    last. Whether each layer has a bias is configured per-layer via the
    ``layer_dims`` and ``layer_biases`` constructor args.

    Parameter mapping to the .vse format:

    - ``self.layers[i].weight`` → ``mlp.{i}.weight``,
      shape ``[out_features, in_features]`` in both PyTorch and the
      .vse spec. Direct mapping.
    - ``self.layers[i].bias``   → ``mlp.{i}.bias`` (when present),
      shape ``[out_features]``.

    Construction takes a list of ``(in_features, out_features)`` pairs
    so that callers (and the .vse loader) can describe the full chain
    without us having to infer it from the first layer's input dim
    plus a list of widths. The first ``in_features`` must equal the
    encoder's ``hidden_dim``; subsequent layers must chain
    (``out_features[i] == in_features[i+1]``).
    """

    def __init__(
        self,
        layer_dims: list[tuple[int, int]],
        layer_biases: list[bool] | None = None,
    ) -> None:
        super().__init__()
        if not layer_dims:
            raise ValueError("MlpProjector requires at least one layer")
        if layer_biases is None:
            layer_biases = [True] * len(layer_dims)
        if len(layer_biases) != len(layer_dims):
            raise ValueError(
                f"layer_biases length ({len(layer_biases)}) must match "
                f"layer_dims length ({len(layer_dims)})"
            )

        # Validate the chain. Same check the .vse parser does at load
        # time — catching it here gives a clean error before training
        # rather than a confusing shape mismatch deep in forward.
        for i in range(len(layer_dims) - 1):
            if layer_dims[i][1] != layer_dims[i + 1][0]:
                raise ValueError(
                    f"layer chain mismatch at index {i}: "
                    f"out_features={layer_dims[i][1]} but next "
                    f"in_features={layer_dims[i + 1][0]}"
                )

        self.input_dim = layer_dims[0][0]
        self.output_dim = layer_dims[-1][1]
        self.layers = nn.ModuleList(
            nn.Linear(in_features=in_f, out_features=out_f, bias=bias)
            for (in_f, out_f), bias in zip(layer_dims, layer_biases)
        )

    def forward(
        self,
        sequence: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Pool, project, and normalize.

        Args:
            sequence: ``(batch, seq_len, hidden_dim)`` from the encoder.
                Padded positions must already be zero (the encoder
                handles this; see ``ByteTextEncoder.forward``).
            attention_mask: ``(batch, seq_len)`` 1.0 at real positions,
                0.0 at padding. Used to compute a masked mean — pad
                positions do not dilute the average.

        Returns:
            ``(batch, output_dim)`` L2-normalized embeddings. For a row
            whose mask is entirely zero (empty input), returns a zero
            vector for that row instead of producing NaN.
        """
        # Masked mean. Pad positions in `sequence` are already zero, so
        # the sum is correct; we divide by the per-row token count.
        # Clamp the divisor away from zero so all-pad rows produce a
        # zero pooled vector instead of NaN — the L2-normalize step
        # passes the zero vector through unchanged, matching the Rust
        # `l2_normalize_in_place` contract for sub-EPSILON norms.
        token_counts = attention_mask.sum(dim=1, keepdim=True)        # (B, 1)
        safe_counts = token_counts.clamp(min=1.0)
        pooled = sequence.sum(dim=1) / safe_counts                    # (B, C)

        # Run the MLP. GELU between layers but not after the last.
        last_idx = len(self.layers) - 1
        h = pooled
        for i, layer in enumerate(self.layers):
            h = layer(h)
            if i != last_idx:
                h = F.gelu(h)

        # L2 normalize. The Rust runtime returns the zero vector
        # unchanged when its pre-norm length is < f32::EPSILON; we
        # match that with a clamped denominator. F.normalize's eps is
        # added to the denominator (not used as a threshold), so for
        # pre-norm vectors of magnitude > eps it scales correctly, and
        # for the zero vector it produces a zero vector — same outcome.
        h = F.normalize(h, p=2.0, dim=-1, eps=1e-12)
        return h


class AttentionPoolMlpProjector(nn.Module):
    """Learned-query attention pool → MLP → L2 normalize.

    Mirrors ``projectors::attention_mlp::AttentionMlpProjector`` in the
    Rust runtime. Same interface as :class:`MlpProjector` — the only
    structural difference is the pooling step: instead of a masked mean,
    a single learned query vector decides how much each token contributes
    to the pooled representation via a softmax over dot-product scores.

    Forward path:

    1. Score each token with the learned query: ``scores[b, t] = <query, sequence[b, t]>``.
       Raw dot product, no ``1/sqrt(hidden_dim)`` scaling — matching the
       Rust forward exactly. Any scaling change here is a parity break.
    2. Mask padded positions with ``-inf`` so they receive zero weight
       under softmax. Rust never sees pad positions (it processes one
       un-padded sequence at a time), so this masking is a no-op for
       parity: a Python row of length T with K real tokens produces the
       same pooled vector as the Rust runtime would on the unpadded
       length-K sequence.
    3. Softmax over the time dimension. PyTorch's ``softmax`` already
       does the log-sum-exp trick (subtract the max before ``exp``)
       internally, matching Rust's stable form.
    4. Weighted sum: ``pooled = Σ_t weights[b, t] · sequence[b, t]``.
    5. MLP chain: GELU between layers, no activation after the last
       (identical to :class:`MlpProjector` post-pool).
    6. L2 normalize.

    The learned query is initialized to zeros so step-zero attention
    weights are uniform and the pooled vector equals the mean — training
    starts from the same baseline as a mean-pool projector and learns
    away from there. An ``ones / sqrt(hidden_dim)`` initialization would
    not give uniform weights since scores would still depend on the
    encoder outputs.

    Edge case for rows whose mask is entirely zero: a softmax of all
    ``-inf`` scores is ``NaN``. We replace those rows' scores with zeros
    so softmax produces uniform weights; since the encoder zero-pads
    every position in such a row the pooled vector is zero anyway, and
    :class:`ComposedEmbedder` zeroes the row's output post-projector.
    The point of this branch is just to keep ``NaN`` out of the graph.

    Parameter mapping to the .vse format:

    - ``self.query`` → ``attention.query``, shape ``[hidden_dim]``.
    - ``self.layers[i].weight`` → ``mlp.{i}.weight``, same as
      :class:`MlpProjector`.
    - ``self.layers[i].bias`` → ``mlp.{i}.bias`` (when present).
    """

    def __init__(
        self,
        hidden_dim: int,
        layer_dims: list[tuple[int, int]],
        layer_biases: list[bool] | None = None,
    ) -> None:
        super().__init__()
        if not layer_dims:
            raise ValueError("AttentionPoolMlpProjector requires at least one layer")
        if layer_biases is None:
            layer_biases = [True] * len(layer_dims)
        if len(layer_biases) != len(layer_dims):
            raise ValueError(
                f"layer_biases length ({len(layer_biases)}) must match "
                f"layer_dims length ({len(layer_dims)})"
            )
        if layer_dims[0][0] != hidden_dim:
            raise ValueError(
                f"first layer in_features ({layer_dims[0][0]}) must equal "
                f"hidden_dim ({hidden_dim}); the learned query has shape "
                f"({hidden_dim},) and the pooled vector feeds the first "
                f"MLP layer"
            )

        # Same chain validation as MlpProjector — same wire format, same
        # constraints. Catch it here, not in forward.
        for i in range(len(layer_dims) - 1):
            if layer_dims[i][1] != layer_dims[i + 1][0]:
                raise ValueError(
                    f"layer chain mismatch at index {i}: "
                    f"out_features={layer_dims[i][1]} but next "
                    f"in_features={layer_dims[i + 1][0]}"
                )

        self.hidden_dim = hidden_dim
        self.input_dim = layer_dims[0][0]
        self.output_dim = layer_dims[-1][1]
        self.query = nn.Parameter(torch.zeros(hidden_dim))
        self.layers = nn.ModuleList(
            nn.Linear(in_features=in_f, out_features=out_f, bias=bias)
            for (in_f, out_f), bias in zip(layer_dims, layer_biases)
        )

    def forward(
        self,
        sequence: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Pool by attention, project, and normalize.

        Args:
            sequence: ``(batch, seq_len, hidden_dim)`` from the encoder.
                Padded positions must already be zero.
            attention_mask: ``(batch, seq_len)`` 1.0 at real positions,
                0.0 at padding. Pad positions are excluded from the
                softmax via a ``-inf`` fill.

        Returns:
            ``(batch, output_dim)`` L2-normalized embeddings. Rows whose
            mask is entirely zero are zeroed by :class:`ComposedEmbedder`
            after this projector returns; this method only guarantees
            no ``NaN`` in such rows.
        """
        # Raw dot product against the learned query — no 1/sqrt(d) scale.
        # Rust does the same; introducing a scale here would be a parity
        # break. einsum is shape-self-documenting; the equivalent
        # `(sequence * self.query).sum(-1)` produces identical numerics.
        scores = torch.einsum("btc,c->bt", sequence, self.query)        # (B, T)
        scores = scores.masked_fill(attention_mask == 0, float("-inf"))

        # Replace all-pad rows' scores with zeros so softmax produces
        # uniform-but-finite weights. The encoder has zeroed every
        # position in those rows already, so the pooled vector is zero
        # regardless of the weights — this branch exists purely to keep
        # NaN out of the graph (otherwise gradients break for the whole
        # batch even though those rows are zeroed downstream).
        any_real = (attention_mask.sum(dim=1, keepdim=True) > 0)        # (B, 1)
        scores = torch.where(any_real, scores, torch.zeros_like(scores))

        weights = torch.softmax(scores, dim=1)                          # (B, T)
        pooled = (weights.unsqueeze(-1) * sequence).sum(dim=1)          # (B, hidden_dim)

        # MLP — duplicated from MlpProjector rather than refactored to a
        # shared helper. The forward path of a projector is short enough
        # that reading it top-to-bottom in one place is more valuable
        # than DRY.
        last_idx = len(self.layers) - 1
        h = pooled
        for i, layer in enumerate(self.layers):
            h = layer(h)
            if i != last_idx:
                h = F.gelu(h)

        return F.normalize(h, p=2.0, dim=-1, eps=1e-12)


class ComposedEmbedder(nn.Module):
    """Encoder + projector wired together.

    Mirrors ``Composed<E, P>`` on the Rust side: the user-facing
    embedder. Takes a batch of (input_ids, attention_mask) and produces
    a batch of L2-normalized embedding vectors.

    This is the module trained end-to-end by the distillation loop and
    serialized to ``.vse`` for deployment.
    """

    def __init__(
        self,
        encoder: "ByteTextEncoder | BpeTextEncoder",
        projector: "MlpProjector | AttentionPoolMlpProjector",
    ) -> None:
        super().__init__()
        if encoder.hidden_dim != projector.input_dim:
            raise ValueError(
                f"encoder hidden_dim ({encoder.hidden_dim}) must equal "
                f"projector input_dim ({projector.input_dim}); this is "
                f"the same EncoderProjectorDimMismatch the .vse parser "
                f"would report at load time"
            )
        self.encoder = encoder
        self.projector = projector

    @property
    def embedding_dim(self) -> int:
        """Output dimension — the size of vectors this embedder produces."""
        return self.projector.output_dim

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Embed a batch.

        Args:
            input_ids: ``(batch, seq_len)`` long tensor.
            attention_mask: ``(batch, seq_len)`` float tensor.

        Returns:
            ``(batch, output_dim)`` L2-normalized embeddings. Rows
            whose attention_mask is entirely zero (empty input) are
            returned as zero vectors. The Rust runtime errors on
            empty input upstream; we choose to keep the batch alive
            instead — a single empty string shouldn't crash a long
            training run. Filter empties at the dataset layer if you
            want strict behaviour.
        """
        sequence = self.encoder(input_ids, attention_mask)
        embeddings = self.projector(sequence, attention_mask)

        # Zero out rows that had no real tokens. Without this, the MLP
        # biases turn a zero pooled vector into a non-zero output, which
        # L2-normalize then turns into a deterministic-but-meaningless
        # unit vector — silently bad data for distillation.
        has_tokens = (attention_mask.sum(dim=1) > 0).to(embeddings.dtype)  # (B,)
        return embeddings * has_tokens.unsqueeze(-1)

    def embed_texts(
        self,
        texts: list[str],
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        """Convenience wrapper: tokenize then forward.

        Useful for parity tests and quick checks. Training loops will
        usually tokenize once into a DataLoader-friendly dataset and
        skip this path.

        Tokenization is delegated to the encoder so this method works
        uniformly for byte-level and BPE encoders. The encoder's
        ``tokenize_batch`` knows its own ``max_seq_len`` and (for BPE)
        which tokenizer to use.
        """
        if device is None:
            device = next(self.parameters()).device
        ids, mask = self.encoder.tokenize_batch(texts, device=device)
        return self(ids, mask)