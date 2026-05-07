"""Writer for the .vse v1 weight format.

Specification of record: ``serialization.md`` in the embedding crate.
This module implements the writer side; the Rust crate implements the
parser. They are tested against each other via round-trip: build a
model, write it here, parse it on the Rust side, embed the same input
both ways, and compare.

Design choices that matter:

- Pure stdlib + numpy. No torch dep at write time — we accept the
  weights as already-extracted numpy arrays so this module can be
  tested without torch installed.
- Two-pass writer (per the spec's implementation notes): first
  enumerate tensors and assign 4-aligned offsets within the data
  block, then emit header + descriptor + metadata + padding + data.
- f32-only tensors in v1.0. The .vse format supports i8 with quant
  params, but post-training quantization is a follow-up task; getting
  the f32 path round-trip-verified first is the priority.
"""

from __future__ import annotations

import struct


# Magic bytes from the spec: ASCII "VSE1".
MAGIC = b"VSE1"
FORMAT_VERSION = 1

# Tag values from the spec. Repeated here as named constants so the
# writer code reads as "encoder kind = ByteText" not "encoder kind = 1".
ENCODER_KIND_BYTE_TEXT = 0x01
ENCODER_KIND_BPE_TEXT = 0x02
PROJECTOR_KIND_MLP = 0x01
PROJECTOR_KIND_ATTENTION_POOL_MLP = 0x02
ACTIVATION_GELU = 0x01
DTYPE_F32 = 0x01
DTYPE_I8 = 0x02

# Tokenizer kind tags for the BpeText descriptor's tokenizer_kind
# field. Only EmbeddedBpeV1 is defined in v1.0; the payload is a
# complete .bpe v1 binary (magic "VSBP"). Future kinds (e.g. an
# externally-referenced tokenizer file) would land as new tags.
TOKENIZER_KIND_EMBEDDED_BPE_V1 = 0x01


def _u8(v: int) -> bytes:
    """Encode a u8. Raises on out-of-range so a stray negative or oversize
    value fails fast instead of wrapping silently."""
    if not 0 <= v <= 0xFF:
        raise ValueError(f"u8 value out of range: {v}")
    return struct.pack("<B", v)


def _u32(v: int) -> bytes:
    """Encode a u32 little-endian. Same range check as ``_u8``."""
    if not 0 <= v <= 0xFFFF_FFFF:
        raise ValueError(f"u32 value out of range: {v}")
    return struct.pack("<I", v)


def _f32(v: float) -> bytes:
    """Encode a single-precision float little-endian."""
    return struct.pack("<f", v)


def _length_prefixed_string(s: str) -> bytes:
    """Encode a UTF-8 string as ``u32 length || bytes`` per spec §Strings.

    No null terminator; length is in bytes, not characters. The spec
    forbids empty names for tensors — we enforce that here so a writer
    bug surfaces immediately rather than producing a parser-rejectable
    file.
    """
    encoded = s.encode("utf-8")
    if not encoded:
        raise ValueError("tensor names must be non-empty per spec")
    return _u32(len(encoded)) + encoded


def _pad_to_4(offset: int) -> int:
    """Return the smallest multiple of 4 >= ``offset``.

    Used in two places: the alignment between the metadata table and
    the tensor data block, and the per-tensor offset assignment within
    the data block. f32 tensors require 4-byte alignment for the
    zero-copy ``&[u8]`` → ``&[f32]`` reinterpretation on the Rust side.
    """
    return (offset + 3) & ~3


# ---------------------------------------------------------------------------
# Architecture description
# ---------------------------------------------------------------------------
#
# These dataclasses are the writer's input contract. They mirror the
# spec's architecture descriptor section exactly — every field here
# becomes one or more bytes in the output. We intentionally do NOT
# infer fields from tensor shapes: the descriptor is supposed to
# declare what the writer intends, and the parser cross-checks against
# the metadata table. Keeping these explicit catches descriptor/tensor
# disagreements at write time, before the file ever hits a parser.

from dataclasses import dataclass


@dataclass(frozen=True)
class ConvBlockSpec:
    """Per-conv-block bias presence. Two bytes on the wire."""
    depthwise_has_bias: bool
    pointwise_has_bias: bool


@dataclass(frozen=True)
class MlpLayerSpec:
    """Per-MLP-layer dimensions and bias presence. Nine bytes on the wire."""
    in_features: int
    out_features: int
    has_bias: bool


@dataclass(frozen=True)
class ByteTextEncoderSpec:
    """Architecture descriptor for the ``ByteText`` encoder kind."""
    hidden_dim: int
    max_seq_len: int
    kernel_size: int
    blocks: list[ConvBlockSpec]

    @property
    def num_blocks(self) -> int:
        return len(self.blocks)

    @property
    def vocab_size(self) -> int:
        """Embedding row count. Fixed at 256 for the byte vocab."""
        return 256


@dataclass(frozen=True)
class BpeTextEncoderSpec:
    """Architecture descriptor for the ``BpeText`` encoder kind.

    Structurally identical to :class:`ByteTextEncoderSpec` plus three
    tokenizer-related fields. The conv stack is the same (same
    `hidden_dim`, `kernel_size`, per-block bias spec); the encoder
    differs only in vocabulary size (configurable, not fixed at 256)
    and in carrying its tokenizer's serialized state.

    The embedded tokenizer travels inside the .vse file as opaque
    bytes. The format is determined by ``tokenizer_kind``: v1 only
    defines :data:`TOKENIZER_KIND_EMBEDDED_BPE_V1` (0x01), whose
    payload is a complete .bpe v1 binary built by
    :func:`vectorscry_train.bpe_format.write_bpe`.

    The writer enforces three-way agreement between this spec's
    ``vocab_size``, the embedding tensor's row count, and (when the
    tokenizer payload is parseable here) the tokenizer's own vocab
    size. The Rust loader does the same check at load time and
    surfaces mismatches as ``TokenizerVocabMismatch``.
    """
    hidden_dim: int
    max_seq_len: int
    kernel_size: int
    blocks: list[ConvBlockSpec]
    vocab_size: int
    tokenizer_kind: int  # TOKENIZER_KIND_EMBEDDED_BPE_V1
    tokenizer_payload: bytes

    @property
    def num_blocks(self) -> int:
        return len(self.blocks)


@dataclass(frozen=True)
class MlpProjectorSpec:
    """Architecture descriptor for the ``Mlp`` projector kind.

    Activation is fixed at GELU in v1; the field exists for forward
    compatibility but only one value is currently legal.
    """
    activation_kind: int  # ACTIVATION_GELU
    layers: list[MlpLayerSpec]

    @property
    def num_layers(self) -> int:
        return len(self.layers)


@dataclass(frozen=True)
class AttentionPoolMlpProjectorSpec:
    """Architecture descriptor for the ``AttentionMlp`` projector kind.

    Wire-format identical to :class:`MlpProjectorSpec` except for the
    leading kind byte (``0x02`` vs ``0x01``). The learned query vector
    travels as a tensor named ``attention.query`` of shape
    ``(hidden_dim,)``, not as a descriptor field — the descriptor only
    declares the MLP chain that follows the pool.

    Parity with the Rust runtime: see
    ``crates/vectorscry-embedding/src/projectors/attention_mlp.rs`` for
    the loader, and the matching ``Mlp`` arms in ``serialization/parser.rs``
    / ``serialization/load.rs``. The Python writer must match the Rust
    reader byte-for-byte.

    Activation is fixed at GELU in v1; the field exists for forward
    compatibility but only one value is currently legal.
    """
    activation_kind: int  # ACTIVATION_GELU
    layers: list[MlpLayerSpec]

    @property
    def num_layers(self) -> int:
        return len(self.layers)


def _encode_arch_descriptor(
    encoder: "ByteTextEncoderSpec | BpeTextEncoderSpec",
    projector: "MlpProjectorSpec | AttentionPoolMlpProjectorSpec",
) -> bytes:
    """Serialize the architecture descriptor section.

    Dispatches on encoder kind: :class:`ByteTextEncoderSpec` produces
    a ``ByteText`` (kind 0x01) descriptor; :class:`BpeTextEncoderSpec`
    produces a ``BpeText`` (kind 0x02) descriptor with the embedded
    tokenizer payload appended after the conv-block specs.

    Common fields (encoder shared between byte and BPE):

        encoder_kind (u8)
        hidden_dim (u32 LE)
        max_seq_len (u32 LE)
        kernel_size (u8)
        num_blocks (u8)
        block_specs (num_blocks × 2 bytes)

    BpeText only, appended after block_specs:

        vocab_size            (u32 LE)
        tokenizer_kind        (u8)
        tokenizer_payload_len (u32 LE)
        tokenizer_payload     ([u8; tokenizer_payload_len])

    Then the projector section, identical for both encoder kinds and
    byte-identical between the two projector kinds aside from the
    leading ``projector_kind`` byte (``0x01`` Mlp vs ``0x02``
    AttentionMlp). The learned query of the AttentionMlp projector is
    not a descriptor field — it travels as a tensor named
    ``attention.query``.

        projector_kind (u8)
        activation_kind (u8)
        num_layers (u8)
        layer_specs (num_layers × 9 bytes)
    """
    # Shared validation. ``vocab_size`` lives on both spec types via
    # the property defined on ByteTextEncoderSpec; for BpeTextEncoderSpec
    # it's a real field. Either way the check is uniform.
    if encoder.max_seq_len < 1:
        raise ValueError(f"max_seq_len must be >= 1 (got {encoder.max_seq_len})")
    if encoder.kernel_size < 1:
        raise ValueError(f"kernel_size must be >= 1 (got {encoder.kernel_size})")
    if not encoder.blocks:
        raise ValueError("encoder must have at least one conv block")
    if len(encoder.blocks) > 255:
        raise ValueError(
            f"too many conv blocks: {len(encoder.blocks)} (u8 limit is 255)"
        )
    if encoder.kernel_size > 255:
        raise ValueError(
            f"kernel_size too large: {encoder.kernel_size} (u8 limit is 255)"
        )

    if not projector.layers:
        raise ValueError("projector must have at least one layer")
    if len(projector.layers) > 255:
        raise ValueError(
            f"too many MLP layers: {len(projector.layers)} (u8 limit is 255)"
        )

    # Validate the layer chain. The parser will check this too, but
    # catching it here gives a clearer error than "your file is malformed."
    for i in range(len(projector.layers) - 1):
        a, b = projector.layers[i], projector.layers[i + 1]
        if a.out_features != b.in_features:
            raise ValueError(
                f"MLP layer chain mismatch at index {i}: "
                f"out_features={a.out_features} but next "
                f"in_features={b.in_features}"
            )

    # Encoder/projector dimension agreement.
    if projector.layers[0].in_features != encoder.hidden_dim:
        raise ValueError(
            f"first MLP layer's in_features ({projector.layers[0].in_features}) "
            f"must equal encoder hidden_dim ({encoder.hidden_dim})"
        )

    parts: list[bytes] = []

    # Encoder section. Dispatch on type for the kind byte and the
    # tokenizer-trailer (BpeText only).
    if isinstance(encoder, ByteTextEncoderSpec):
        parts.append(_u8(ENCODER_KIND_BYTE_TEXT))
        parts.extend(_encode_conv_stack_common(encoder))
    elif isinstance(encoder, BpeTextEncoderSpec):
        parts.append(_u8(ENCODER_KIND_BPE_TEXT))
        parts.extend(_encode_conv_stack_common(encoder))
        parts.extend(_encode_bpe_tokenizer_trailer(encoder))
    else:
        raise TypeError(
            f"unknown encoder spec type: {type(encoder).__name__}"
        )

    # Projector section. Same body for both projector kinds — only the
    # kind byte differs. Dispatch is by spec type, not a parameter, so
    # callers can't accidentally write a kind tag that doesn't match the
    # tensor list the rest of the writer is about to emit.
    if isinstance(projector, MlpProjectorSpec):
        parts.append(_u8(PROJECTOR_KIND_MLP))
    elif isinstance(projector, AttentionPoolMlpProjectorSpec):
        parts.append(_u8(PROJECTOR_KIND_ATTENTION_POOL_MLP))
    else:
        raise TypeError(
            f"unknown projector spec type: {type(projector).__name__}"
        )
    parts.append(_u8(projector.activation_kind))
    parts.append(_u8(projector.num_layers))
    for layer in projector.layers:
        parts.append(_u32(layer.in_features))
        parts.append(_u32(layer.out_features))
        parts.append(_u8(1 if layer.has_bias else 0))

    return b"".join(parts)


def _encode_conv_stack_common(
    encoder: "ByteTextEncoderSpec | BpeTextEncoderSpec",
) -> list[bytes]:
    """Encode the conv-stack fields shared by ByteText and BpeText.

    Layout (from ``hidden_dim`` through ``block_specs``):

        hidden_dim   (u32 LE)
        max_seq_len  (u32 LE)
        kernel_size  (u8)
        num_blocks   (u8)
        block_specs  (num_blocks × 2 bytes)

    Caller has already emitted the ``encoder_kind`` byte and is
    responsible for any kind-specific trailing fields.
    """
    parts: list[bytes] = [
        _u32(encoder.hidden_dim),
        _u32(encoder.max_seq_len),
        _u8(encoder.kernel_size),
        _u8(encoder.num_blocks),
    ]
    for block in encoder.blocks:
        parts.append(_u8(1 if block.depthwise_has_bias else 0))
        parts.append(_u8(1 if block.pointwise_has_bias else 0))
    return parts


def _encode_bpe_tokenizer_trailer(encoder: BpeTextEncoderSpec) -> list[bytes]:
    """Encode BpeText-only fields after the conv-block specs.

    Layout:

        vocab_size            (u32 LE)
        tokenizer_kind        (u8)
        tokenizer_payload_len (u32 LE)
        tokenizer_payload     ([u8; tokenizer_payload_len])

    Validates that ``vocab_size`` is at least 1 (the spec requires it,
    and a zero would also be rejected by the loader as
    ``InvalidVocabSize``) and that ``tokenizer_kind`` is the only
    value v1 currently defines. Future kinds can be added by extending
    the check here.
    """
    if encoder.vocab_size < 1:
        raise ValueError(
            f"vocab_size must be >= 1 (got {encoder.vocab_size})"
        )
    if encoder.vocab_size > 0xFFFF_FFFF:
        # u32 wire field. A vocab this large is unrealistic but the
        # range check belongs here, not as a runtime surprise.
        raise ValueError(
            f"vocab_size too large: {encoder.vocab_size} (u32 limit)"
        )
    if encoder.tokenizer_kind != TOKENIZER_KIND_EMBEDDED_BPE_V1:
        raise ValueError(
            f"unsupported tokenizer_kind {encoder.tokenizer_kind:#x}; "
            f"only {TOKENIZER_KIND_EMBEDDED_BPE_V1:#x} (EmbeddedBpeV1) "
            f"is defined in v1"
        )
    if len(encoder.tokenizer_payload) > 0xFFFF_FFFF:
        raise ValueError(
            f"tokenizer_payload too large: {len(encoder.tokenizer_payload)} "
            f"bytes (u32 limit)"
        )

    parts: list[bytes] = [
        _u32(encoder.vocab_size),
        _u8(encoder.tokenizer_kind),
        _u32(len(encoder.tokenizer_payload)),
        encoder.tokenizer_payload,
    ]
    return parts


# ---------------------------------------------------------------------------
# Tensor metadata + data block
# ---------------------------------------------------------------------------


@dataclass
class TensorEntry:
    """A single named tensor to be written.

    ``data`` holds the raw little-endian bytes of the tensor's element
    array, in row-major order. The writer does not interpret the bytes
    — it expects the caller (or the high-level ``save_composed`` path)
    to have already serialized via ``arr.astype('<f4').tobytes()`` or
    equivalent. This keeps the writer framework-agnostic.

    ``shape`` is the logical shape (e.g. ``[256, hidden_dim]`` for the
    embedding table). ``data`` length must equal
    ``prod(shape) * sizeof(element)`` — the writer checks this so a
    mismatch surfaces here rather than as ``TensorSizeInconsistent``
    on the Rust side.
    """
    name: str
    dtype: int            # DTYPE_F32 or DTYPE_I8
    shape: tuple[int, ...]
    data: bytes
    # quant params, only meaningful for DTYPE_I8. f32 path leaves these None.
    quant_scale: float | None = None
    quant_zero_point: int | None = None


def _element_size(dtype: int) -> int:
    if dtype == DTYPE_F32:
        return 4
    if dtype == DTYPE_I8:
        return 1
    raise ValueError(f"unknown dtype tag: {dtype}")


def _validate_tensor(t: TensorEntry) -> None:
    """Check shape/data consistency before any layout work."""
    if t.dtype not in (DTYPE_F32, DTYPE_I8):
        raise ValueError(f"tensor {t.name!r}: unknown dtype {t.dtype}")
    if not t.shape:
        raise ValueError(f"tensor {t.name!r}: empty shape (rank 0 not supported in v1)")
    if len(t.shape) > 2:
        raise ValueError(
            f"tensor {t.name!r}: rank {len(t.shape)} not supported in v1 (max 2)"
        )
    for dim in t.shape:
        if dim <= 0:
            raise ValueError(f"tensor {t.name!r}: invalid shape {t.shape}")

    expected_elements = 1
    for dim in t.shape:
        expected_elements *= dim
    expected_bytes = expected_elements * _element_size(t.dtype)
    if len(t.data) != expected_bytes:
        raise ValueError(
            f"tensor {t.name!r}: data length {len(t.data)} bytes does not "
            f"match shape {t.shape} × {_element_size(t.dtype)}-byte element "
            f"= {expected_bytes} bytes"
        )

    if t.dtype == DTYPE_I8:
        if t.quant_scale is None or t.quant_zero_point is None:
            raise ValueError(
                f"tensor {t.name!r}: i8 dtype requires scale and zero_point"
            )
        if not (t.quant_scale > 0 and _is_finite(t.quant_scale)):
            raise ValueError(
                f"tensor {t.name!r}: scale must be strictly positive and finite "
                f"(got {t.quant_scale})"
            )
        if not -128 <= t.quant_zero_point <= 127:
            raise ValueError(
                f"tensor {t.name!r}: zero_point out of i8 range "
                f"(got {t.quant_zero_point})"
            )


def _is_finite(x: float) -> bool:
    """Local NaN/inf check that doesn't depend on math.isfinite for clarity."""
    return x == x and x not in (float("inf"), float("-inf"))


def _layout_data_block(tensors: list[TensorEntry]) -> list[int]:
    """Assign 4-byte-aligned offsets within the data block, one per tensor.

    Returns offsets in the same order as the input list. The data block
    layout is contiguous: each tensor's data starts at the next 4-aligned
    offset after the previous tensor's end. Per spec §Alignment, every
    tensor's offset must be a multiple of 4 even for i8 (the spec
    enforces this for layout uniformity, not correctness).

    Note: the spec permits non-contiguous layouts and tolerates interior
    holes, but we always emit contiguous to keep the file as small as
    possible.
    """
    offsets = []
    cursor = 0
    for t in tensors:
        cursor = _pad_to_4(cursor)
        offsets.append(cursor)
        cursor += len(t.data)
    return offsets


def _encode_tensor_metadata(
    tensors: list[TensorEntry],
    data_offsets: list[int],
) -> bytes:
    """Serialize the tensor metadata table.

    Layout per spec §Tensor metadata table:

        num_tensors (u32 LE)
        for each tensor:
            name_length (u32 LE)
            name bytes
            dtype (u8)
            rank (u8)
            shape (rank × u32 LE)
            quant_params (8 bytes; i8 only)
            data_offset (u32 LE)  - relative to data block start
            data_length (u32 LE)
    """
    seen_names: set[str] = set()
    parts: list[bytes] = [_u32(len(tensors))]

    for t, offset in zip(tensors, data_offsets):
        if t.name in seen_names:
            raise ValueError(f"duplicate tensor name: {t.name!r}")
        seen_names.add(t.name)

        parts.append(_length_prefixed_string(t.name))
        parts.append(_u8(t.dtype))
        parts.append(_u8(len(t.shape)))
        for dim in t.shape:
            parts.append(_u32(dim))
        if t.dtype == DTYPE_I8:
            # 8 bytes: scale (f32) || zero_point (i8) || 3 reserved bytes.
            # Reserved bytes are zero per the v1 writer contract; the
            # parser must accept any value, but emitting zeros is the
            # documented good-citizen behaviour.
            assert t.quant_scale is not None and t.quant_zero_point is not None
            parts.append(_f32(t.quant_scale))
            parts.append(struct.pack("<b", t.quant_zero_point))
            parts.append(b"\x00\x00\x00")
        parts.append(_u32(offset))
        parts.append(_u32(len(t.data)))

    return b"".join(parts)


# ---------------------------------------------------------------------------
# Top-level writer
# ---------------------------------------------------------------------------


def write_vse(
    encoder: "ByteTextEncoderSpec | BpeTextEncoderSpec",
    projector: "MlpProjectorSpec | AttentionPoolMlpProjectorSpec",
    tensors: list[TensorEntry],
) -> bytes:
    """Build a complete .vse v1 file.

    Returns the file bytes; caller writes them to disk. Buffer-based
    rather than file-based so callers can hash, transmit, or test
    without touching the filesystem.

    Validation done here (the rest is delegated to the helpers):

    - Each tensor's data length matches its declared shape.
    - Tensor names are unique.
    - The architecture descriptor's bias declarations agree with which
      bias tensors are actually present. This is the cross-check the
      parser performs at load time; failing it here means we'd ship a
      file that won't load, so we want the writer to fail first.
    - Required tensors (by architecture) are all present.
    """
    for t in tensors:
        _validate_tensor(t)

    _check_required_tensors(encoder, projector, tensors)

    descriptor_bytes = _encode_arch_descriptor(encoder, projector)
    data_offsets = _layout_data_block(tensors)
    metadata_bytes = _encode_tensor_metadata(tensors, data_offsets)

    # Header.
    header = (
        MAGIC
        + _u32(FORMAT_VERSION)
        + _u32(len(descriptor_bytes))
        + _u32(len(metadata_bytes))
    )
    assert len(header) == 16

    # Data block: lay out each tensor's bytes at its assigned offset,
    # padding between tensors with zero bytes to maintain 4-byte
    # alignment. cursor walks alongside data_offsets to keep them in sync.
    data_chunks: list[bytes] = []
    cursor = 0
    for t, offset in zip(tensors, data_offsets):
        if cursor < offset:
            data_chunks.append(b"\x00" * (offset - cursor))
            cursor = offset
        data_chunks.append(t.data)
        cursor += len(t.data)
    data_block = b"".join(data_chunks)

    # Pre-data alignment padding: the data block must start at a file
    # offset that is a multiple of 4. The metadata table sits at offset
    # 16 + len(descriptor); compute the padding to bring the cursor up.
    pre_data_offset = 16 + len(descriptor_bytes) + len(metadata_bytes)
    aligned = _pad_to_4(pre_data_offset)
    pre_data_padding = b"\x00" * (aligned - pre_data_offset)

    return header + descriptor_bytes + metadata_bytes + pre_data_padding + data_block


def _check_required_tensors(
    encoder: "ByteTextEncoderSpec | BpeTextEncoderSpec",
    projector: "MlpProjectorSpec | AttentionPoolMlpProjectorSpec",
    tensors: list[TensorEntry],
) -> None:
    """Enforce the spec's bias-presence cross-check and shape rules.

    The .vse parser performs this same check at load time. We do it
    here too so a writer bug surfaces before the file is shipped.

    The embedding tensor's expected row count is taken from
    ``encoder.vocab_size``: 256 for ByteText, the configured vocab
    size for BpeText. Both spec types expose this property so the
    check is uniform.
    """
    by_name = {t.name: t for t in tensors}

    def require(name: str, shape: tuple[int, ...]) -> None:
        t = by_name.get(name)
        if t is None:
            raise ValueError(f"required tensor missing: {name!r}")
        if tuple(t.shape) != shape:
            raise ValueError(
                f"tensor {name!r}: expected shape {shape}, got {tuple(t.shape)}"
            )

    def forbid(name: str) -> None:
        if name in by_name:
            raise ValueError(
                f"tensor {name!r} present but architecture descriptor "
                f"declares no bias for that layer"
            )

    # Embedding table. Row count is encoder.vocab_size — 256 for
    # ByteText (fixed) or the configured value for BpeText.
    require("embedding", (encoder.vocab_size, encoder.hidden_dim))

    # Conv blocks.
    for i, block in enumerate(encoder.blocks):
        require(f"block.{i}.depthwise.weight", (encoder.hidden_dim, encoder.kernel_size))
        if block.depthwise_has_bias:
            require(f"block.{i}.depthwise.bias", (encoder.hidden_dim,))
        else:
            forbid(f"block.{i}.depthwise.bias")
        require(f"block.{i}.pointwise.weight", (encoder.hidden_dim, encoder.hidden_dim))
        if block.pointwise_has_bias:
            require(f"block.{i}.pointwise.bias", (encoder.hidden_dim,))
        else:
            forbid(f"block.{i}.pointwise.bias")

    # MLP layers. Same names and shapes for both projector kinds — the
    # post-pool MLP chain is structurally identical; only the pool step
    # differs upstream of it.
    for i, layer in enumerate(projector.layers):
        require(f"mlp.{i}.weight", (layer.out_features, layer.in_features))
        if layer.has_bias:
            require(f"mlp.{i}.bias", (layer.out_features,))
        else:
            forbid(f"mlp.{i}.bias")

    # Attention pool's learned query — present iff projector is the
    # AttentionMlp kind. Forbid the tensor for plain Mlp so a stray
    # extra tensor doesn't sneak into a file the loader would then
    # reject as unknown.
    if isinstance(projector, AttentionPoolMlpProjectorSpec):
        require("attention.query", (encoder.hidden_dim,))
    else:
        forbid("attention.query")


# ---------------------------------------------------------------------------
# High-level convenience: save a ComposedEmbedder
# ---------------------------------------------------------------------------


def _tensor_to_f32_bytes(t) -> bytes:
    """Convert a torch tensor (or anything with a numpy() method) to
    little-endian f32 bytes in row-major order.

    The spec requires LE; numpy on every supported host is LE-native,
    but ``.astype('<f4')`` makes the byte order explicit so a future
    big-endian build won't silently produce a wrong file. ``ascontiguousarray``
    handles the case where a parameter view came from a slice.
    """
    import numpy as np
    arr = t.detach().cpu().numpy()
    arr = np.ascontiguousarray(arr).astype('<f4', copy=False)
    return arr.tobytes(order='C')


def save_composed(model, path: str) -> bytes:
    """Serialize a trained ``ComposedEmbedder`` to a .vse v1 file.

    Returns the file bytes (also written to ``path``). The byte return
    is convenient for round-trip tests — you can hand the bytes
    straight to a Rust parser without an extra disk round-trip.

    The model must be a ``ComposedEmbedder`` from ``model.py`` with
    either a :class:`ByteTextEncoder` or a :class:`BpeTextEncoder`,
    and either an :class:`MlpProjector` or an
    :class:`AttentionPoolMlpProjector`. The encoder type determines
    which architecture descriptor (kind 0x01 or 0x02) gets written;
    the projector type determines the projector kind byte (0x01 Mlp
    vs 0x02 AttentionMlp) and whether an extra ``attention.query``
    tensor is appended. Other shapes will raise.

    For BpeText encoders, the encoder's tokenizer is serialized to
    .bpe v1 bytes via :meth:`BpeTokenizer.to_bytes` and embedded as
    the descriptor's ``tokenizer_payload``. The Rust runtime parses
    the payload back into a :class:`BpeTokenizer` at load time, so
    the .vse file is fully self-contained — no sidecar tokenizer
    file to ship alongside.
    """
    # Late import: keep the writer's low-level path framework-agnostic.
    # save_composed is the only place that pulls in torch + the model
    # types, so casual users of write_vse can avoid the torch dep.
    from .model import (
        AttentionPoolMlpProjector,
        BpeTextEncoder,
        ByteTextEncoder,
        ComposedEmbedder,
        MlpProjector,
    )

    if not isinstance(model, ComposedEmbedder):
        raise TypeError(
            f"save_composed expects ComposedEmbedder, got {type(model).__name__}"
        )

    encoder = model.encoder
    projector = model.projector
    if not isinstance(encoder, (ByteTextEncoder, BpeTextEncoder)):
        raise TypeError(
            f"encoder must be ByteTextEncoder or BpeTextEncoder, "
            f"got {type(encoder).__name__}"
        )
    if not isinstance(projector, (MlpProjector, AttentionPoolMlpProjector)):
        raise TypeError(
            f"projector must be MlpProjector or AttentionPoolMlpProjector, "
            f"got {type(projector).__name__}"
        )

    # Build the architecture descriptor from the model's hyperparameters.
    # The conv-block spec build is identical for both encoder kinds —
    # we only differ in which encoder spec class to instantiate and
    # whether a tokenizer payload needs to come along for the ride.
    blocks = []
    for block in encoder.blocks:
        blocks.append(ConvBlockSpec(
            depthwise_has_bias=block.depthwise.bias is not None,
            pointwise_has_bias=block.pointwise.bias is not None,
        ))

    if isinstance(encoder, ByteTextEncoder):
        enc_spec: "ByteTextEncoderSpec | BpeTextEncoderSpec" = ByteTextEncoderSpec(
            hidden_dim=encoder.hidden_dim,
            max_seq_len=encoder.max_seq_len,
            kernel_size=encoder.kernel_size,
            blocks=blocks,
        )
        embedding_rows = 256
    else:
        # BpeTextEncoder. Serialize the tokenizer to .bpe v1 bytes;
        # the Rust loader parses those back via BpeTokenizer::from_bytes.
        # The vocab_size is taken from the tokenizer (the authority,
        # per the Rust BpeTextEncoder construction docs) and must agree
        # with the embedding's row count — which write_vse will check.
        tokenizer_bytes = encoder.tokenizer.to_bytes()
        enc_spec = BpeTextEncoderSpec(
            hidden_dim=encoder.hidden_dim,
            max_seq_len=encoder.max_seq_len,
            kernel_size=encoder.kernel_size,
            blocks=blocks,
            vocab_size=encoder.vocab_size,
            tokenizer_kind=TOKENIZER_KIND_EMBEDDED_BPE_V1,
            tokenizer_payload=tokenizer_bytes,
        )
        embedding_rows = encoder.vocab_size

    layers = []
    for layer in projector.layers:
        layers.append(MlpLayerSpec(
            in_features=layer.in_features,
            out_features=layer.out_features,
            has_bias=layer.bias is not None,
        ))
    proj_spec: "MlpProjectorSpec | AttentionPoolMlpProjectorSpec"
    if isinstance(projector, AttentionPoolMlpProjector):
        proj_spec = AttentionPoolMlpProjectorSpec(
            activation_kind=ACTIVATION_GELU,  # only option in v1
            layers=layers,
        )
    else:
        proj_spec = MlpProjectorSpec(
            activation_kind=ACTIVATION_GELU,  # only option in v1
            layers=layers,
        )

    # Extract tensors. Order is "architecture-required, descriptor
    # order" — required first, then per-block, then per-layer. Order
    # is not parser-significant (the parser indexes by name) but emitting
    # in a stable order produces diffable files.
    tensors: list[TensorEntry] = []

    tensors.append(TensorEntry(
        name="embedding",
        dtype=DTYPE_F32,
        shape=(embedding_rows, encoder.hidden_dim),
        data=_tensor_to_f32_bytes(encoder.embedding.weight),
    ))

    for i, block in enumerate(encoder.blocks):
        # Depthwise weight: PyTorch stores Conv1d-with-groups weights as
        # (channels, 1, kernel_size). The .vse spec wants (channels,
        # kernel_size) row-major. After .squeeze(1) the bytes are
        # identical — both are row-major, channel is the slowest axis.
        dw_weight = block.depthwise.weight.squeeze(1)
        tensors.append(TensorEntry(
            name=f"block.{i}.depthwise.weight",
            dtype=DTYPE_F32,
            shape=(encoder.hidden_dim, encoder.kernel_size),
            data=_tensor_to_f32_bytes(dw_weight),
        ))
        if block.depthwise.bias is not None:
            tensors.append(TensorEntry(
                name=f"block.{i}.depthwise.bias",
                dtype=DTYPE_F32,
                shape=(encoder.hidden_dim,),
                data=_tensor_to_f32_bytes(block.depthwise.bias),
            ))

        # Pointwise: stored as nn.Linear, weight is already (out, in)
        # row-major, no reshape needed.
        tensors.append(TensorEntry(
            name=f"block.{i}.pointwise.weight",
            dtype=DTYPE_F32,
            shape=(encoder.hidden_dim, encoder.hidden_dim),
            data=_tensor_to_f32_bytes(block.pointwise.weight),
        ))
        if block.pointwise.bias is not None:
            tensors.append(TensorEntry(
                name=f"block.{i}.pointwise.bias",
                dtype=DTYPE_F32,
                shape=(encoder.hidden_dim,),
                data=_tensor_to_f32_bytes(block.pointwise.bias),
            ))

    for i, layer in enumerate(projector.layers):
        tensors.append(TensorEntry(
            name=f"mlp.{i}.weight",
            dtype=DTYPE_F32,
            shape=(layer.out_features, layer.in_features),
            data=_tensor_to_f32_bytes(layer.weight),
        ))
        if layer.bias is not None:
            tensors.append(TensorEntry(
                name=f"mlp.{i}.bias",
                dtype=DTYPE_F32,
                shape=(layer.out_features,),
                data=_tensor_to_f32_bytes(layer.bias),
            ))

    # AttentionMlp's learned query — appended last to match the Rust
    # writer's emission order (see crates/vectorscry-embedding/src/
    # serialization/save.rs). The .vse parser indexes by name, so order
    # is not parser-significant; matching it just keeps diffs between
    # Python- and Rust-written files minimal.
    if isinstance(projector, AttentionPoolMlpProjector):
        tensors.append(TensorEntry(
            name="attention.query",
            dtype=DTYPE_F32,
            shape=(encoder.hidden_dim,),
            data=_tensor_to_f32_bytes(projector.query),
        ))

    file_bytes = write_vse(enc_spec, proj_spec, tensors)
    with open(path, 'wb') as f:
        f.write(file_bytes)
    return file_bytes