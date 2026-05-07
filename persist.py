"""Save/load helpers for trained students.

Each saved student is a directory:

    <name>/
        model.vse          # deployment-ready .vse v1
        model.pt           # PyTorch state_dict for fine-tuning
        arch.json          # StudentArch hyperparams (no tokenizer)
        tokenizer.bpe      # raw .bpe v1 bytes (BPE archs only)
        meta.json          # training provenance (teacher, corpus, etc.)

The split exists because:
- .vse is what the Rust runtime loads for inference
- .pt + arch.json + tokenizer.bpe is what *we* load for fine-tuning,
  because rebuilding a StudentArch from a .vse alone would require
  parsing the format twice (once to load, once to extract metadata)
  and is a yak-shave we don't need
- meta.json captures "what was this trained on" so a later benchmark
  run can sanity-check it's pairing the right student with the right
  evaluation regime
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from vectorscry_train import (
    BpeTokenizer,
    ComposedEmbedder,
    StudentArch,
    save_composed,
)


def save_student(
    model: ComposedEmbedder,
    arch: StudentArch,
    out_dir: Path,
    meta: dict[str, Any],
) -> None:
    """Persist a trained student to ``out_dir`` (created if needed).

    The model's tokenizer (if any) is serialized to ``tokenizer.bpe``
    via :meth:`BpeTokenizer.to_bytes` — same payload that's embedded
    in the .vse file, written out separately so :func:`load_student`
    can rebuild the StudentArch without parsing the .vse.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # .vse — deployment artifact. save_composed also writes to disk
    # but we want it under our chosen filename.
    save_composed(model, str(out_dir / "model.vse"))

    # .pt — state_dict only. Loading requires rebuilding the arch
    # first, which is why arch.json + tokenizer.bpe ride alongside.
    torch.save(model.state_dict(), out_dir / "model.pt")

    # arch.json — every StudentArch field except the tokenizer (which
    # is unpickleable JSON-wise and lives in tokenizer.bpe instead).
    arch_dict = {
        "hidden_dim": arch.hidden_dim,
        "student_dim": arch.student_dim,
        "mlp_hidden": list(arch.mlp_hidden),
        "max_seq_len": arch.max_seq_len,
        "kernel_size": arch.kernel_size,
        "num_blocks": arch.num_blocks,
        "pooling": arch.pooling,
        "has_tokenizer": arch.tokenizer is not None,
    }
    (out_dir / "arch.json").write_text(json.dumps(arch_dict, indent=2))

    if arch.tokenizer is not None:
        (out_dir / "tokenizer.bpe").write_bytes(arch.tokenizer.to_bytes())

    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, default=str))


def load_student(
    in_dir: Path,
    device: torch.device | str = "cpu",
) -> tuple[ComposedEmbedder, StudentArch, dict[str, Any]]:
    """Inverse of :func:`save_student`.

    Returns ``(model, arch, meta)``. The arch is rebuilt with the
    saved tokenizer (if any) so it's ready for further training:
    a caller can grab ``model.parameters()`` and feed them into a
    fresh AdamW.
    """
    arch_dict = json.loads((in_dir / "arch.json").read_text())
    meta = json.loads((in_dir / "meta.json").read_text())

    tokenizer: BpeTokenizer | None = None
    if arch_dict["has_tokenizer"]:
        tok_bytes = (in_dir / "tokenizer.bpe").read_bytes()
        tokenizer = _load_bpe_from_bytes(tok_bytes)

    arch = StudentArch(
        hidden_dim=arch_dict["hidden_dim"],
        student_dim=arch_dict["student_dim"],
        mlp_hidden=arch_dict["mlp_hidden"],
        max_seq_len=arch_dict["max_seq_len"],
        kernel_size=arch_dict["kernel_size"],
        num_blocks=arch_dict["num_blocks"],
        pooling=arch_dict["pooling"],
        tokenizer=tokenizer,
    )
    model = arch.build()
    model.load_state_dict(torch.load(in_dir / "model.pt", map_location=device))
    model.to(device)
    return model, arch, meta


def _load_bpe_from_bytes(payload: bytes) -> BpeTokenizer:
    """Parse a .bpe v1 payload back into a BpeTokenizer.

    The Python codebase has a writer (``write_bpe``) but not a reader —
    the Rust side reads, the Python side writes. We need a reader here
    for the fine-tune reload path. Mirrors ``BpeTokenizer::from_bytes``
    on the Rust side.
    """
    import struct
    from vectorscry_train import (
        Merge,
        NUM_BASE_BYTE_TOKENS,
        PRE_TOKENIZER_KIND_WHITESPACE,
    )

    if len(payload) < 16 or payload[:4] != b"VSBP":
        raise ValueError("not a .bpe v1 file (bad magic or too short)")
    (version,) = struct.unpack_from("<H", payload, 4)
    if version != 1:
        raise ValueError(f"unsupported .bpe version {version}")
    pre_kind = payload[6]
    if pre_kind != PRE_TOKENIZER_KIND_WHITESPACE:
        raise ValueError(f"unsupported pre_tokenizer_kind {pre_kind:#x}")
    # payload[7] is reserved
    (vocab_size,) = struct.unpack_from("<I", payload, 8)
    (num_merges,) = struct.unpack_from("<I", payload, 12)

    cursor = 16
    vocab: list[bytes] = []
    for _ in range(vocab_size):
        (n,) = struct.unpack_from("<H", payload, cursor)
        cursor += 2
        vocab.append(payload[cursor : cursor + n])
        cursor += n

    merges: list[Merge] = []
    for _ in range(num_merges):
        left, right, result = struct.unpack_from("<III", payload, cursor)
        cursor += 12
        merges.append(Merge(left=left, right=right, result=result))

    return BpeTokenizer(
        vocab=vocab,
        merges=merges,
        pre_tokenizer_kind=PRE_TOKENIZER_KIND_WHITESPACE,
    )
