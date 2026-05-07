"""MS MARCO passage loader + cached teacher embedding.

Two responsibilities:

1. Pull MS MARCO passages from HuggingFace (`Tevatron/msmarco-passage-corpus`),
   optionally subsampled, and expose them as a plain ``list[str]``.
2. Encode that list with a sentence-transformers teacher model and
   cache the result to disk, keyed by ``(teacher_name, corpus_hash)``,
   so a re-run of the same teacher/corpus combo is instant.

The cache lives at::

    <cache_dir>/
        corpus-<hash>.txt        # one passage per line, newline-escaped
        corpus-<hash>.meta.json  # source spec for reproducibility
        teacher-<teacher_slug>-<corpus_hash>.npy  # (N, d) float32

``corpus_hash`` is a content hash of the passage list, so the same
subsample seed reliably hits the same cache entry. Teachers are
keyed by both name and corpus hash because the same teacher run on
a different corpus subsample is a different artifact.

Resumability: teacher encoding is the longest step. We checkpoint
every ``checkpoint_every`` batches by writing a partial .npy + a
``.progress`` file with the row count. On restart we mmap the
partial, resume from ``progress.row_count``, and finalize when done.
This makes a 4-hour bge-base encode survive a spot-instance preemption.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


# Keep the slug short, filesystem-safe, and reversible-ish so a
# directory listing is human-readable.
def _teacher_slug(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]", "_", name)


def _corpus_hash(texts: list[str]) -> str:
    """Stable 12-hex-char hash of a passage list.

    SHA-256 of the newline-joined corpus, truncated. Two runs that
    produce the same passage list (same source + same subsample seed)
    will hit the same cache entry. Two runs that differ in even one
    passage will not — which is what we want, since a different
    subsample needs its own teacher embedding.
    """
    h = hashlib.sha256()
    for t in texts:
        h.update(t.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()[:12]


@dataclass
class Corpus:
    """A loaded passage corpus + its content hash."""
    texts: list[str]
    source: str       # human description e.g. "msmarco-passage:1000000@seed=0"
    hash: str

    def __len__(self) -> int:
        return len(self.texts)


def load_msmarco_passages(
    n: int | None = None,
    seed: int = 0,
    cache_dir: Path | str = "./cache",
) -> Corpus:
    """Load MS MARCO passages, optionally subsampled to ``n``.

    Uses the ``Tevatron/msmarco-passage-corpus`` HuggingFace dataset
    (the cleanest passage-only mirror — official MS MARCO is split
    across multiple files and includes IDs we don't need). Falls back
    to ``BeIR/msmarco`` corpus split if Tevatron is unavailable.

    Args:
        n: subsample size. ``None`` keeps all ~8.8M passages.
        seed: subsample RNG seed. Same seed + same ``n`` = same
            corpus hash = cache hit on the teacher embeddings.
        cache_dir: where to materialize the corpus text file. The
            HuggingFace dataset itself is cached in ``~/.cache/huggingface``
            independently; this is just our normalized passage list.

    Returns:
        A :class:`Corpus` with the passages and a stable hash.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Construct a source string and check the corpus-hash cache before
    # touching HF at all. If the same (n, seed) combo was loaded
    # before, the corpus file already exists; we just read it back.
    # Note: the hash cache is keyed by content, not by (n, seed) —
    # but we don't know the content until we load it. So we store a
    # tiny "spec → hash" lookup file.
    source = f"msmarco-passage:n={n}:seed={seed}"
    spec_path = cache_dir / "corpus-spec-lookup.json"
    spec_lookup: dict[str, str] = {}
    if spec_path.exists():
        spec_lookup = json.loads(spec_path.read_text())

    if source in spec_lookup:
        cached_hash = spec_lookup[source]
        corpus_path = cache_dir / f"corpus-{cached_hash}.txt"
        if corpus_path.exists():
            texts = _read_corpus_file(corpus_path)
            return Corpus(texts=texts, source=source, hash=cached_hash)
        # Stale lookup entry — corpus file was deleted. Fall through
        # to re-load from HF.

    print(f"[corpus] Loading {source} from HuggingFace...")
    texts = _fetch_msmarco_from_hf(n=n, seed=seed)

    h = _corpus_hash(texts)
    corpus_path = cache_dir / f"corpus-{h}.txt"
    _write_corpus_file(corpus_path, texts)
    (cache_dir / f"corpus-{h}.meta.json").write_text(
        json.dumps({"source": source, "size": len(texts)}, indent=2)
    )
    spec_lookup[source] = h
    spec_path.write_text(json.dumps(spec_lookup, indent=2))

    print(f"[corpus] {len(texts):,} passages loaded, hash={h}")
    return Corpus(texts=texts, source=source, hash=h)


def _fetch_msmarco_from_hf(n: int | None, seed: int) -> list[str]:
    """Fetch MS MARCO passages via HuggingFace datasets.

    Tries Tevatron's mirror first (single ``text`` column, easy);
    falls back to BeIR's mirror (``title`` + ``text``, we concatenate
    with a space the way most retrieval systems do).
    """
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError(
            "msmarco loading needs `datasets`. Install with: "
            "pip install datasets"
        ) from e

    try:
        ds = load_dataset("Tevatron/msmarco-passage-corpus", split="train")
        col = "text"
    except Exception:
        # Fallback. BeIR's corpus has 'title' (often empty) and 'text'.
        ds = load_dataset("BeIR/msmarco", "corpus", split="corpus")
        col = "_beir"

    if n is not None and n < len(ds):
        # Deterministic subsample. ``ds.shuffle(seed=...).select(range(n))``
        # produces the same sample for the same seed/n.
        ds = ds.shuffle(seed=seed).select(range(n))

    if col == "text":
        return list(ds["text"])
    else:
        # BeIR concatenation: "title text" with a space, stripped.
        # An empty title produces just the text; we don't want a
        # leading space leaking in.
        out: list[str] = []
        for row in ds:
            title = (row.get("title") or "").strip()
            text = (row.get("text") or "").strip()
            out.append(f"{title} {text}".strip() if title else text)
        return out


def _write_corpus_file(path: Path, texts: list[str]) -> None:
    """Write one passage per line, escaping embedded newlines.

    MS MARCO passages occasionally contain literal newlines. We
    encode them as ``\\n`` so the line-per-passage invariant holds.
    """
    with open(path, "w", encoding="utf-8") as f:
        for t in texts:
            f.write(t.replace("\\", "\\\\").replace("\n", "\\n"))
            f.write("\n")


def _read_corpus_file(path: Path) -> list[str]:
    """Inverse of ``_write_corpus_file``."""
    out: list[str] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            # Reverse the escape: \\n → \n, \\\\ → \\.
            # Two-pass so we don't double-decode an escaped backslash.
            out.append(_unescape(line))
    return out


def _unescape(s: str) -> str:
    out: list[str] = []
    i = 0
    while i < len(s):
        if s[i] == "\\" and i + 1 < len(s):
            nxt = s[i + 1]
            if nxt == "n":
                out.append("\n")
                i += 2
                continue
            elif nxt == "\\":
                out.append("\\")
                i += 2
                continue
        out.append(s[i])
        i += 1
    return "".join(out)


def encode_with_teacher(
    corpus: Corpus,
    teacher_name: str,
    cache_dir: Path | str = "./cache",
    batch_size: int = 64,
    device: str | None = None,
    checkpoint_every: int = 100,
) -> np.ndarray:
    """Encode a corpus with a sentence-transformers teacher, cached.

    Returns ``(N, d_teacher)`` float32 array. On cache hit, no model
    is loaded — we mmap the .npy and copy it into RAM.

    Resumability: writes a ``.progress`` file alongside the partial
    .npy every ``checkpoint_every`` batches. On restart, picks up
    from the recorded row count. The partial .npy is pre-allocated
    at full size so we can write into it row-by-row without resizing.

    Args:
        corpus: the loaded corpus.
        teacher_name: HuggingFace model id, e.g.
            ``"BAAI/bge-small-en-v1.5"``.
        cache_dir: same directory as the corpus cache.
        batch_size: forward batch size on the teacher.
        device: ``"cuda"`` / ``"cpu"`` / ``None`` (auto).
        checkpoint_every: how often to flush progress to disk
            (in batches).

    Returns:
        ``(N, d_teacher)`` float32 numpy array.
    """
    cache_dir = Path(cache_dir)
    slug = _teacher_slug(teacher_name)
    final_path = cache_dir / f"teacher-{slug}-{corpus.hash}.npy"
    progress_path = cache_dir / f"teacher-{slug}-{corpus.hash}.progress"

    if final_path.exists() and not progress_path.exists():
        print(f"[teacher] Cache hit: {final_path.name}")
        return np.load(final_path)

    # Either no cache, or a partial encode that didn't finish. Load
    # the teacher and (re)start.
    from sentence_transformers import SentenceTransformer
    import torch as _torch

    if device is None:
        device = "cuda" if _torch.cuda.is_available() else "cpu"

    print(f"[teacher] Loading {teacher_name} on {device}...")
    model = SentenceTransformer(teacher_name, device=device)
    d_teacher = model.get_sentence_embedding_dimension()
    n = len(corpus.texts)

    # Pre-allocate / load existing partial. mmap='r+' lets us write
    # into the existing file in place; for a fresh start, np.lib.format
    # writes a full-size header and pre-allocates on disk.
    if final_path.exists() and progress_path.exists():
        progress = json.loads(progress_path.read_text())
        start_row = int(progress["row_count"])
        if progress.get("d_teacher") != d_teacher or progress.get("n") != n:
            # Shape mismatch — partial doesn't match this run.
            # Discard and start over.
            print(f"[teacher] Partial cache shape mismatch, restarting.")
            final_path.unlink()
            progress_path.unlink()
            start_row = 0
        else:
            print(f"[teacher] Resuming from row {start_row:,} / {n:,}")
    else:
        start_row = 0

    if start_row == 0:
        # Pre-allocate the .npy on disk. np.lib.format.open_memmap
        # creates a real .npy file with header + body; no special
        # finalize step needed when we're done.
        out = np.lib.format.open_memmap(
            str(final_path),
            mode="w+",
            dtype=np.float32,
            shape=(n, d_teacher),
        )
    else:
        out = np.lib.format.open_memmap(str(final_path), mode="r+")
        if out.shape != (n, d_teacher):
            raise RuntimeError(
                f"partial teacher cache has shape {out.shape}, "
                f"expected {(n, d_teacher)}"
            )

    # Encode in batches. sentence-transformers' .encode() handles
    # truncation/padding internally; we don't need to pre-pad.
    batches_done = 0
    for batch_start in range(start_row, n, batch_size):
        batch_end = min(batch_start + batch_size, n)
        batch_texts = corpus.texts[batch_start:batch_end]
        embs = model.encode(
            batch_texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=False,  # losses normalize themselves
        )
        out[batch_start:batch_end] = embs.astype(np.float32, copy=False)
        batches_done += 1

        if batches_done % checkpoint_every == 0:
            out.flush()
            progress_path.write_text(json.dumps({
                "row_count": batch_end,
                "n": n,
                "d_teacher": d_teacher,
            }))
            pct = 100 * batch_end / n
            print(f"[teacher] {batch_end:,}/{n:,} ({pct:.1f}%)")

    out.flush()
    # Done — drop the progress file; presence-without-progress means
    # "fully encoded" on the next run.
    if progress_path.exists():
        progress_path.unlink()
    print(f"[teacher] Done: {final_path.name}")
    # Return as a regular (non-mmap) array so the caller can move it
    # to GPU without surprises. Copy is the price.
    return np.array(out)
