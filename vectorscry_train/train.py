"""Distillation training loop. Replaces ``train_step.rs`` and
``train_run.rs`` from the Rust trainer with PyTorch.

The API is intentionally close to the existing Rust binding's
``vst.distill(...)`` so callers (notably ``distill_eval_trec.py``)
need only minimal changes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, Sequence

import numpy as np
import torch

from .losses import contrastive_loss, relational_loss
from .model import (
    AttentionPoolMlpProjector,
    BYTE_VOCAB_SIZE,
    BpeTextEncoder,
    ByteTextEncoder,
    ComposedEmbedder,
    KERNEL_SIZE,
    MlpProjector,
    NUM_BLOCKS,
)
from .vse import save_composed

if TYPE_CHECKING:
    from .augment import Augmenter
    from .bpe_tokenizer import BpeTokenizer


@dataclass
class StudentArch:
    """Architecture description for the student model.

    Mirrors the existing ``vst.StudentArch`` so the eval script can
    keep its construction calls unchanged. ``mlp_hidden`` is the list
    of hidden widths between the encoder's ``hidden_dim`` and the final
    ``student_dim``: empty means one linear layer, one entry means two
    layers, etc.

    The hardcoded ``kernel_size`` and ``num_blocks`` fields exist
    because the .vse format wire-encodes them — they're not configurable
    in v1, but we expose them so a future format revision could change
    them without rewriting this dataclass.

    Encoder kind is selected by the ``tokenizer`` field:

    - ``tokenizer is None`` (default): byte-level encoder, vocab size
      fixed at 256. The student tokenizes raw UTF-8 bytes — same as
      the existing path. No new arguments needed; existing eval
      scripts work unchanged.
    - ``tokenizer`` set to a :class:`BpeTokenizer`: BPE encoder with
      that tokenizer. The encoder's vocab size is taken from
      ``tokenizer.vocab_size()``; the embedding table sizes itself
      accordingly. The tokenizer is held by the encoder and travels
      inside the .vse file as the embedded tokenizer payload.

    The same student is sometimes evaluated against both encoder
    kinds in a sweep, so the field is optional rather than positional —
    notebooks can set up byte and BPE archs side by side with the
    minimum diff between them.

    Pooling kind is selected by the ``pooling`` field:

    - ``"mean"`` (default): masked mean pool, the original behaviour.
      Existing notebooks and eval scripts keep working unchanged.
    - ``"attention"``: learned-query attention pool then MLP. The
      learned query is initialized to zeros so step-zero attention is
      uniform — training starts from the same baseline as mean pool
      and learns away from there.
    """
    hidden_dim: int
    student_dim: int
    mlp_hidden: list[int] = field(default_factory=list)
    max_seq_len: int = 256
    init_seed: int | None = None
    kernel_size: int = KERNEL_SIZE
    num_blocks: int = NUM_BLOCKS
    # When set, build a BpeTextEncoder that holds this tokenizer.
    # When None, build a ByteTextEncoder (the original behaviour).
    tokenizer: "BpeTokenizer | None" = None
    # Pooling kind for the projector. "mean" reproduces the original
    # MlpProjector path; "attention" builds an AttentionPoolMlpProjector.
    pooling: Literal["mean", "attention"] = "mean"

    def build(self) -> ComposedEmbedder:
        """Construct a freshly-initialized ``ComposedEmbedder`` for this
        architecture."""
        if self.init_seed is not None:
            torch.manual_seed(self.init_seed)

        encoder: "ByteTextEncoder | BpeTextEncoder"
        if self.tokenizer is None:
            encoder = ByteTextEncoder(
                hidden_dim=self.hidden_dim,
                max_seq_len=self.max_seq_len,
                kernel_size=self.kernel_size,
                num_blocks=self.num_blocks,
            )
        else:
            encoder = BpeTextEncoder(
                hidden_dim=self.hidden_dim,
                max_seq_len=self.max_seq_len,
                tokenizer=self.tokenizer,
                kernel_size=self.kernel_size,
                num_blocks=self.num_blocks,
            )

        # Build the (in, out) chain for the MLP. layer_dims is a list of
        # (in_features, out_features) pairs — the projector validates the
        # chain at construction.
        widths = [self.hidden_dim, *self.mlp_hidden, self.student_dim]
        layer_dims = list(zip(widths[:-1], widths[1:]))

        projector: "MlpProjector | AttentionPoolMlpProjector"
        if self.pooling == "mean":
            projector = MlpProjector(layer_dims=layer_dims)
        elif self.pooling == "attention":
            projector = AttentionPoolMlpProjector(
                hidden_dim=self.hidden_dim,
                layer_dims=layer_dims,
            )
        else:
            raise ValueError(
                f"unknown pooling kind: {self.pooling!r} "
                f"(expected 'mean' or 'attention')"
            )

        return ComposedEmbedder(encoder, projector)


@dataclass
class RunOptions:
    """Training-run hyperparameters.

    Mirrors the existing ``vst.RunOptions``. The only intentional
    deviation: ``shuffle_seed`` is accepted but unused in this port —
    we don't bother seeding the per-epoch shuffle. If you need
    deterministic behaviour, set ``init_seed`` on ``StudentArch`` and
    accept that batch order will vary; for distillation runs in
    practice this doesn't materially affect final accuracy.

    ``temperature`` and ``relational_weight`` mirror the Rust defaults.
    ``contrastive_weight = 0`` disables the contrastive term entirely.

    ``augmenter`` lets ``distill`` build the paired ``[view_a..., view_b...]``
    corpus internally from an unpaired source corpus. When set, the
    caller passes the original ``N`` texts and ``(N, d_t)`` teacher
    matrix; ``distill`` doubles them via ``augment.build_paired_corpus``
    before training. When unset, the caller is responsible for the
    paired layout (the original Rust-trainer convention). Setting
    ``augmenter`` without ``contrastive_weight > 0`` is allowed but
    pointless — the relational loss alone doesn't benefit from view
    pairs — so we warn rather than error.
    """
    epochs: int = 1
    batch_size: int = 32
    learning_rate: float = 1e-3
    shuffle_seed: int | None = None
    temperature: float = 0.05
    relational_weight: float = 1.0
    contrastive_weight: float = 0.0
    contrastive_temperature: float = 0.05
    weight_decay: float = 0.01
    device: str | None = None  # None = auto
    # None = caller pre-builds paired corpus (or doesn't use contrastive).
    # Set = distill() builds it from the unpaired source corpus.
    augmenter: "Augmenter | None" = None


@dataclass
class RunResult:
    """Return value of ``distill``.

    ``bytes`` holds the .vse file contents. The loss vectors record
    per-step values (already weighted by their respective term weights,
    so summing reproduces what the optimizer actually saw).

    ``model`` is the trained ``ComposedEmbedder`` itself, in eval mode,
    on whatever device training used. Callers that want to fine-tune,
    save a torch state_dict, or run further inference without parsing
    the .vse bytes back can use this directly. Callers that only want
    the deployment artifact can keep using ``bytes``.
    """
    bytes: bytes
    losses: list[float]
    relational_losses: list[float]
    contrastive_losses: list[float]
    model: "ComposedEmbedder | None" = None


def _pick_device(requested: str | None) -> torch.device:
    if requested is not None:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def distill(
    texts: Sequence[str],
    teacher_vectors: np.ndarray,
    arch: StudentArch,
    options: RunOptions,
    save_path: str | None = None,
) -> RunResult:
    """Distill a student against teacher embeddings.

    Args:
        texts: training corpus. When ``options.augmenter`` is set,
            this is the unpaired source corpus of length ``N``;
            ``distill`` builds the paired layout internally. When
            ``options.augmenter`` is ``None`` and
            ``options.contrastive_weight`` is nonzero, the caller must
            supply the paired layout directly: length ``2N`` arranged
            as ``[view_a_0, ..., view_a_{N-1}, view_b_0, ..., view_b_{N-1}]``
            — the convention the Rust trainer used.
        teacher_vectors: numpy array of teacher embeddings, in the
            same row order as ``texts``. Need not be normalized —
            the relational loss applies its own normalization.
            Duplicated automatically when ``options.augmenter`` is set.
        arch: student architecture.
        options: training hyperparameters.
        save_path: if given, write the .vse file to this path in
            addition to returning the bytes. Convenience for the
            common case.

    Returns:
        ``RunResult`` with the trained model's .vse bytes and per-step
        loss values.
    """
    if len(texts) != teacher_vectors.shape[0]:
        raise ValueError(
            f"texts length ({len(texts)}) must equal teacher_vectors row "
            f"count ({teacher_vectors.shape[0]})"
        )
    if options.batch_size <= 0:
        raise ValueError(f"batch_size must be > 0 (got {options.batch_size})")

    # If an augmenter is configured, double the corpus into the
    # [view_a..., view_b...] layout the paired training path expects.
    # Done before the even-length check below so the doubled corpus
    # passes that check by construction.
    #
    # We warn (not error) when augmenter is set but contrastive_weight
    # is zero: the resulting paired corpus is harmless to the relational
    # path (it just sees 2N rows where each pair shares a teacher
    # target) but the augmentation work is wasted. A user who hits
    # this is probably mid-experiment and we don't want to crash them.
    if options.augmenter is not None:
        if options.contrastive_weight == 0.0:
            import warnings
            warnings.warn(
                "augmenter is set but contrastive_weight is 0; the "
                "augmented view pairs will not contribute to any loss "
                "term. Set contrastive_weight > 0 to use them.",
                stacklevel=2,
            )
        # Late import — keeps train.py importable without nlpaug
        # installed in the no-augmenter case.
        from .augment import build_paired_corpus
        texts, teacher_vectors = build_paired_corpus(
            list(texts), teacher_vectors, options.augmenter,
        )

    contrastive_active = options.contrastive_weight != 0.0
    if contrastive_active and len(texts) % 2 != 0:
        raise ValueError(
            "contrastive_weight > 0 requires an even-length corpus "
            "(top-half / bottom-half view pairing)"
        )

    device = _pick_device(options.device)
    model = arch.build().to(device)
    model.train()

    teacher_t = torch.from_numpy(np.ascontiguousarray(teacher_vectors, dtype=np.float32)).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=options.learning_rate,
        weight_decay=options.weight_decay,
    )

    n = len(texts)
    losses: list[float] = []
    relational_losses: list[float] = []
    contrastive_losses: list[float] = []

    for _epoch in range(options.epochs):
        if contrastive_active:
            _train_one_epoch_paired(
                model, list(texts), teacher_t, optimizer, options, device,
                losses, relational_losses, contrastive_losses,
            )
        else:
            _train_one_epoch_unpaired(
                model, list(texts), teacher_t, optimizer, options, device,
                losses, relational_losses, contrastive_losses,
            )

    # Final save. eval() to drop any train-time behaviour (we don't have
    # any in this model — no dropout, no batchnorm — but it's the right
    # idiom and costs nothing).
    model.eval()

    if save_path is not None:
        file_bytes = save_composed(model, save_path)
    else:
        # save_composed always writes to disk; use a tempfile when the
        # caller didn't ask for one. The bytes are the source of truth
        # — eval_trec writes them itself via result.bytes.
        import tempfile, os
        with tempfile.NamedTemporaryFile(suffix=".vse", delete=False) as f:
            tmp_path = f.name
        try:
            file_bytes = save_composed(model, tmp_path)
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    return RunResult(
        bytes=file_bytes,
        losses=losses,
        relational_losses=relational_losses,
        contrastive_losses=contrastive_losses,
        model=model,
    )


def _train_one_epoch_unpaired(
    model: ComposedEmbedder,
    texts: list[str],
    teacher: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    options: RunOptions,
    device: torch.device,
    losses: list[float],
    relational_losses: list[float],
    contrastive_losses: list[float],
) -> None:
    """One pass over the corpus with relational loss only.

    The corpus is shuffled per-epoch (the Rust path used a seeded
    shuffle; here we use torch's default RNG). Each batch runs forward
    through the model, computes relational loss against the teacher
    rows, and steps AdamW.
    """
    n = len(texts)
    perm = torch.randperm(n).tolist()

    for start in range(0, n, options.batch_size):
        idx = perm[start : start + options.batch_size]
        if len(idx) < 2:
            # Relational loss is degenerate at N=1 (returns zero with no
            # gradient). Skipping saves a wasted step.
            continue

        batch_texts = [texts[i] for i in idx]
        batch_teacher = teacher[idx]

        # Encoder picks the right tokenization (byte vs BPE) based
        # on its type. Same call site for both paths — see
        # ByteTextEncoder.tokenize_batch and BpeTextEncoder.tokenize_batch.
        input_ids, attention_mask = model.encoder.tokenize_batch(
            batch_texts, device=device,
        )

        optimizer.zero_grad()
        student = model(input_ids, attention_mask)
        rel = relational_loss(student, batch_teacher, temperature=options.temperature)
        loss = options.relational_weight * rel
        loss.backward()
        optimizer.step()

        # Record per-step values. Already weighted, matching the Rust
        # convention so summing reproduces the scalar that drove
        # gradients.
        losses.append(loss.item())
        relational_losses.append(loss.item())
        contrastive_losses.append(0.0)


def _train_one_epoch_paired(
    model: ComposedEmbedder,
    texts: list[str],
    teacher: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    options: RunOptions,
    device: torch.device,
    losses: list[float],
    relational_losses: list[float],
    contrastive_losses: list[float],
) -> None:
    """One pass over the corpus with relational + contrastive losses.

    Texts are laid out as ``[view_a_0..N/2-1, view_b_0..N/2-1]``. We
    shuffle pair indices and within each batch materialize a
    ``2 * batch_size`` row layout matching the Rust trainer's
    convention: top half is view_a's, bottom half is matching view_b's,
    same ordering. The relational term sees the full ``2*B``
    embeddings; the contrastive term sees the two halves as positive
    pairs.
    """
    half = len(texts) // 2
    pair_perm = torch.randperm(half).tolist()

    for start in range(0, half, options.batch_size):
        chunk = pair_perm[start : start + options.batch_size]
        k = len(chunk)
        if k < 2:
            # Both losses are degenerate at N<2 (paired k=1 means
            # batch size 2 of view_a + view_b, which the relational
            # term tolerates but the contrastive doesn't gain from).
            # Skipping.
            continue

        # Build the 2k-row batch: top half view_a's, bottom half view_b's.
        a_indices = chunk
        b_indices = [p + half for p in chunk]
        ordered_indices = a_indices + b_indices
        batch_texts = [texts[i] for i in ordered_indices]
        batch_teacher = teacher[ordered_indices]

        # Encoder picks the right tokenization — same dispatch as
        # the unpaired path.
        input_ids, attention_mask = model.encoder.tokenize_batch(
            batch_texts, device=device,
        )

        optimizer.zero_grad()
        student = model(input_ids, attention_mask)              # (2k, d_s)

        rel = relational_loss(student, batch_teacher, temperature=options.temperature)
        view_a_emb = student[:k]
        view_b_emb = student[k:]
        con = contrastive_loss(
            view_a_emb, view_b_emb, temperature=options.contrastive_temperature,
        )

        rel_w = options.relational_weight * rel
        con_w = options.contrastive_weight * con
        loss = rel_w + con_w
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        relational_losses.append(rel_w.item())
        contrastive_losses.append(con_w.item())