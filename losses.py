"""Distillation losses, PyTorch port of the Rust trainer's loss layer.

Replaces ``relational.rs`` and ``contrastive.rs``. Autograd handles the
backward pass — the hand-rolled gradient code on the Rust side is gone,
which is most of the simplification.

Numerical-parity note: a forward pass through these losses on the same
inputs as the Rust version should produce the same scalar loss value
to within f32 precision. We don't currently test that against the Rust
losses directly because the existing Rust trainer is being retired —
but the formulas here are intentionally identical so a regression
against the Rust path would be a code bug, not a math one.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def relational_loss(
    student: torch.Tensor,
    teacher: torch.Tensor,
    temperature: float = 0.05,
) -> torch.Tensor:
    """KL(P_t || P_s) over row-softmaxed cosine-similarity matrices.

    Mirrors ``relational.rs`` on the Rust side. Both inputs are
    ``(N, d)`` matrices of un-normalized embeddings — their ``d`` may
    differ between teacher and student. The diagonal is masked out of
    the softmax so self-pairs don't dominate the distribution.

    Returns a scalar loss; autograd handles the backward.

    Args:
        student: ``(N, d_s)`` student embeddings, already on the same
            device as teacher. Gradient flows back into these.
        teacher: ``(N, d_t)`` teacher embeddings. Detached internally
            so gradient does not flow into the teacher network even if
            the caller forgot to detach.
        temperature: softmax temperature applied to similarity rows.
            Smaller values concentrate gradient on top neighbors.
            Default ``0.05`` matches the Rust default.
    """
    n = student.shape[0]
    if n < 2:
        # No off-diagonal pairs in the batch — degenerate case. Return
        # a zero scalar that's still part of the autograd graph so a
        # downstream backward call doesn't fail.
        return student.sum() * 0.0

    teacher = teacher.detach()

    # L2-normalize each row. Rows whose norm is zero stay zero — same
    # behaviour as the Rust path's `l2_normalize_rows_capture` for
    # zero-norm rows. F.normalize's default eps (1e-12) handles this
    # by adding to the denominator rather than thresholding, which is
    # close enough to the Rust contract for any non-pathological input.
    s_norm = F.normalize(student, p=2.0, dim=1, eps=1e-12)
    t_norm = F.normalize(teacher, p=2.0, dim=1, eps=1e-12)

    # Cosine similarity matrices. Both (N, N).
    s_sim = s_norm @ s_norm.t()
    t_sim = t_norm @ t_norm.t()

    # Mask the diagonal to -inf so self-similarities drop out of the
    # softmax. Done on a fresh tensor (so we don't mutate the matmul
    # output that autograd needs the original of).
    diag_mask = torch.eye(n, dtype=torch.bool, device=student.device)
    s_sim = s_sim.masked_fill(diag_mask, float('-inf'))
    t_sim = t_sim.masked_fill(diag_mask, float('-inf'))

    # KL(P_t || P_s) = sum_j P_t * (log P_t - log P_s).
    #
    # We use log_softmax on the student and softmax on the teacher so
    # the student log-probs are computed in a numerically stable way
    # without us having to add an explicit epsilon (the Rust path's
    # LOG_EPS = 1e-12 is what's avoiding log(0) there). The teacher
    # entropy term is kept so the loss bottoms out at 0 on perfect
    # match — same as Rust.
    s_logp = F.log_softmax(s_sim / temperature, dim=1)
    t_p = F.softmax(t_sim / temperature, dim=1)
    t_logp = F.log_softmax(t_sim / temperature, dim=1)

    # Per-row KL, then mean across the batch (Rust divides by N at the
    # end of forward; matching that exactly).
    #
    # Caveat on the diagonal: log_softmax produces -inf where we masked
    # the similarity to -inf, and softmax produces exact zero. The
    # product ``0 * (-inf - -inf)`` is ``0 * nan`` = nan in IEEE
    # arithmetic. Rust dodges this with ``if pt_ij <= 0.0 { continue; }``.
    # We replace the problematic -inf log-probs with 0 before the
    # multiply — they're about to be multiplied by zero anyway, so the
    # value doesn't matter, only that it's finite. This keeps gradient
    # flow intact at the off-diagonal entries.
    s_logp_safe = s_logp.masked_fill(diag_mask, 0.0)
    t_logp_safe = t_logp.masked_fill(diag_mask, 0.0)
    per_row_kl = (t_p * (t_logp_safe - s_logp_safe)).sum(dim=1)
    return per_row_kl.mean()


def contrastive_loss(
    view_a: torch.Tensor,
    view_b: torch.Tensor,
    temperature: float = 0.05,
) -> torch.Tensor:
    """Symmetric InfoNCE for student-vs-student augmentation pairs.

    Mirrors ``contrastive.rs`` on the Rust side. Row ``i`` of ``view_a``
    and row ``i`` of ``view_b`` are the matched positive pair; every
    off-diagonal pair in the (N, N) similarity matrix is a negative.

    The teacher is not involved — this term only depends on the
    student. It exists to prevent representation collapse and to
    provide a strong gradient near degenerate solutions, complementing
    the relational term which is teacher-dependent.

    Args:
        view_a: ``(N, d_s)`` student embeddings of view-A items.
        view_b: ``(N, d_s)`` student embeddings of view-B items.
            Same row order, same student dim.
        temperature: softmax temperature. Smaller values sharpen the
            contrast and punish near-positives more aggressively.
            Default ``0.05`` matches the Rust default.

    Returns:
        Scalar loss. Range is ``[0, log N]``-ish in practice; the value
        bottoms out near zero only when each diagonal pair dominates
        its row and column.
    """
    n = view_a.shape[0]
    if n < 2:
        # No off-diagonal pairs — degenerate. Return a zero scalar
        # that's still part of the autograd graph.
        return view_a.sum() * 0.0

    a_norm = F.normalize(view_a, p=2.0, dim=1, eps=1e-12)
    b_norm = F.normalize(view_b, p=2.0, dim=1, eps=1e-12)

    # Similarity matrix scaled by inverse temperature. Not symmetric.
    logits = (a_norm @ b_norm.t()) / temperature

    # Symmetric InfoNCE = average of a→b and b→a cross-entropies, with
    # the diagonal as the target in both directions. ``cross_entropy``
    # takes raw logits and target indices, applies log_softmax along
    # dim=1, and gathers the target columns. For the b→a direction we
    # transpose the logits matrix and reuse the same target index list.
    targets = torch.arange(n, device=view_a.device)
    loss_ab = F.cross_entropy(logits, targets)
    loss_ba = F.cross_entropy(logits.t(), targets)
    return 0.5 * (loss_ab + loss_ba)