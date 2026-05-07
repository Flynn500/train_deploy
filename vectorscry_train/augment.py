"""Text augmentation for contrastive distillation.

Produces the ``[view_a_0, ..., view_a_{N-1}, view_b_0, ..., view_b_{N-1}]``
corpus layout that ``train._train_one_epoch_paired`` expects. The
training-loop side of the convention is documented there; this module
is the producer side.

Design choices that matter:

- ``Augmenter`` is a small protocol, not a class hierarchy. Anything
  callable with the right signature works — including a bare lambda
  for tests. The concrete wrappers below exist because ``nlpaug``'s
  augmenters have stateful RNGs and per-call options that benefit
  from a thin adapter, not because a class is required.
- Augmentation runs once, eagerly, before training starts. We don't
  re-augment per epoch: the SimCSE-style contrastive setup wants
  *some* view diversity but not so much that the student chases a
  moving target. Re-augmenting per epoch is a knob we can add later
  if a run benefits from it.
- The teacher embeddings are duplicated, not re-computed on the
  augmented views. Both views share the teacher target of the
  original text. This matches the relational-loss intent: we want
  the *student's* representation of (view_a, view_b) to mirror the
  teacher's representation of the original, while the contrastive
  term separately pulls (view_a, view_b) toward each other.
"""

from __future__ import annotations

from typing import Callable, Protocol

import numpy as np


class Augmenter(Protocol):
    """A pure function from a string to a (probably modified) string.

    Implementations should be deterministic given a seed they manage
    internally — ``build_paired_corpus`` does not pass a seed in. If
    you need reproducibility, seed the augmenter at construction time.

    Returning the input unchanged is allowed; returning an empty
    string is *not* — see ``build_paired_corpus`` for why.
    """

    def __call__(self, text: str) -> str: ...


def build_paired_corpus(
    texts: list[str],
    teacher_vectors: np.ndarray,
    augmenter: Augmenter,
) -> tuple[list[str], np.ndarray]:
    """Produce the paired ``[view_a..., view_b...]`` corpus and the
    matching duplicated teacher matrix.

    For each input text ``t`` with teacher vector ``v``:

    - ``view_a`` is ``augmenter(t)`` (one augmented sample)
    - ``view_b`` is ``augmenter(t)`` (a *second* augmented sample,
      independently drawn — same source, different noise)
    - the teacher row ``v`` appears twice in the output: once at
      position ``i`` (for view_a) and once at position ``i + N``
      (for view_b)

    This mirrors SimCLR/SimCSE: both views are augmentations of the
    same source, neither is the original. If you want one view to be
    the original, wrap your augmenter in an ``IdentityAugmenter``
    composition (not provided here — add when needed).

    Empty-string outputs from the augmenter are replaced with the
    original text. Without this, ``ComposedEmbedder`` would emit a
    zero embedding for that view, which makes the contrastive loss's
    diagonal logit zero regardless of the other view — silently
    degrading the gradient for that pair. The fallback isn't perfect
    (a pair where view_a falls back to the original and view_b doesn't
    is mildly informative; a pair where both fall back is just a
    duplicate row), but it's better than zero embeddings.

    Args:
        texts: ``N`` source strings.
        teacher_vectors: ``(N, d_t)`` teacher embeddings, row-aligned
            with ``texts``.
        augmenter: callable producing one augmented view per call.
            Called ``2N`` times total.

    Returns:
        ``(paired_texts, paired_teacher)`` where:
        - ``paired_texts`` has length ``2N``, layout ``[view_a..., view_b...]``
        - ``paired_teacher`` has shape ``(2N, d_t)``, layout
          ``[teacher..., teacher...]`` (the original matrix, vertically
          stacked with itself)
    """
    if len(texts) != teacher_vectors.shape[0]:
        raise ValueError(
            f"texts length ({len(texts)}) must equal teacher_vectors row "
            f"count ({teacher_vectors.shape[0]})"
        )

    view_a = [_augment_with_fallback(augmenter, t) for t in texts]
    view_b = [_augment_with_fallback(augmenter, t) for t in texts]

    paired_texts = view_a + view_b
    paired_teacher = np.concatenate([teacher_vectors, teacher_vectors], axis=0)
    return paired_texts, paired_teacher


def _augment_with_fallback(augmenter: Augmenter, text: str) -> str:
    """Call the augmenter; fall back to the original on empty output.

    See ``build_paired_corpus`` docstring for the rationale. Also
    falls back if the augmenter's output is whitespace-only — same
    failure mode (zero-byte UTF-8 → all-pad row → zero embedding) once
    the encoder strips nothing but treats every byte as content. We
    don't strip in the encoder, but a whitespace-only string still
    encodes to a near-degenerate sequence that the conv stack maps to
    near-zero, which is bad for the same reasons.
    """
    out = augmenter(text)
    if not out or not out.strip():
        return text
    return out


# ---------------------------------------------------------------------------
# nlpaug wrappers
# ---------------------------------------------------------------------------
#
# Late import: nlpaug pulls in numpy/nltk transitively which the rest
# of this package already needs, but importing it at module load would
# force every consumer of `vectorscry_train` to have nlpaug installed
# even when they never augment. We import inside __init__ so the
# dependency is only required when an augmenter is actually built.


class CharAugmenter:
    """Character-level augmenter wrapping ``nlpaug.augmenter.char.RandomCharAug``.

    Operates at the Unicode character level: insert, substitute, swap,
    or delete characters within selected words. Choice of action
    matters for byte-level encoders — ``swap`` and ``substitute`` keep
    sequence length stable, ``insert`` and ``delete`` change it. For
    distillation against a teacher with fixed embeddings, length-stable
    actions tend to produce smaller perturbations and easier positive
    pairs; ``delete`` produces the strongest augmentation but risks
    semantic drift on short inputs.

    The default action is ``substitute`` because it's the SimCSE-paper
    baseline for character augmentation and gives a sensible
    starting point. Tune via the ``action`` arg.

    Determinism: ``nlpaug`` reads a seed at construction time. Pass
    ``seed`` here to make the augmenter reproducible. Calls to the
    augmenter advance an internal RNG, so two consecutive calls on the
    same input produce different outputs (which is what
    ``build_paired_corpus`` relies on for view diversity).
    """

    def __init__(
        self,
        action: str = "substitute",
        aug_char_p: float = 0.1,
        aug_word_p: float = 0.3,
        seed: int | None = None,
    ) -> None:
        # Late import — see module-level note above.
        try:
            import nlpaug.augmenter.char as nac
        except ImportError as e:
            raise ImportError(
                "CharAugmenter requires nlpaug. Install with: pip install nlpaug"
            ) from e

        if action not in ("insert", "substitute", "swap", "delete"):
            raise ValueError(
                f"action must be one of insert/substitute/swap/delete "
                f"(got {action!r})"
            )
        if not 0.0 <= aug_char_p <= 1.0:
            raise ValueError(f"aug_char_p must be in [0, 1] (got {aug_char_p})")
        if not 0.0 <= aug_word_p <= 1.0:
            raise ValueError(f"aug_word_p must be in [0, 1] (got {aug_word_p})")

        # nlpaug's RandomCharAug takes aug_char_p (fraction of chars
        # within a selected word to alter) and aug_word_p (fraction of
        # words in the input to select). Defaults here are gentle: 30%
        # of words, 10% of chars within those words.
        self._aug = nac.RandomCharAug(
            action=action,
            aug_char_p=aug_char_p,
            aug_word_p=aug_word_p,
        )
        if seed is not None:
            # nlpaug exposes its RNG via .seed() on the underlying
            # generator. We set it once at construction; subsequent
            # calls advance from that seed deterministically.
            import random
            random.seed(seed)
            np.random.seed(seed)

    def __call__(self, text: str) -> str:
        # nlpaug returns either a string or a single-element list
        # depending on version. Normalize to a string.
        out = self._aug.augment(text)
        if isinstance(out, list):
            out = out[0] if out else ""
        return out # type: ignore

