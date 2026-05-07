"""Pretrain all student archs on MS MARCO via teacher distillation.

Usage:

    # Smoke test (1k passages, 1 epoch, runs in a few minutes on CPU)
    python pretrain.py --smoke-test --out runs/smoke

    # Real run on a GPU box
    python pretrain.py --n 1000000 --epochs 3 --out runs/v1

    # Resume / re-run a specific arch (skips already-saved students)
    python pretrain.py --n 1000000 --epochs 3 --out runs/v1 --only sibyl_2layer

The script is idempotent at the per-student level: if ``runs/v1/<name>/model.vse``
exists, that arch is skipped. This makes it safe to re-run after a crash.

Outputs::

    <out>/
        <arch_name>/
            model.vse, model.pt, arch.json, tokenizer.bpe (if BPE), meta.json
        run.json    # full configuration of this invocation

Architectures match ``spec.md`` and the user's stated configs:
- rune       : byte, hidden=64, dim=64, no MLP hidden, mean pool
- auger_4096 : BPE(4096), hidden=96, dim=192, MLP=[192], attention pool, K=7, blocks=4
- auger_2048 : BPE(2048), hidden=96, dim=192, MLP=[192], attention pool, K=5, blocks=2
- sibyl_1layer : BPE(8192), hidden=128, dim=256, MLP=[256], attention pool, K=7, blocks=4
- sibyl_2layer : BPE(8192), hidden=128, dim=256, MLP=[256, 256], attention pool, K=7, blocks=4

Teachers: bge-small for rune+auger, bge-base for sibyl. The two teachers
each get encoded once and cached; the BPE tokenizers (one per unique
vocab size) are also trained once each.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

# Make `vectorscry_train` and the local `scripts.persist` importable
# regardless of where the script is invoked from. Assumes this file
# lives in <repo>/scripts/ and the package lives in <repo>/.
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))
sys.path.insert(0, str(_HERE))

from vectorscry_train import (
    RunOptions,
    StudentArch,
    distill,
    train_bpe,
)

from data import Corpus, encode_with_teacher, load_msmarco_passages
from persist import save_student


# Teacher assignment from the user's spec.
TEACHER_SMALL = "BAAI/bge-small-en-v1.5"
TEACHER_BASE = "BAAI/bge-base-en-v1.5"


@dataclass
class ArchPlan:
    """One row of the training plan: arch + teacher + BPE vocab needs."""
    name: str
    teacher: str
    bpe_vocab_size: int | None  # None = byte-level
    build: callable                # () -> StudentArch (tokenizers injected)


def make_plans(tokenizers: dict[int, "object"]) -> list[ArchPlan]:
    """Build the 5 student plans, injecting pre-trained tokenizers.

    ``tokenizers`` maps vocab_size → BpeTokenizer. The byte-level rune
    plan ignores it. Splitting "spec" from "tokenizer" lets us train
    each unique BPE once and reuse it across archs that share a vocab
    (which doesn't happen in this lineup, but the structure is the
    same it would be if it did).
    """
    plans: list[ArchPlan] = [
        ArchPlan(
            name="rune",
            teacher=TEACHER_SMALL,
            bpe_vocab_size=None,
            build=lambda: StudentArch(
                hidden_dim=64,
                student_dim=64,
                mlp_hidden=[],
                max_seq_len=256,
                kernel_size=5,
                num_blocks=2,
            ),
        ),
        ArchPlan(
            name="auger_4096",
            teacher=TEACHER_SMALL,
            bpe_vocab_size=4096,
            build=lambda: StudentArch(
                hidden_dim=96,
                student_dim=192,
                mlp_hidden=[192],
                max_seq_len=512,
                kernel_size=7,
                num_blocks=4,
                tokenizer=tokenizers[4096],
                pooling="attention",
            ),
        ),
        ArchPlan(
            name="auger_2048",
            teacher=TEACHER_SMALL,
            bpe_vocab_size=2048,
            build=lambda: StudentArch(
                hidden_dim=96,
                student_dim=192,
                mlp_hidden=[192],
                max_seq_len=256,
                kernel_size=5,
                num_blocks=2,
                tokenizer=tokenizers[2048],
                pooling="attention",
            ),
        ),
        ArchPlan(
            name="sibyl_1layer",
            teacher=TEACHER_BASE,
            bpe_vocab_size=8192,
            build=lambda: StudentArch(
                hidden_dim=128,
                student_dim=256,
                mlp_hidden=[256],
                max_seq_len=1024,
                kernel_size=7,
                num_blocks=4,
                tokenizer=tokenizers[8192],
                pooling="attention",
            ),
        ),
        ArchPlan(
            name="sibyl_2layer",
            teacher=TEACHER_BASE,
            bpe_vocab_size=8192,
            build=lambda: StudentArch(
                hidden_dim=128,
                student_dim=256,
                mlp_hidden=[256, 256],
                max_seq_len=1024,
                kernel_size=7,
                num_blocks=4,
                tokenizer=tokenizers[8192],
                pooling="attention",
            ),
        ),
    ]
    return plans


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, required=True,
                   help="Output directory for trained students.")
    p.add_argument("--cache", type=Path, default=Path("./cache"),
                   help="Cache dir for corpus + teacher embeddings.")
    p.add_argument("--n", type=int, default=1_000_000,
                   help="Subsample size for MS MARCO. None = full ~8.8M.")
    p.add_argument("--seed", type=int, default=0,
                   help="Subsample seed (also used for student init).")
    p.add_argument("--epochs", type=int, default=3,
                   help="Distillation epochs per arch.")
    p.add_argument("--batch-size", type=int, default=256,
                   help="Student training batch size.")
    p.add_argument("--lr", type=float, default=1e-3,
                   help="AdamW learning rate.")
    p.add_argument("--teacher-batch-size", type=int, default=64,
                   help="Forward batch size for teacher encoding.")
    p.add_argument("--device", type=str, default=None,
                   help="cuda / cpu. Default: auto.")
    p.add_argument("--only", type=str, nargs="+", default=None,
                   help="Subset of arch names to train (default: all).")
    p.add_argument("--force", action="store_true",
                   help="Re-train archs even if their output dir exists.")
    p.add_argument("--smoke-test", action="store_true",
                   help="Tiny config: n=1000, epochs=1, batch=32.")
    return p.parse_args()


def apply_smoke_test(args: argparse.Namespace) -> None:
    """In-place override args for a fast end-to-end smoke run."""
    args.n = 1000
    args.epochs = 1
    args.batch_size = 32
    args.teacher_batch_size = 32
    print("[smoke-test] Overriding: n=1000, epochs=1, batch_size=32")


def main() -> None:
    args = parse_args()
    if args.smoke_test:
        apply_smoke_test(args)

    args.out.mkdir(parents=True, exist_ok=True)

    # Persist the run config so a later inspection can tell what
    # produced these artifacts. Done up front so a crash mid-training
    # still leaves a record.
    (args.out / "run.json").write_text(json.dumps({
        "n": args.n,
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "device": args.device,
        "smoke_test": args.smoke_test,
    }, indent=2))

    # ----- 1. Corpus -----
    corpus = load_msmarco_passages(n=args.n, seed=args.seed, cache_dir=args.cache)

    # ----- 2. BPE tokenizers (one per unique vocab size) -----
    # Plans not yet built (they need tokenizers); collect needed sizes
    # by inspecting a placeholder plan list with empty dict.
    placeholder_plans = make_plans(tokenizers={})
    needed_vocabs = sorted({
        p.bpe_vocab_size
        for p in placeholder_plans
        if p.bpe_vocab_size is not None
    })

    tokenizers: dict[int, object] = {}
    for vocab_size in needed_vocabs:
        if args.only is not None:
            # Skip vocab sizes none of the selected archs need.
            needed_by_selected = any(
                p.bpe_vocab_size == vocab_size and p.name in args.only
                for p in placeholder_plans
            )
            if not needed_by_selected:
                continue

        print(f"[bpe] Training tokenizer vocab_size={vocab_size}...")
        t0 = time.time()
        tokenizers[vocab_size] = train_bpe(corpus.texts, vocab_size=vocab_size)
        print(f"[bpe] vocab_size={vocab_size} done in {time.time() - t0:.1f}s")

    # ----- 3. Plans (now with real tokenizers) -----
    plans = make_plans(tokenizers=tokenizers)
    if args.only is not None:
        plans = [p for p in plans if p.name in args.only]
        if not plans:
            raise SystemExit(
                f"--only filter selected no archs. Valid names: "
                f"{[p.name for p in make_plans({})]}"
            )

    # ----- 4. Teacher embeddings (cached per teacher) -----
    teacher_cache: dict[str, "object"] = {}
    needed_teachers = sorted({p.teacher for p in plans})
    for teacher_name in needed_teachers:
        teacher_cache[teacher_name] = encode_with_teacher(
            corpus=corpus,
            teacher_name=teacher_name,
            cache_dir=args.cache,
            batch_size=args.teacher_batch_size,
            device=args.device,
        )

    # ----- 5. Train each student -----
    for plan in plans:
        out_dir = args.out / plan.name
        if out_dir.exists() and not args.force:
            if (out_dir / "model.vse").exists():
                print(f"[{plan.name}] Already trained at {out_dir}, skipping. "
                      f"Pass --force to retrain.")
                continue

        print(f"[{plan.name}] Training (teacher={plan.teacher})...")
        t0 = time.time()
        arch = plan.build()
        # Same seed across archs is intentional — gives reproducibility
        # within a run. Different archs randomize differently because
        # their parameter shapes differ.
        arch.init_seed = args.seed

        teacher_vectors = teacher_cache[plan.teacher]
        run_opts = RunOptions(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=args.device,
        )
        result = distill(
            texts=corpus.texts,
            teacher_vectors=teacher_vectors,
            arch=arch,
            options=run_opts,
        )
        elapsed = time.time() - t0
        print(f"[{plan.name}] Done in {elapsed:.1f}s. "
              f"Final loss: {result.losses[-1]:.4f}")

        # result.model is the trained ComposedEmbedder (added to
        # RunResult via the train.py patch). save_student writes the
        # .vse, .pt, arch.json, tokenizer.bpe (if BPE), and meta.json.
        if result.model is None:
            raise RuntimeError(
                "distill() returned no model — did the train.py patch "
                "(adding `model` to RunResult) get applied?"
            )
        save_student(
            model=result.model,
            arch=arch,
            out_dir=out_dir,
            meta={
                "teacher": plan.teacher,
                "corpus_source": corpus.source,
                "corpus_hash": corpus.hash,
                "corpus_size": len(corpus),
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "final_loss": result.losses[-1] if result.losses else None,
                "elapsed_seconds": elapsed,
            },
        )
        print(f"[{plan.name}] Saved to {out_dir}")


if __name__ == "__main__":
    main()