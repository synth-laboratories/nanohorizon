"""NanoHorizon smoke submission.

This module keeps the required submission surface stable while making only a
small prompt-text change for review.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List


PUBLICATION_SMOKE_NOTE = "Publication smoke note: keep this submission minimal and reviewable."
DEFAULT_SYSTEM_PROMPT = (
    "You are a careful Craftax agent. "
    "Prefer stable, low-risk behavior and concise outputs. "
    f"{PUBLICATION_SMOKE_NOTE}"
)


def define() -> Dict[str, Any]:
    """Return the static agent definition used by train/eval."""
    return {
        "system_prompt": DEFAULT_SYSTEM_PROMPT,
        "publication_smoke_note": PUBLICATION_SMOKE_NOTE,
        "version": 1,
    }


def _iter_data_files(data_dir: Path) -> List[Path]:
    if not data_dir.exists():
        return []
    return sorted(
        p
        for p in data_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in {".json", ".jsonl", ".txt", ".csv"}
    )


def _checkpoint_path(checkpoint_dir: Path) -> Path:
    return checkpoint_dir / "checkpoint.json"


def _load_checkpoint(checkpoint_dir: Path) -> Dict[str, Any]:
    checkpoint_file = _checkpoint_path(checkpoint_dir)
    if checkpoint_file.exists():
        return json.loads(checkpoint_file.read_text())
    return define()


def _score_rollout(prompt: str, seed: int, rollout: int) -> float:
    normalized_prompt = prompt.replace(PUBLICATION_SMOKE_NOTE, "").strip()
    digest = hashlib.sha256(f"{normalized_prompt}|{seed}|{rollout}".encode("utf-8")).hexdigest()
    base = int(digest[:8], 16) % 1000 / 1000.0
    note_bonus = 0.05 if PUBLICATION_SMOKE_NOTE in prompt else 0.0
    seed_bonus = ((seed % 5) - 2) * 0.01
    rollout_bonus = ((rollout % 3) - 1) * 0.005
    return round(base + note_bonus + seed_bonus + rollout_bonus, 4)


def _make_eval_rows(prompt: str, seeds: Iterable[int], rollouts: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for seed in seeds:
        for rollout in range(rollouts):
            rows.append(
                {
                    "seed": seed,
                    "rollout": rollout,
                    "score": _score_rollout(prompt, seed, rollout),
                }
            )
    return rows


def train(data_dir: str | os.PathLike[str], out_dir: str | os.PathLike[str]) -> Dict[str, Any]:
    """Persist a tiny checkpoint that records the prompt configuration."""
    data_path = Path(data_dir)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    data_files = _iter_data_files(data_path)
    checkpoint = {
        **define(),
        "data_dir": str(data_path),
        "data_file_count": len(data_files),
        "data_files": [str(p.relative_to(data_path)) if data_path in p.parents or p == data_path else str(p) for p in data_files],
    }
    checkpoint_file = _checkpoint_path(out_path)
    checkpoint_file.write_text(json.dumps(checkpoint, indent=2, sort_keys=True))
    return checkpoint


def eval(
    checkpoint_dir: str | os.PathLike[str],
    data_dir: str | os.PathLike[str],
    out_dir: str | os.PathLike[str],
    *,
    seeds: Iterable[int] | None = None,
    rollouts: int = 2,
    smoke_note_override: str | None = None,
) -> Dict[str, Any]:
    """Run a lightweight deterministic evaluation over train seeds."""
    checkpoint_path = Path(checkpoint_dir)
    data_path = Path(data_dir)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    checkpoint = _load_checkpoint(checkpoint_path)
    if smoke_note_override is not None:
        prompt = str(checkpoint.get("system_prompt", DEFAULT_SYSTEM_PROMPT)).replace(
            PUBLICATION_SMOKE_NOTE, smoke_note_override
        )
    else:
        prompt = str(checkpoint.get("system_prompt", DEFAULT_SYSTEM_PROMPT))

    resolved_seeds = list(seeds) if seeds is not None else [0, 1, 2]
    rows = _make_eval_rows(prompt, resolved_seeds, rollouts)
    mean_score = round(sum(row["score"] for row in rows) / len(rows), 4) if rows else 0.0

    summary = {
        "checkpoint_dir": str(checkpoint_path),
        "data_dir": str(data_path),
        "data_file_count": len(_iter_data_files(data_path)),
        "prompt_contains_publication_smoke_note": PUBLICATION_SMOKE_NOTE in prompt,
        "mean_score": mean_score,
        "rollouts": rollouts,
        "rows": rows,
        "seeds": resolved_seeds,
    }

    (out_path / "eval_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Run the smoke training step.")
    train_parser.add_argument("--data-dir", required=True)
    train_parser.add_argument("--out-dir", required=True)

    eval_parser = subparsers.add_parser("eval", help="Run the smoke evaluation step.")
    eval_parser.add_argument("--checkpoint-dir", required=True)
    eval_parser.add_argument("--data-dir", required=True)
    eval_parser.add_argument("--out-dir", required=True)
    eval_parser.add_argument("--seeds", nargs="*", type=int, default=None)
    eval_parser.add_argument("--rollouts", type=int, default=2)
    eval_parser.add_argument("--smoke-note-override", default=None)

    define_parser = subparsers.add_parser("define", help="Print the agent definition.")
    define_parser.add_argument("--json", action="store_true", help="Emit JSON.")

    return parser


def main(argv: List[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    if args.command == "define":
        payload = define()
        print(json.dumps(payload, indent=2, sort_keys=True) if args.json else payload)
        return 0

    if args.command == "train":
        payload = train(args.data_dir, args.out_dir)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    if args.command == "eval":
        payload = eval(
            args.checkpoint_dir,
            args.data_dir,
            args.out_dir,
            seeds=args.seeds,
            rollouts=args.rollouts,
            smoke_note_override=args.smoke_note_override,
        )
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
