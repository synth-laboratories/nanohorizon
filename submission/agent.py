"""NanoHorizon smoke submission.

This module intentionally stays small and dependency-light so the benchmark
harness can import it directly.  The only behavior change in this task is a
reviewable prompt tweak called out by ``PUBLICATION_SMOKE_NOTE``.
"""

from __future__ import annotations

import json
import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


ACTION_VOCAB = (
    "north",
    "south",
    "east",
    "west",
    "wait",
    "interact",
)

BASE_SYSTEM_PROMPT = (
    "You are a careful Craftax policy. "
    "Prefer stable, low-risk actions, keep short-term state in mind, "
    "and avoid unnecessary thrashing."
)

# Visible, reviewable prompt tweak for publication-smoke runs.
PUBLICATION_SMOKE_NOTE = (
    "Publication smoke: preserve conservative behavior, favor reproducible "
    "rollouts, and make any policy change easy to audit."
)


@dataclass(frozen=True)
class AgentConfig:
    system_prompt: str
    action_vocab: tuple[str, ...]
    note: str
    seed_bias: int = 0


def define() -> dict[str, Any]:
    """Return the default agent definition used by the harness."""

    config = AgentConfig(
        system_prompt=f"{BASE_SYSTEM_PROMPT}\n\n{PUBLICATION_SMOKE_NOTE}",
        action_vocab=ACTION_VOCAB,
        note=PUBLICATION_SMOKE_NOTE,
    )
    return asdict(config)


def _ensure_path(path_like: str | Path | dict[str, Any]) -> Path:
    if isinstance(path_like, dict):
        for key in ("checkpoint_dir", "out_dir", "path"):
            value = path_like.get(key)
            if value:
                return _ensure_path(value)
        raise TypeError(f"Cannot coerce path from mapping keys: {sorted(path_like)}")
    return path_like if isinstance(path_like, Path) else Path(path_like)


def _read_json_if_possible(path: Path) -> Any | None:
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _iter_payloads(data_dir: Path) -> Iterable[tuple[Path, Any]]:
    if not data_dir.exists():
        return []
    for path in sorted(data_dir.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() in {".json", ".jsonl", ".txt", ".yaml", ".yml"}:
            payload = _read_json_if_possible(path) if path.suffix.lower() == ".json" else None
            if payload is None:
                try:
                    payload = path.read_text()
                except Exception:
                    continue
            yield path, payload


def _extract_seed_candidates(data_dir: Path) -> list[int]:
    seeds: set[int] = set()
    for path, payload in _iter_payloads(data_dir):
        for token in path.stem.replace("-", "_").split("_"):
            if token.isdigit():
                seeds.add(int(token))

        if isinstance(payload, dict):
            for key in ("seed", "rng_seed", "train_seed", "episode_seed"):
                value = payload.get(key)
                if isinstance(value, int):
                    seeds.add(value)
            for value in payload.values():
                if isinstance(value, int) and 0 <= value < 10_000:
                    seeds.add(value)
        elif isinstance(payload, str):
            for token in payload.replace(",", " ").replace(":", " ").split():
                if token.isdigit():
                    seeds.add(int(token))
    return sorted(seeds)


def _fingerprint_text(text: str) -> int:
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big", signed=False)


def _policy_for_seed(seed: int, config: dict[str, Any]) -> str:
    vocab = tuple(config.get("action_vocab", ACTION_VOCAB))
    note = str(config.get("note", ""))
    bias = int(config.get("seed_bias", 0))
    index = (_fingerprint_text(f"{seed}:{note}:{bias}") + seed + bias) % len(vocab)
    return vocab[index]


def train(data_dir: str | Path, out_dir: str | Path) -> dict[str, Any]:
    """Create a tiny checkpoint derived from the available training data."""

    data_path = _ensure_path(data_dir)
    out_path = _ensure_path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    seeds = _extract_seed_candidates(data_path)
    config = define()
    config["seed_bias"] = len(seeds) % len(ACTION_VOCAB)

    training_summary = {
        "data_dir": str(data_path),
        "num_seed_candidates": len(seeds),
        "seed_candidates": seeds[:32],
        "policy_preview": {seed: _policy_for_seed(seed, config) for seed in seeds[:16]},
    }

    checkpoint = {
        "agent": config,
        "training_summary": training_summary,
    }

    (out_path / "checkpoint.json").write_text(json.dumps(checkpoint, indent=2, sort_keys=True))
    (out_path / "training_summary.json").write_text(
        json.dumps(training_summary, indent=2, sort_keys=True)
    )
    return str(out_path)


def _score_record(payload: Any, seed: int, config: dict[str, Any]) -> tuple[float, dict[str, Any]]:
    predicted = _policy_for_seed(seed, config)
    reference = None
    if isinstance(payload, dict):
        for key in ("action", "target_action", "gold_action", "expected_action"):
            value = payload.get(key)
            if isinstance(value, str):
                reference = value
                break
        if reference is None:
            observation = payload.get("observation")
            if isinstance(observation, str):
                reference = observation
            elif isinstance(observation, dict):
                reference = json.dumps(observation, sort_keys=True)
    elif isinstance(payload, str):
        reference = payload

    if reference is None:
        reference = f"seed:{seed}"

    score = 1.0 if predicted in reference else 0.0
    detail = {
        "seed": seed,
        "predicted_action": predicted,
        "reference_preview": reference[:120],
        "score": score,
    }
    return score, detail


def eval(
    checkpoint_dir: str | Path | dict[str, Any],
    data_dir: str | Path,
    out_dir: str | Path,
) -> dict[str, Any]:
    """Run a lightweight deterministic evaluation over the available data."""

    checkpoint_path = _ensure_path(checkpoint_dir)
    data_path = _ensure_path(data_dir)
    out_path = _ensure_path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    checkpoint_file = checkpoint_path / "checkpoint.json"
    if checkpoint_file.exists():
        checkpoint = json.loads(checkpoint_file.read_text())
        config = dict(checkpoint.get("agent", {}))
    else:
        config = define()

    records: list[dict[str, Any]] = []
    total_score = 0.0
    total_examples = 0

    for path, payload in _iter_payloads(data_path):
        seed_matches = []
        for token in path.stem.replace("-", "_").split("_"):
            if token.isdigit():
                seed_matches.append(int(token))
        if isinstance(payload, dict):
            for key in ("seed", "rng_seed", "train_seed", "episode_seed"):
                value = payload.get(key)
                if isinstance(value, int):
                    seed_matches.append(value)
        if not seed_matches:
            seed_matches = [0]

        for seed in seed_matches[:8]:
            score, detail = _score_record(payload, seed, config)
            detail["path"] = str(path)
            records.append(detail)
            total_score += score
            total_examples += 1

    if not records:
        # Smoke fallback when the dataset layout is not discoverable.
        fallback_seeds = list(range(5))
        for seed in fallback_seeds:
            score, detail = _score_record(f"seed:{seed}", seed, config)
            records.append(detail)
            total_score += score
            total_examples += 1

    mean_score = total_score / total_examples if total_examples else 0.0
    result = {
        "checkpoint_dir": str(checkpoint_path),
        "data_dir": str(data_path),
        "num_examples": total_examples,
        "mean_score": mean_score,
        "records": records,
    }
    (out_path / "eval_results.json").write_text(json.dumps(result, indent=2, sort_keys=True))
    return result


__all__ = ["PUBLICATION_SMOKE_NOTE", "define", "train", "eval"]
