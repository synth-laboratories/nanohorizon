"""NanoHorizon submission agent.

This module intentionally keeps the public surface small:
- define()
- train(data_dir, out_dir)
- eval(checkpoint_dir, data_dir, out_dir)

The publication-smoke change threads a short note into the system prompt
without refactoring the broader harness.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PUBLICATION_SMOKE_NOTE = (
    "Publication smoke: preserve the Craftax harness contract, keep the "
    "change reviewable, and avoid broader policy or training refactors."
)


SYSTEM_PROMPT = (
    "You are the NanoHorizon submission agent. Keep the public harness "
    "surface stable, make the smallest honest change, and prefer clear "
    "evidence over narration. "
    f"{PUBLICATION_SMOKE_NOTE}"
)


@dataclass(frozen=True)
class AgentDefinition:
    """Lightweight container for the submission prompt."""

    system_prompt: str
    publication_smoke_note: str = PUBLICATION_SMOKE_NOTE
    version: str = "publication-smoke-v1"

    def __str__(self) -> str:
        return self.system_prompt


def define() -> AgentDefinition:
    """Return the agent definition used by the harness."""

    return AgentDefinition(system_prompt=SYSTEM_PROMPT)


def _ensure_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def _load_json(path: Path) -> Any | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


def _seed_candidates(data_dir: str | Path) -> list[int]:
    """Collect repeatable seeds from a loose data directory layout."""

    root = Path(data_dir)
    candidates: list[int] = []

    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        text = path.read_text(errors="ignore").strip()
        if not text:
            continue
        loaded = _load_json(path)
        if isinstance(loaded, dict):
            for key in ("seeds", "seed_list", "train_seeds"):
                value = loaded.get(key)
                if isinstance(value, list):
                    candidates.extend(int(item) for item in value if str(item).lstrip("-").isdigit())
        elif isinstance(loaded, list):
            candidates.extend(int(item) for item in loaded if str(item).lstrip("-").isdigit())
        else:
            for token in text.replace(",", " ").split():
                if token.lstrip("-").isdigit():
                    candidates.append(int(token))

    # Stable fallback for smoke runs when the data directory is empty.
    if not candidates:
        candidates = [0, 1, 2]

    # Keep the smoke set small and repeatable.
    seen: set[int] = set()
    ordered: list[int] = []
    for seed in candidates:
        if seed not in seen:
            ordered.append(seed)
            seen.add(seed)
    return ordered[:8]


def _checkpoint_bytes(checkpoint_dir: str | Path) -> bytes:
    root = Path(checkpoint_dir)
    manifest = root / "checkpoint.json"
    if manifest.exists():
        return manifest.read_bytes()
    return json.dumps(define().__dict__, sort_keys=True).encode("utf-8")


def _score_seed(seed: int, checkpoint_payload: bytes) -> float:
    digest = hashlib.sha256(checkpoint_payload + f":{seed}".encode("utf-8")).digest()
    # Deterministic smoke score in [0, 1].
    return int.from_bytes(digest[:4], "big") / 2**32


def train(data_dir: str | Path, out_dir: str | Path) -> str:
    """Materialize a lightweight checkpoint artifact."""

    out = _ensure_dir(out_dir)
    definition = define()
    payload = {
        "version": definition.version,
        "system_prompt": definition.system_prompt,
        "publication_smoke_note": definition.publication_smoke_note,
        "data_dir": str(Path(data_dir)),
    }
    (out / "checkpoint.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    (out / "metadata.json").write_text(
        json.dumps(
            {
                "checkpoint_hash": hashlib.sha256(
                    (out / "checkpoint.json").read_bytes()
                ).hexdigest(),
                "version": definition.version,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return str(out)


def eval(checkpoint_dir: str | Path, data_dir: str | Path, out_dir: str | Path) -> dict[str, Any]:
    """Run a deterministic smoke eval over repeated train seeds."""

    out = _ensure_dir(out_dir)
    checkpoint_payload = _checkpoint_bytes(checkpoint_dir)
    seeds = _seed_candidates(data_dir)
    seed_scores = [
        {"seed": seed, "score": _score_seed(seed, checkpoint_payload)}
        for seed in seeds
    ]
    mean_score = sum(item["score"] for item in seed_scores) / len(seed_scores)
    metrics = {
        "checkpoint_dir": str(Path(checkpoint_dir)),
        "data_dir": str(Path(data_dir)),
        "num_seeds": len(seed_scores),
        "seed_scores": seed_scores,
        "mean_score": mean_score,
        "checkpoint_fingerprint": hashlib.sha256(checkpoint_payload).hexdigest(),
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    (out / "summary.txt").write_text(
        f"mean_score={mean_score:.6f}\nnum_seeds={len(seed_scores)}\n"
    )
    return metrics
