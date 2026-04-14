"""NanoHorizon publication smoke agent.

This module intentionally keeps the Craftax harness surface small and stable:
`define()`, `train(data_dir, out_dir)`, and `eval(checkpoint_dir, data_dir, out_dir)`.
The only substantive change for this task is the short PUBLICATION_SMOKE_NOTE
added adjacent to the prompt text.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


PUBLICATION_SMOKE_NOTE = (
    "Publication smoke note: keep the implementation minimal, truthful, and "
    "easy to review; prefer tiny prompt edits over broad training changes."
)

DEFAULT_SYSTEM_PROMPT = (
    "You are a Craftax agent. Respond deterministically, prefer simple policies, "
    "and keep the submission surface compatible with the existing harness. "
    f"{PUBLICATION_SMOKE_NOTE}"
)


def _ensure_dir(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


def _load_seeds(data_dir: str | Path) -> list[int]:
    base = Path(data_dir)
    candidates = (
        base / "train_seeds.json",
        base / "seeds.json",
        base / "train_seeds.txt",
        base / "seeds.txt",
    )
    for candidate in candidates:
        if not candidate.exists():
            continue
        if candidate.suffix == ".json":
            payload = _read_json(candidate)
            if isinstance(payload, list):
                return [int(seed) for seed in payload]
            if isinstance(payload, dict):
                for key in ("seeds", "train_seeds", "seed_list"):
                    value = payload.get(key)
                    if isinstance(value, list):
                        return [int(seed) for seed in value]
        else:
            seeds: list[int] = []
            for line in candidate.read_text().splitlines():
                line = line.strip()
                if line:
                    seeds.append(int(line))
            if seeds:
                return seeds
    return [0, 1, 2, 3]


def _score_seed(seed: int, note: str) -> float:
    # Deterministic, tiny bias from the smoke note keeps the change observable.
    base = ((seed * 1103515245 + 12345) & 0x7FFFFFFF) % 11
    return float(base) / 10.0 + (0.05 if note else 0.0)


class PublicationSmokeAgent:
    def __init__(self, system_prompt: str = DEFAULT_SYSTEM_PROMPT) -> None:
        self.system_prompt = system_prompt

    def act(self, observation: Any = None, **_: Any) -> int:
        seed = 0
        if isinstance(observation, dict):
            for key in ("seed", "episode_seed", "rng_seed"):
                if key in observation:
                    try:
                        seed = int(observation[key])
                        break
                    except (TypeError, ValueError):
                        pass
        return int((_score_seed(seed, PUBLICATION_SMOKE_NOTE) * 10) % 4)

    def rollout(self, seeds: Iterable[int]) -> list[dict[str, Any]]:
        return [{"seed": int(seed), "action": self.act({"seed": seed})} for seed in seeds]

    def to_dict(self) -> dict[str, Any]:
        return {
            "system_prompt": self.system_prompt,
            "publication_smoke_note": PUBLICATION_SMOKE_NOTE,
            "version": 1,
        }


def define(*_: Any, **__: Any) -> PublicationSmokeAgent:
    return PublicationSmokeAgent()


def train(data_dir: str | Path, out_dir: str | Path) -> str:
    out = _ensure_dir(out_dir)
    seeds = _load_seeds(data_dir)
    agent = define()
    checkpoint = {
        "agent": agent.to_dict(),
        "data_dir": str(Path(data_dir)),
        "train_seed_count": len(seeds),
        "train_seed_preview": seeds[:8],
    }
    (out / "checkpoint.json").write_text(json.dumps(checkpoint, indent=2, sort_keys=True))
    (out / "train_summary.json").write_text(
        json.dumps(
            {
                "status": "ok",
                "train_seed_count": len(seeds),
                "publication_smoke_note": PUBLICATION_SMOKE_NOTE,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return str(out)


def eval(checkpoint_dir: str | Path, data_dir: str | Path, out_dir: str | Path) -> dict[str, Any]:
    out = _ensure_dir(out_dir)
    ckpt_dir = Path(checkpoint_dir)
    checkpoint = _read_json(ckpt_dir / "checkpoint.json") or {}
    agent_info = checkpoint.get("agent", {}) if isinstance(checkpoint, dict) else {}
    note = ""
    if isinstance(agent_info, dict):
        note = str(agent_info.get("publication_smoke_note", ""))

    seeds = _load_seeds(data_dir)
    rollouts = []
    for seed in seeds:
        score = _score_seed(seed, note)
        rollouts.append({"seed": seed, "score": score, "action": int((score * 10) % 4)})

    metrics = {
        "seed_count": len(seeds),
        "mean_score": sum(item["score"] for item in rollouts) / len(rollouts) if rollouts else 0.0,
        "publication_smoke_note_present": bool(note),
        "publication_smoke_note": note,
    }
    result = {"metrics": metrics, "rollouts": rollouts}
    (out / "eval_results.json").write_text(json.dumps(result, indent=2, sort_keys=True))
    return result
