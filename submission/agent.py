from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path
from statistics import mean
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from nanohorizon.shared.common import write_json
from nanohorizon.shared.eval_model import evaluate_model

_SEED_MANIFEST_PATH = REPO_ROOT / "data" / "craftax" / "craftax_prompt_opt_starter_seeds.json"


def _env_int(name: str, default: int) -> int:
    raw = str(os.getenv(name, "")).strip()
    if not raw:
        return int(default)
    try:
        return int(raw)
    except ValueError:
        return int(default)


def _env_str(name: str, default: str) -> str:
    raw = str(os.getenv(name, "")).strip()
    return raw or default


def _default_train_seeds() -> list[int]:
    if _SEED_MANIFEST_PATH.exists():
        payload = json.loads(_SEED_MANIFEST_PATH.read_text(encoding="utf-8"))
        values = payload.get("train_seeds") if isinstance(payload, dict) else None
        if isinstance(values, list) and values:
            return [int(item) for item in values]
    return [seed for seed in range(0, 20)]


def define() -> dict[str, Any]:
    return {
        "name": "craftax_submission_agent",
        "description": "Single-file NanoHorizon submission surface for prompt-first Craftax agents.",
        "base_model": _env_str("NANOHORIZON_SUBMISSION_BASE_MODEL", "Qwen/Qwen3.5-4B"),
        "train_seeds": _default_train_seeds(),
        "max_steps": _env_int("NANOHORIZON_SUBMISSION_MAX_STEPS", 10),
        "max_concurrent_rollouts": 1,
        "max_length": 8192,
        "max_new_tokens": _env_int("NANOHORIZON_SUBMISSION_MAX_NEW_TOKENS", 512),
        "thinking_budget_tokens": _env_int("NANOHORIZON_SUBMISSION_THINKING_BUDGET_TOKENS", 3000),
        "enable_thinking": False,
        "target_action_batch_size": _env_int("NANOHORIZON_SUBMISSION_TARGET_ACTION_BATCH_SIZE", 8),
        "min_action_batch_size": _env_int("NANOHORIZON_SUBMISSION_MIN_ACTION_BATCH_SIZE", 5),
        "system_prompt": (
            "You are a Craftax policy.\n"
            "Think briefly, then return a short useful macro-action with valid full-Craftax actions.\n"
            "Explore when nothing useful is adjacent, but keep wood gathering reliable as the default.\n"
            "If a sapling, drink, or place_plant opportunity is immediately safe and useful, take it as the next deliberate target instead of wandering.\n"
            "Use 'do' only when facing a useful nearby object or resource.\n"
            "Read the recent action history and avoid repeating unproductive loops.\n"
            "Call the action tool exactly once in the final answer."
        ),
    }


def train(data_dir: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "define": define(),
        "train_data_dir": str(data_dir),
        "trained": False,
    }
    write_json(out_dir / "checkpoint.json", checkpoint)


def _resolve_seeds(data_dir: Path, config: dict[str, Any]) -> list[int]:
    seeds_path = data_dir / "seeds.json"
    if seeds_path.exists():
        payload = json.loads(seeds_path.read_text(encoding="utf-8"))
        values = payload.get("seeds") if isinstance(payload, dict) else payload
        if isinstance(values, list):
            return [int(item) for item in values]
    return [int(item) for item in config.get("train_seeds", [])]


def _can_capture_video() -> bool:
    return importlib.util.find_spec("imageio_ffmpeg") is not None


def eval(checkpoint_dir: Path, data_dir: Path, out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "checkpoint.json"
    checkpoint = (
        json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if checkpoint_path.exists()
        else {"define": define()}
    )
    config = checkpoint.get("define") if isinstance(checkpoint, dict) else None
    if not isinstance(config, dict):
        config = define()

    seeds = _resolve_seeds(data_dir, config)
    rollout_root = out_dir / "rollouts"
    rollout_root.mkdir(parents=True, exist_ok=True)
    details: list[dict[str, Any]] = []
    rewards: list[float] = []
    llm_calls: list[float] = []
    achievement_counts: dict[str, int] = {}
    achievement_names: set[str] = set()

    for index, seed in enumerate(seeds):
        rollout_dir = rollout_root / f"{index:05d}_{seed}"
        rollout_dir.mkdir(parents=True, exist_ok=True)
        capture_video = _can_capture_video()
        summary = evaluate_model(
            base_model=str(config.get("base_model", "Qwen/Qwen3.5-4B")),
            output_dir=rollout_dir,
            container_url=str(os.getenv("NANOHORIZON_CRAFTAX_CONTAINER_URL", "direct://local")),
            seed_start=int(seed),
            num_rollouts=1,
            max_steps=int(config.get("max_steps", 10)),
            max_concurrent_rollouts=1,
            max_length=int(config.get("max_length", 8192)),
            max_new_tokens=int(config.get("max_new_tokens", 512)),
            thinking_budget_tokens=int(config.get("thinking_budget_tokens", 3000)),
            enable_thinking=bool(config.get("enable_thinking", False)),
            system_prompt=str(config.get("system_prompt", "")),
            inference_url=str(os.getenv("NANOHORIZON_EVAL_INFERENCE_URL", os.getenv("NANOHORIZON_EVAL_INFERENCE_BASE_URL", ""))),
            inference_api_key=str(os.getenv("NANOHORIZON_EVAL_API_KEY", "")),
            request_model=str(os.getenv("NANOHORIZON_EVAL_REQUEST_MODEL", "")),
            video_capture_rollout_index=0 if capture_video else None,
            video_capture_output_dir=str(rollout_dir) if capture_video else "",
            summary_name=f"rollout_{index:05d}_{seed}.json",
        )
        detail = dict((summary.get("details") or [{}])[0])
        detail.setdefault("seed", int(seed))
        detail.setdefault("rollout_id", f"rollout_{index:05d}")
        if not detail.get("mp4_path"):
            candidate = rollout_dir / "rollout.mp4"
            if candidate.exists():
                detail["mp4_path"] = str(candidate)
        details.append(detail)
        if not detail.get("error"):
            rewards.append(float(detail.get("outcome_reward", 0.0) or 0.0))
            llm_calls.append(float(detail.get("llm_call_count", 0.0) or 0.0))
        for achievement in detail.get("achievements", []) or []:
            name = str(achievement).strip()
            if not name:
                continue
            achievement_names.add(name)
            achievement_counts[name] = achievement_counts.get(name, 0) + 1

    requested = len(seeds)
    result = {
        "primary_score": mean(rewards) if rewards else 0.0,
        "requested_num_eval_rollouts": requested,
        "num_eval_rollouts": len([detail for detail in details if not detail.get("error")]),
        "num_rollout_errors": len([detail for detail in details if detail.get("error")]),
        "mean_outcome_reward": mean(rewards) if rewards else 0.0,
        "mean_outcome_reward_over_requested_rollouts": (sum(rewards) / float(requested)) if requested else 0.0,
        "max_outcome_reward": max(rewards) if rewards else 0.0,
        "mean_llm_calls_per_rollout": mean(llm_calls) if llm_calls else 0.0,
        "achievement_names": sorted(achievement_names),
        "achievement_frequencies": {
            name: {
                "count": int(achievement_counts.get(name, 0)),
                "frequency": (float(achievement_counts.get(name, 0)) / float(requested)) if requested else 0.0,
            }
            for name in sorted(achievement_names)
        },
        "details": details,
        "seeds": seeds,
        "checkpoint": checkpoint,
    }
    write_json(out_dir / "result.json", result)
    write_json(out_dir / "eval_summary.json", result)
    return result


def _main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["define", "train", "eval"])
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "out")
    parser.add_argument("--checkpoint-dir", type=Path, default=REPO_ROOT / "out")
    args = parser.parse_args()
    if args.phase == "define":
        print(json.dumps(define(), indent=2, sort_keys=True))
        return 0
    if args.phase == "train":
        train(args.data_dir, args.out_dir)
        return 0
    print(json.dumps(eval(args.checkpoint_dir, args.data_dir, args.out_dir), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
