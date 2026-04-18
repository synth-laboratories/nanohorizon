from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from nanohorizon.craftax_core.metadata import PRIMARY_TOOL_NAME
from nanohorizon.shared.common import write_json
from nanohorizon.shared.craftax_data import collect_rollouts_concurrently, summarize_rollouts

_SEED_MANIFEST_PATH = REPO_ROOT / "data" / "craftax" / "craftax_prompt_opt_starter_seeds.json"
_TARGET_ACTION_BATCH_SIZE = 4
_MIN_ACTION_BATCH_SIZE = 3
_THINKING_BUDGET_TOKENS = 1200


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


def _rollout_system_prompt() -> str:
    return (
        "You are a Craftax policy.\n"
        f"You may think for up to about {_THINKING_BUDGET_TOKENS} tokens before answering.\n"
        "Keep a tiny private scratchpad with exactly three items: immediate target, loop breaker, and "
        "fallback if blocked.\n"
        f"Return a short useful macro-action with {_MIN_ACTION_BATCH_SIZE}-{_TARGET_ACTION_BATCH_SIZE} "
        "valid full-Craftax actions.\n"
        "Prefer the nearest useful resource in this order: tree, sapling, then anything that unlocks progress.\n"
        "Use 'do' only when facing a useful nearby object or resource.\n"
        "If the last two turns did not change inventory, achievements, or position meaningfully, switch "
        "target or direction instead of repeating the same move.\n"
        "Read the recent action history and avoid repeating unproductive loops.\n"
        f"Use the provided `{PRIMARY_TOOL_NAME}` tool exactly once for the final answer.\n"
        "Do not reveal the scratchpad or plain-text chain of thought.\n"
        "Your final assistant action must be a tool call with valid full-Craftax actions."
    )


def define() -> dict[str, Any]:
    system_prompt = _rollout_system_prompt()
    return {
        "name": "craftax_submission_short_macro_agent",
        "description": "Single-file NanoHorizon submission surface for short-macro Craftax agents.",
        "base_model": _env_str("NANOHORIZON_SUBMISSION_BASE_MODEL", "Qwen/Qwen3.5-4B"),
        "train_seeds": _default_train_seeds(),
        "max_steps": _env_int("NANOHORIZON_SUBMISSION_MAX_STEPS", 10),
        "max_concurrent_rollouts": 1,
        "max_length": 8192,
        "max_new_tokens": _env_int("NANOHORIZON_SUBMISSION_MAX_NEW_TOKENS", 384),
        "thinking_budget_tokens": _env_int("NANOHORIZON_SUBMISSION_THINKING_BUDGET_TOKENS", _THINKING_BUDGET_TOKENS),
        "enable_thinking": False,
        "system_prompt": system_prompt,
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


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


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
    rollouts = asyncio.run(
        collect_rollouts_concurrently(
            container_url=str(os.getenv("NANOHORIZON_CRAFTAX_CONTAINER_URL", "direct://local")),
            container_worker_token=str(os.getenv("NANOHORIZON_CRAFTAX_CONTAINER_WORKER_TOKEN", "")),
            environment_api_key=str(os.getenv("NANOHORIZON_EVAL_API_KEY", "")),
            inference_url=str(os.getenv("NANOHORIZON_EVAL_INFERENCE_URL", os.getenv("NANOHORIZON_EVAL_INFERENCE_BASE_URL", ""))),
            model=str(config.get("base_model", "Qwen/Qwen3.5-4B")),
            api_key=str(os.getenv("NANOHORIZON_EVAL_API_KEY", "")),
            seeds=seeds,
            max_steps=int(config.get("max_steps", 10)),
            system_prompt=str(config.get("system_prompt", _rollout_system_prompt())),
            temperature=0.0,
            max_tokens=int(config.get("max_new_tokens", 384)),
            enable_thinking=bool(config.get("enable_thinking", False)),
            thinking_budget_tokens=int(config.get("thinking_budget_tokens", _THINKING_BUDGET_TOKENS)),
            policy_version=str(config.get("name", "craftax_submission_short_macro_agent")),
            target_action_batch_size=int(config.get("target_action_batch_size", _TARGET_ACTION_BATCH_SIZE)),
            min_action_batch_size=int(config.get("min_action_batch_size", _MIN_ACTION_BATCH_SIZE)),
            request_timeout_seconds=300.0,
            max_concurrent_rollouts=int(config.get("max_concurrent_rollouts", 1)),
            trace_prefix="submission_agent_train_eval",
            video_capture_rollout_index=0 if _can_capture_video() else None,
            video_capture_output_dir=str(out_dir / "video_capture"),
            request_logprobs=False,
        )
    )
    summary = summarize_rollouts(rollouts)
    details: list[dict[str, Any]] = []
    achievement_counts: dict[str, int] = {}
    achievement_names: set[str] = set()
    for index, rollout in enumerate(rollouts):
        metadata = rollout.get("metadata", {})
        reward_info = rollout.get("reward_info", {})
        if not isinstance(metadata, dict):
            metadata = {}
        if not isinstance(reward_info, dict):
            reward_info = {}
        achievements = [str(item).strip() for item in metadata.get("achievements", []) if str(item).strip()]
        detail = {
            "seed": int(metadata.get("seed") or rollout.get("_request_seed") or 0),
            "rollout_id": str(metadata.get("rollout_id") or rollout.get("rollout_id") or f"rollout_{index:05d}"),
            "trace_correlation_id": str(rollout.get("trace_correlation_id") or ""),
            "outcome_reward": float(reward_info.get("outcome_reward", 0.0) or 0.0),
            "llm_call_count": float(metadata.get("llm_call_count", 0.0) or 0.0),
            "achievements": achievements,
            "success_status": rollout.get("success_status"),
            "error": rollout.get("error"),
        }
        details.append(detail)
        for achievement in achievements:
            achievement_names.add(achievement)
            achievement_counts[achievement] = achievement_counts.get(achievement, 0) + 1

    requested = len(seeds)
    valid_rewards = [float(item["outcome_reward"]) for item in details if not item.get("error")]
    result = {
        "primary_score": float(summary.get("mean_outcome_reward", 0.0) or 0.0),
        "requested_num_eval_rollouts": requested,
        "num_eval_rollouts": int(summary.get("num_rollouts", 0) or 0),
        "num_rollout_errors": int(summary.get("num_errors", 0) or 0),
        "mean_outcome_reward": float(summary.get("mean_outcome_reward", 0.0) or 0.0),
        "mean_outcome_reward_over_requested_rollouts": (sum(valid_rewards) / float(requested)) if requested else 0.0,
        "max_outcome_reward": float(summary.get("max_outcome_reward", 0.0) or 0.0),
        "mean_llm_calls_per_rollout": float(summary.get("mean_llm_calls_per_rollout", 0.0) or 0.0),
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
        "rollout_summary": summary,
    }
    _write_jsonl(out_dir / "rollouts.jsonl", rollouts)
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
