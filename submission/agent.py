from __future__ import annotations

import argparse
import asyncio
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

from nanohorizon.shared.common import ensure_dir, write_json
from nanohorizon.shared.craftax_data import (
    collect_rollouts_concurrently,
    is_rollout_payload,
    rollout_achievements,
    rollout_llm_call_count,
    rollout_outcome_reward,
    summarize_achievement_frequencies,
)
from nanohorizon.shared.train_lora import release_cuda_memory
from nanohorizon.shared.vllm_eval import LocalVLLMEvalConfig, local_vllm_server

_SEED_MANIFEST_PATH = REPO_ROOT / "data" / "craftax" / "craftax_prompt_opt_starter_seeds.json"
TODO_SCRATCHPAD_REQUIREMENTS = [
    "Keep a tiny private todo list with exactly three items before each tool call.",
    "The three items must track (1) the immediate danger or blocker, (2) the next tile, object, or resource target, and (3) the loop-break or fallback progress action.",
    "Refresh completed todo items every turn.",
    "If the policy repeats the same movement pattern without progress or new information, replace the stale target item instead of continuing the loop.",
    "Do not reveal the todo list or scratchpad in the final answer.",
]


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


def _submission_system_prompt() -> str:
    todo_contract = " ".join(TODO_SCRATCHPAD_REQUIREMENTS)
    return (
        "You are a Craftax policy agent.\n"
        f"{todo_contract}\n"
        "Prioritize survival first, then the nearest concrete unlock: a reachable resource, tool, "
        "crafting station, enemy, or other state change that moves the run forward.\n"
        "Use `do` only when the current observation makes a direct interaction obvious.\n"
        "If two turns pass without progress, replace the stale target with a new concrete one instead of extending the loop.\n"
        "Choose a short 3 or 4 action batch that follows the first todo item and, when safe, "
        "ends in a position that makes the next turn easier to decide.\n"
        "Think carefully, then call the `craftax_interact` tool exactly once in the final answer.\n"
        "Return 3 or 4 valid full-Craftax actions unless the episode is already done.\n"
        "Use only the tool call as the final answer. Do not output JSON, prose, or a plain-text action list."
    )


def define() -> dict[str, Any]:
    return {
        "name": "craftax_submission_todo_candidate",
        "description": "Single-file NanoHorizon submission surface for a prompt-first Craftax candidate with a private todo contract.",
        "base_model": _env_str("NANOHORIZON_SUBMISSION_BASE_MODEL", "Qwen/Qwen3.5-4B"),
        "train_seeds": _default_train_seeds(),
        "max_steps": _env_int("NANOHORIZON_SUBMISSION_MAX_STEPS", 10),
        "max_concurrent_rollouts": 1,
        "max_length": 8192,
        "max_new_tokens": _env_int("NANOHORIZON_SUBMISSION_MAX_NEW_TOKENS", 1024),
        "thinking_budget_tokens": _env_int("NANOHORIZON_SUBMISSION_THINKING_BUDGET_TOKENS", 2000),
        "enable_thinking": True,
        "target_action_batch_size": _env_int("NANOHORIZON_SUBMISSION_TARGET_ACTION_BATCH_SIZE", 4),
        "min_action_batch_size": _env_int("NANOHORIZON_SUBMISSION_MIN_ACTION_BATCH_SIZE", 3),
        "system_prompt": _submission_system_prompt(),
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
    out_dir = ensure_dir(out_dir)
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
    base_model = str(config.get("base_model", "Qwen/Qwen3.5-4B"))
    request_model = str(os.getenv("NANOHORIZON_EVAL_REQUEST_MODEL", "")).strip() or base_model
    inference_url = str(
        os.getenv("NANOHORIZON_EVAL_INFERENCE_URL", os.getenv("NANOHORIZON_EVAL_INFERENCE_BASE_URL", ""))
    ).strip()
    inference_api_key = str(os.getenv("NANOHORIZON_EVAL_API_KEY", "")).strip()
    container_url = str(os.getenv("NANOHORIZON_CRAFTAX_CONTAINER_URL", "direct://local"))
    container_worker_token = str(os.getenv("NANOHORIZON_CRAFTAX_CONTAINER_WORKER_TOKEN", "")).strip()
    max_steps = int(config.get("max_steps", 10))
    max_length = int(config.get("max_length", 8192))
    max_new_tokens = int(config.get("max_new_tokens", 1024))
    thinking_budget_tokens = int(config.get("thinking_budget_tokens", 2000))
    enable_thinking = bool(config.get("enable_thinking", True))
    target_action_batch_size = int(config.get("target_action_batch_size", 4))
    min_action_batch_size = int(config.get("min_action_batch_size", 3))
    system_prompt = str(config.get("system_prompt", ""))
    rollout_kwargs = dict(
        container_url=container_url,
        container_worker_token=container_worker_token,
        inference_url=inference_url,
        model=request_model,
        api_key=inference_api_key,
        seeds=seeds,
        max_steps=max_steps,
        system_prompt=system_prompt,
        temperature=0.0,
        max_tokens=max_new_tokens,
        enable_thinking=enable_thinking,
        thinking_budget_tokens=thinking_budget_tokens,
        policy_version="submission-eval",
        target_action_batch_size=target_action_batch_size,
        min_action_batch_size=min_action_batch_size,
        request_timeout_seconds=300.0,
        max_concurrent_rollouts=1,
        trace_prefix="rollout",
        video_capture_rollout_index=None,
        video_capture_output_dir="",
        video_capture_fps=6,
        video_capture_tile_size=16,
        video_capture_show_status_bars=True,
    )
    release_cuda_memory()
    if not inference_url:
        local_config = LocalVLLMEvalConfig(
            model=base_model,
            served_model_name=request_model or base_model,
            api_key=inference_api_key,
            max_model_len=max_length,
            max_new_tokens=max_new_tokens,
            enable_thinking=enable_thinking,
            enforce_eager=False,
        )
        with local_vllm_server(
            config=local_config,
            log_path=out_dir / "eval_summary_vllm_eval_server.log",
        ) as server:
            rollout_kwargs["inference_url"] = f"{str(server['base_url']).rstrip('/')}/chat/completions"
            rollout_kwargs["model"] = request_model or base_model
            rollouts = asyncio.run(collect_rollouts_concurrently(**rollout_kwargs))
    else:
        rollouts = asyncio.run(collect_rollouts_concurrently(**rollout_kwargs))

    valid_rollouts = [
        item for item in rollouts if isinstance(item, dict) and not item.get("error") and is_rollout_payload(item)
    ]
    details: list[dict[str, Any]] = []
    rewards: list[float] = []
    llm_calls: list[float] = []
    achievement_counts: dict[str, int] = {}
    achievement_names: set[str] = set()

    for index, rollout in enumerate(rollouts):
        detail = {
            "seed": int(rollout.get("_request_seed") or seeds[index]),
            "rollout_id": str(rollout.get("rollout_id") or f"rollout_{index:05d}"),
            "outcome_reward": rollout_outcome_reward(rollout),
            "llm_call_count": rollout_llm_call_count(rollout),
            "achievements": rollout_achievements(rollout),
            "success_status": rollout.get("success_status"),
            "error": rollout.get("error"),
            "trace_correlation_id": str(rollout.get("trace_correlation_id") or ""),
        }
        details.append(detail)
        if not detail.get("error") and is_rollout_payload(rollout):
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
        "num_eval_rollouts": len(valid_rollouts),
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
        "evaluation": {
            "base_model": base_model,
            "request_model": request_model,
            "inference_url": inference_url,
            "target_action_batch_size": target_action_batch_size,
            "min_action_batch_size": min_action_batch_size,
            "thinking_budget_tokens": thinking_budget_tokens,
            "enable_thinking": enable_thinking,
        },
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
