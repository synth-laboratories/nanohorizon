from __future__ import annotations

import argparse
import asyncio
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
from nanohorizon.shared.craftax_data import (
    collect_rollouts_concurrently,
    is_rollout_payload,
    rollout_achievements,
    rollout_llm_call_count,
    rollout_outcome_reward,
    summarize_achievement_frequencies,
)
from nanohorizon.shared.train_lora import release_cuda_memory

_SEED_MANIFEST_PATH = REPO_ROOT / "data" / "craftax" / "craftax_prompt_opt_starter_seeds.json"
_DEFAULT_BASE_MODEL = "Qwen/Qwen3.5-4B"
_DEFAULT_MAX_STEPS = 16
_DEFAULT_MAX_NEW_TOKENS = 3072
_DEFAULT_THINKING_BUDGET_TOKENS = 2000


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


def _system_prompt() -> str:
    return (
        "You are a Craftax policy agent.\n"
        "Before choosing actions, silently maintain a tiny private three-item scratchpad:\n"
        "1. the immediate blocker or danger,\n"
        "2. the nearest concrete tile, object, or resource target,\n"
        "3. the loop-break or fallback action if progress stalls.\n"
        "Refresh the scratchpad every turn and never reveal it.\n"
        "\n"
        "Priority order:\n"
        "- If a tree, sapling, or other gatherable resource is adjacent, use `do` immediately.\n"
        "- If a gatherable resource is nearby but not adjacent, move directly toward the closest one and then harvest it; do not wander if the target is already visible.\n"
        "- After collecting wood, prefer the shortest legal progression step: place a table if needed, then craft the wood pickaxe as soon as the local state supports it.\n"
        "- If a table, wood, and stone are available, move straight to the next crafting milestone instead of exploring.\n"
        "- Treat drink, plant, and recovery opportunities as first-class goals when the observation explicitly shows them; take the shortest legal recovery action rather than spending the turn on idle movement.\n"
        "- If the observation explicitly says there is no nearby progress and it is night or energy is low, sleep instead of looping.\n"
        "- If the same movement pattern is not producing new information or progress, replace it with a different target on the next turn.\n"
        "\n"
        "Keep each action batch short, concrete, and anchored to the visible local state. "
        "Use the provided `craftax_interact` tool exactly once for the final answer. "
        "Do not output JSON or plain text actions."
    )


def define() -> dict[str, Any]:
    return {
        "name": "craftax_submission_agent",
        "description": "Single-file NanoHorizon Craftax submission with a deterministic milestone-first policy prompt.",
        "base_model": _env_str("NANOHORIZON_SUBMISSION_BASE_MODEL", _DEFAULT_BASE_MODEL),
        "train_seeds": _default_train_seeds(),
        "max_steps": _env_int("NANOHORIZON_SUBMISSION_MAX_STEPS", _DEFAULT_MAX_STEPS),
        "max_concurrent_rollouts": 1,
        "max_length": 8192,
        "max_new_tokens": _env_int("NANOHORIZON_SUBMISSION_MAX_NEW_TOKENS", _DEFAULT_MAX_NEW_TOKENS),
        "thinking_budget_tokens": _env_int(
            "NANOHORIZON_SUBMISSION_THINKING_BUDGET_TOKENS",
            _DEFAULT_THINKING_BUDGET_TOKENS,
        ),
        "enable_thinking": True,
        "target_action_batch_size": _env_int("NANOHORIZON_SUBMISSION_TARGET_ACTION_BATCH_SIZE", 4),
        "min_action_batch_size": _env_int("NANOHORIZON_SUBMISSION_MIN_ACTION_BATCH_SIZE", 3),
        "system_prompt": _system_prompt(),
    }


def train(data_dir: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "define": define(),
        "train_data_dir": str(data_dir),
        "trained": False,
        "notes": [
            "Prompt-only Craftax candidate.",
            "Milestone order: gather wood, convert it into station/tool progress, then prioritize any explicitly visible drink/sapling/plant opportunity.",
        ],
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
    release_cuda_memory()

    container_url = str(os.getenv("NANOHORIZON_CRAFTAX_CONTAINER_URL") or "direct://local").strip()
    container_worker_token = str(os.getenv("NANOHORIZON_CRAFTAX_CONTAINER_WORKER_TOKEN") or "").strip()
    inference_url = str(
        os.getenv("NANOHORIZON_EVAL_INFERENCE_URL")
        or os.getenv("NANOHORIZON_EVAL_INFERENCE_BASE_URL")
        or ""
    ).strip()
    inference_api_key = str(os.getenv("NANOHORIZON_EVAL_API_KEY") or "").strip()
    request_model = str(os.getenv("NANOHORIZON_EVAL_REQUEST_MODEL") or "").strip() or str(
        config.get("base_model", _DEFAULT_BASE_MODEL)
    )
    max_steps = int(config.get("max_steps", _DEFAULT_MAX_STEPS))
    max_length = int(config.get("max_length", 8192))
    max_new_tokens = int(config.get("max_new_tokens", _DEFAULT_MAX_NEW_TOKENS))
    thinking_budget_tokens = int(config.get("thinking_budget_tokens", _DEFAULT_THINKING_BUDGET_TOKENS))
    enable_thinking = bool(config.get("enable_thinking", True))
    system_prompt = str(config.get("system_prompt", ""))
    target_action_batch_size = int(config.get("target_action_batch_size", 4))
    min_action_batch_size = int(config.get("min_action_batch_size", 3))

    async def _run_rollouts() -> list[dict[str, Any]]:
        if inference_url:
            return await collect_rollouts_concurrently(
                container_url=container_url,
                container_worker_token=container_worker_token,
                environment_api_key=inference_api_key,
                inference_url=inference_url,
                model=request_model,
                api_key=inference_api_key,
                seeds=[int(seed) for seed in seeds],
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
                trace_prefix="submission_eval",
                request_logprobs=True,
            )

        from nanohorizon.shared.vllm_eval import LocalVLLMEvalConfig, local_vllm_server

        config_obj = LocalVLLMEvalConfig(
            model=str(config.get("base_model", _DEFAULT_BASE_MODEL)),
            served_model_name=str(config.get("base_model", _DEFAULT_BASE_MODEL)),
            max_model_len=max_length,
            max_new_tokens=max_new_tokens,
            enable_thinking=enable_thinking,
            enforce_eager=False,
        )
        with local_vllm_server(
            config=config_obj,
            log_path=out_dir / "eval_summary_vllm_eval_server.log",
        ) as server:
            return await collect_rollouts_concurrently(
                container_url=container_url,
                container_worker_token=container_worker_token,
                environment_api_key=inference_api_key,
                inference_url=f"{str(server['base_url']).rstrip('/')}/chat/completions",
                model=request_model,
                api_key=inference_api_key,
                seeds=[int(seed) for seed in seeds],
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
                trace_prefix="submission_eval",
                request_logprobs=True,
            )

    rollouts = asyncio.run(_run_rollouts())
    valid_rollouts = [
        item for item in rollouts if isinstance(item, dict) and not item.get("error") and is_rollout_payload(item)
    ]
    requested = len(rollouts)
    rewards = [rollout_outcome_reward(item) for item in valid_rollouts]
    llm_calls = [float(rollout_llm_call_count(item)) for item in valid_rollouts]
    achievement_names = sorted(
        {
            achievement
            for item in valid_rollouts
            for achievement in rollout_achievements(item)
            if str(achievement).strip()
        }
    )
    result = {
        "primary_score": mean(rewards) if rewards else 0.0,
        "requested_num_eval_rollouts": requested,
        "num_eval_rollouts": len(valid_rollouts),
        "num_rollout_errors": len(rollouts) - len(valid_rollouts),
        "mean_outcome_reward": mean(rewards) if rewards else 0.0,
        "mean_outcome_reward_over_requested_rollouts": (sum(rewards) / float(requested)) if requested else 0.0,
        "max_outcome_reward": max(rewards) if rewards else 0.0,
        "mean_llm_calls_per_rollout": mean(llm_calls) if llm_calls else 0.0,
        "achievement_names": sorted(achievement_names),
        "achievement_frequencies": summarize_achievement_frequencies(
            rollouts,
            achievement_names=achievement_names,
            denominator=requested,
        ),
        "details": [
            {
                "seed": int(item.get("_request_seed") or 0),
                "rollout_id": str(item.get("rollout_id") or ""),
                "trace_correlation_id": str(item.get("trace_correlation_id") or ""),
                "outcome_reward": rollout_outcome_reward(item),
                "llm_call_count": rollout_llm_call_count(item),
                "achievements": rollout_achievements(item),
                "success_status": item.get("success_status"),
                "error": item.get("error"),
            }
            for item in rollouts
        ],
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
