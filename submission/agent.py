from __future__ import annotations

import asyncio
import argparse
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
    rollout_llm_call_count,
    rollout_outcome_reward,
    summarize_achievement_frequencies,
)

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


def _system_prompt() -> str:
    return (
        "You are a Craftax policy.\n"
        "Before choosing actions, keep a tiny private todo list with exactly three items: "
        "(1) the most urgent danger or blocker, (2) the next tile, object, or resource target, "
        "and (3) the fallback action that breaks a loop if progress stalls.\n"
        "Refresh completed todo items every turn.\n"
        "If you repeat the same movement pattern without progress or new information, replace the stale target item before acting.\n"
        "Do not reveal the todo list.\n"
        "Prefer early-game progression: move toward nearby trees or other gatherable resources, use 'do' only when adjacent to a useful target, and avoid sleep, crafting, or inventory-only actions unless the local state clearly supports them.\n"
        "Choose a short 3 or 4 action batch that follows the first todo item and, when safe, ends next to a useful target for the next turn.\n"
        "Think carefully, then use the `craftax_interact` tool exactly once.\n"
        "Return 3 or 4 valid full-Craftax actions unless the episode is already done.\n"
        "Use only the tool call as the final answer. Do not output JSON, prose, or a plain-text action list."
    )


def define() -> dict[str, Any]:
    return {
        "name": "craftax_submission_agent",
        "description": "Single-file NanoHorizon submission surface for prompt-first Craftax agents with a private todo refresh gate.",
        "base_model": _env_str("NANOHORIZON_SUBMISSION_BASE_MODEL", "Qwen/Qwen3.5-4B"),
        "train_seeds": _default_train_seeds(),
        "max_steps": _env_int("NANOHORIZON_SUBMISSION_MAX_STEPS", 8),
        "max_concurrent_rollouts": _env_int("NANOHORIZON_SUBMISSION_MAX_CONCURRENT_ROLLOUTS", 4),
        "max_length": 8192,
        "max_new_tokens": _env_int("NANOHORIZON_SUBMISSION_MAX_NEW_TOKENS", 3072),
        "thinking_budget_tokens": _env_int("NANOHORIZON_SUBMISSION_THINKING_BUDGET_TOKENS", 2000),
        "enable_thinking": _env_str("NANOHORIZON_SUBMISSION_ENABLE_THINKING", "true").lower() not in {"0", "false", "no", "off"},
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
    base_model = str(config.get("base_model", "Qwen/Qwen3.5-4B"))
    max_steps = int(config.get("max_steps", 8))
    max_concurrent_rollouts = int(config.get("max_concurrent_rollouts", 4))
    max_length = int(config.get("max_length", 8192))
    max_new_tokens = int(config.get("max_new_tokens", 3072))
    thinking_budget_tokens = int(config.get("thinking_budget_tokens", 2000))
    enable_thinking = bool(config.get("enable_thinking", True))
    target_action_batch_size = int(config.get("target_action_batch_size", 4))
    min_action_batch_size = int(config.get("min_action_batch_size", 3))
    system_prompt = str(config.get("system_prompt", ""))
    inference_url = str(
        os.getenv("NANOHORIZON_EVAL_INFERENCE_URL")
        or os.getenv("NANOHORIZON_EVAL_INFERENCE_BASE_URL")
        or ""
    ).strip()
    inference_api_key = str(os.getenv("NANOHORIZON_EVAL_API_KEY", "")).strip()
    request_model = str(os.getenv("NANOHORIZON_EVAL_REQUEST_MODEL", "")).strip() or base_model
    container_url = str(os.getenv("NANOHORIZON_CRAFTAX_CONTAINER_URL", "direct://local")).strip()
    container_worker_token = str(os.getenv("NANOHORIZON_CRAFTAX_CONTAINER_WORKER_TOKEN", "")).strip()

    if inference_url:
        rollouts = asyncio.run(
            collect_rollouts_concurrently(
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
                max_concurrent_rollouts=max_concurrent_rollouts,
                trace_prefix="submission_eval",
                request_logprobs=False,
            )
        )
    else:
        from nanohorizon.shared.vllm_eval import LocalVLLMEvalConfig, local_vllm_server

        vllm_config = LocalVLLMEvalConfig(
            model=base_model,
            served_model_name=request_model,
            max_model_len=max_length,
            max_new_tokens=max_new_tokens,
            enable_thinking=enable_thinking,
            enforce_eager=False,
        )
        with local_vllm_server(
            config=vllm_config,
            log_path=out_dir / "submission_eval_vllm_server.log",
        ) as server:
            rollouts = asyncio.run(
                collect_rollouts_concurrently(
                    container_url=container_url,
                    container_worker_token=container_worker_token,
                    inference_url=f"{str(server['base_url']).rstrip('/')}/chat/completions",
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
                    max_concurrent_rollouts=max_concurrent_rollouts,
                    trace_prefix="submission_eval",
                    request_logprobs=False,
                )
            )

    details = [item if isinstance(item, dict) else {"error": "missing rollout result"} for item in rollouts]
    for index, (seed, detail) in enumerate(zip(seeds, details, strict=True)):
        rollout_dir = rollout_root / f"{index:05d}_{seed}"
        rollout_dir.mkdir(parents=True, exist_ok=True)
        write_json(rollout_dir / "rollout.json", detail)
    valid_rollouts = [item for item in details if not item.get("error") and is_rollout_payload(item)]
    rewards = [rollout_outcome_reward(item) for item in valid_rollouts]
    llm_calls = [rollout_llm_call_count(item) for item in valid_rollouts]
    achievement_frequencies = summarize_achievement_frequencies(
        details,
        denominator=len(details),
    )

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
        "achievement_names": sorted(achievement_frequencies),
        "achievement_frequencies": achievement_frequencies,
        "details": details,
        "seeds": seeds,
        "checkpoint": checkpoint,
        "config": {
            "base_model": base_model,
            "max_steps": max_steps,
            "max_concurrent_rollouts": max_concurrent_rollouts,
            "max_length": max_length,
            "max_new_tokens": max_new_tokens,
            "thinking_budget_tokens": thinking_budget_tokens,
            "enable_thinking": enable_thinking,
            "target_action_batch_size": target_action_batch_size,
            "min_action_batch_size": min_action_batch_size,
            "system_prompt": system_prompt,
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
