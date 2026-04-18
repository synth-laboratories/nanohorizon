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
    collect_rollouts_concurrently_with_summary,
    is_rollout_payload,
    rollout_achievements,
    rollout_llm_call_count,
    rollout_outcome_reward,
    summarize_achievement_frequencies,
)

_SEED_MANIFEST_PATH = REPO_ROOT / "data" / "craftax" / "craftax_prompt_opt_starter_seeds.json"
_BASE_SYSTEM_PROMPT = (
    "You are a Craftax policy agent.\n"
    "Plan briefly, then choose one short action for the current state.\n"
    "Priority order: survive if needed, hunt trees for wood, place a table as soon as wood is available, "
    "craft a wood pickaxe, then collect stone and craft a stone pickaxe.\n"
    "Use `do` only when facing a tree, stone, table, or another immediate crafting target.\n"
    "Never chase saplings or plants unless starvation or an immediate survival problem makes them necessary.\n"
    "Ignore `place_plant` as a goal unless survival forces it.\n"
    "If no useful target is adjacent, use this fallback search rule from the current position: even x+y -> move_right, odd x+y -> move_up.\n"
    "Use that rule consistently until a tree, stone, or table becomes adjacent, then switch to `do` or the relevant craft action immediately.\n"
    "Return exactly one valid full-Craftax action unless the episode is already done.\n"
    "Do not return JSON or plain text actions outside the tool call."
)


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
        "max_steps": _env_int("NANOHORIZON_SUBMISSION_MAX_STEPS", 8),
        "max_concurrent_rollouts": 1,
        "max_length": 8192,
        "max_new_tokens": _env_int("NANOHORIZON_SUBMISSION_MAX_NEW_TOKENS", 256),
        "thinking_budget_tokens": _env_int("NANOHORIZON_SUBMISSION_THINKING_BUDGET_TOKENS", 0),
        "enable_thinking": False,
        "target_action_batch_size": _env_int("NANOHORIZON_SUBMISSION_TARGET_ACTION_BATCH_SIZE", 1),
        "min_action_batch_size": _env_int("NANOHORIZON_SUBMISSION_MIN_ACTION_BATCH_SIZE", 1),
        "system_prompt": _BASE_SYSTEM_PROMPT,
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
    rollout_concurrency = int(config.get("max_concurrent_rollouts", 1))
    rollout_semaphore_limit = int(config.get("max_concurrent_rollouts", 1))
    rollout_results, rollout_summary = asyncio.run(
        collect_rollouts_concurrently_with_summary(
            container_url=str(os.getenv("NANOHORIZON_CRAFTAX_CONTAINER_URL", "direct://local")),
            environment_api_key=str(os.getenv("NANOHORIZON_CRAFTAX_CONTAINER_API_KEY", "")),
            inference_url=str(
                os.getenv("NANOHORIZON_EVAL_INFERENCE_URL", os.getenv("NANOHORIZON_EVAL_INFERENCE_BASE_URL", ""))
            ).strip(),
            model=str(os.getenv("NANOHORIZON_EVAL_REQUEST_MODEL", config.get("base_model", "Qwen/Qwen3.5-4B"))),
            api_key=str(os.getenv("NANOHORIZON_EVAL_API_KEY", "")),
            seeds=[int(seed) for seed in seeds],
            max_steps=int(config.get("max_steps", 8)),
            system_prompt=str(config.get("system_prompt", "")),
            temperature=0.0,
            max_tokens=int(config.get("max_new_tokens", 256)),
            enable_thinking=bool(config.get("enable_thinking", False)),
            thinking_budget_tokens=int(config.get("thinking_budget_tokens", 0)),
            policy_version="submission-reactive-tree-table-pickaxe",
            target_action_batch_size=int(config.get("target_action_batch_size", 1)),
            min_action_batch_size=int(config.get("min_action_batch_size", 1)),
            request_timeout_seconds=300.0,
            max_concurrent_rollouts=rollout_concurrency,
            trace_prefix="submission_agent_train_eval",
            rollout_concurrency=rollout_concurrency,
            rollout_semaphore_limit=rollout_semaphore_limit,
            request_logprobs=False,
        )
    )
    details: list[dict[str, Any]] = []
    rewards: list[float] = []
    llm_calls: list[float] = []
    for index, seed in enumerate(seeds):
        rollout_dir = rollout_root / f"{index:05d}_{seed}"
        rollout_dir.mkdir(parents=True, exist_ok=True)
        detail = {}
        if index < len(rollout_results) and isinstance(rollout_results[index], dict):
            detail = dict(rollout_results[index])
        detail.setdefault("seed", int(seed))
        detail.setdefault("rollout_id", f"rollout_{index:05d}")
        detail.setdefault("achievements", rollout_achievements(detail) if detail else [])
        details.append(detail)
        if not detail.get("error") and is_rollout_payload(detail):
            rewards.append(float(rollout_outcome_reward(detail)))
            llm_calls.append(float(rollout_llm_call_count(detail)))

    requested = len(seeds)
    achievement_frequencies = summarize_achievement_frequencies(rollout_results, denominator=requested)
    result = {
        "primary_score": mean(rewards) if rewards else 0.0,
        "requested_num_eval_rollouts": requested,
        "num_eval_rollouts": len([detail for detail in details if not detail.get("error") and is_rollout_payload(detail)]),
        "num_rollout_errors": len([detail for detail in details if detail.get("error")]),
        "mean_outcome_reward": mean(rewards) if rewards else 0.0,
        "mean_outcome_reward_over_requested_rollouts": (sum(rewards) / float(requested)) if requested else 0.0,
        "max_outcome_reward": max(rewards) if rewards else 0.0,
        "mean_llm_calls_per_rollout": mean(llm_calls) if llm_calls else 0.0,
        "achievement_names": [name for name, payload in achievement_frequencies.items() if int(payload.get("count", 0)) > 0],
        "achievement_frequencies": achievement_frequencies,
        "details": details,
        "seeds": seeds,
        "checkpoint": checkpoint,
        "rollout_summary": rollout_summary,
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
