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


def _system_prompt(thinking_budget_tokens: int) -> str:
    return (
        "You are a Craftax policy agent.\n"
        f"You may think for up to about {int(thinking_budget_tokens)} tokens before answering.\n"
        "Keep a tiny private plan with exactly three items: (1) the most urgent survival or resource need, "
        "(2) the next tile, object, or resource to reach, and (3) the fallback action that breaks a loop if progress stalls.\n"
        "Refresh completed plan items every turn and replace the stale target if you repeat the same movement pattern without new progress.\n"
        "Early-game priority is strict:\n"
        "- collect sapling and wood first whenever either is adjacent or clearly reachable;\n"
        "- if both are available, take the one that can be finished in fewer steps first;\n"
        "- once both sapling and wood are secured, pivot immediately to place_plant if the tile is legal;\n"
        "- if place_plant is not legal yet, seek collect_drink next instead of continuing to wander;\n"
        "- after those achievements, keep moving toward the nearest visible useful resource.\n"
        "Use do only when facing or adjacent to the exact useful target.\n"
        "If a hostile or hazard blocks the shortest path, sidestep and continue toward the current plan item instead of freezing.\n"
        "Prefer a short action batch that ends adjacent to the next useful target.\n"
        "Do not sleep, craft, or spend inventory-only actions unless the local state clearly supports them.\n"
        "Think briefly, then use the `craftax_interact` tool exactly once for the final answer.\n"
        "Return exactly 3 or 4 valid full-Craftax actions unless the episode is already done.\n"
        "Do not output JSON, prose, or a plain-text action list."
    )


def define() -> dict[str, Any]:
    thinking_budget_tokens = _env_int("NANOHORIZON_SUBMISSION_THINKING_BUDGET_TOKENS", 2000)
    return {
        "name": "craftax_submission_agent",
        "description": "Single-file NanoHorizon submission surface for prompt-first Craftax agents.",
        "base_model": _env_str("NANOHORIZON_SUBMISSION_BASE_MODEL", "Qwen/Qwen3.5-4B"),
        "train_seeds": _default_train_seeds(),
        "max_steps": _env_int("NANOHORIZON_SUBMISSION_MAX_STEPS", 10),
        "max_concurrent_rollouts": 1,
        "max_length": 8192,
        "max_new_tokens": _env_int("NANOHORIZON_SUBMISSION_MAX_NEW_TOKENS", 1024),
        "thinking_budget_tokens": thinking_budget_tokens,
        "enable_thinking": True,
        "target_action_batch_size": _env_int("NANOHORIZON_SUBMISSION_TARGET_ACTION_BATCH_SIZE", 4),
        "min_action_batch_size": _env_int("NANOHORIZON_SUBMISSION_MIN_ACTION_BATCH_SIZE", 3),
        "request_timeout_seconds": _env_int("NANOHORIZON_SUBMISSION_REQUEST_TIMEOUT_SECONDS", 300),
        "system_prompt": _system_prompt(thinking_budget_tokens),
    }


def train(data_dir: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "define": define(),
        "train_data_dir": str(data_dir),
        "candidate_focus": [
            "collect_sapling",
            "collect_wood",
            "place_plant",
            "collect_drink",
        ],
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

    container_url = str(os.getenv("NANOHORIZON_CRAFTAX_CONTAINER_URL", "direct://local")).strip()
    inference_url = str(
        os.getenv("NANOHORIZON_EVAL_INFERENCE_URL")
        or os.getenv("NANOHORIZON_EVAL_INFERENCE_BASE_URL")
        or ""
    ).strip()
    api_key = str(os.getenv("NANOHORIZON_EVAL_API_KEY") or os.getenv("OPENAI_API_KEY") or "").strip()
    request_model = _env_str("NANOHORIZON_EVAL_REQUEST_MODEL", str(config.get("base_model", "Qwen/Qwen3.5-4B")))
    request_timeout_seconds = float(config.get("request_timeout_seconds", 300))

    rollouts, rollout_summary = asyncio.run(
        collect_rollouts_concurrently_with_summary(
            container_url=container_url,
            environment_api_key=str(os.getenv("NANOHORIZON_CRAFTAX_CONTAINER_API_KEY", "")).strip(),
            inference_url=inference_url,
            model=request_model,
            api_key=api_key,
            seeds=seeds,
            max_steps=int(config.get("max_steps", 10)),
            system_prompt=str(config.get("system_prompt", "")),
            temperature=0.0,
            max_tokens=int(config.get("max_new_tokens", 1024)),
            enable_thinking=bool(config.get("enable_thinking", True)),
            thinking_budget_tokens=int(config.get("thinking_budget_tokens", 2000)),
            policy_version="craftax-submission-candidate",
            target_action_batch_size=int(config.get("target_action_batch_size", 4)),
            min_action_batch_size=int(config.get("min_action_batch_size", 3)),
            request_timeout_seconds=request_timeout_seconds,
            max_concurrent_rollouts=1,
            trace_prefix="submission_eval",
            request_logprobs=False,
        )
    )

    details: list[dict[str, Any]] = []
    rewards: list[float] = []
    llm_calls: list[float] = []
    achievement_counts: dict[str, int] = {}
    achievement_names: set[str] = set()

    for index, rollout in enumerate(rollouts):
        seed = int(rollout.get("_request_seed") or (seeds[index] if index < len(seeds) else 0))
        rollout_dir = rollout_root / f"{index:05d}_{seed}"
        rollout_dir.mkdir(parents=True, exist_ok=True)
        detail = dict((rollout.get("details") or [{}])[0]) if isinstance(rollout, dict) else {}
        detail.setdefault("seed", seed)
        detail.setdefault("rollout_id", f"rollout_{index:05d}")
        if not detail.get("error") and is_rollout_payload(rollout):
            detail["outcome_reward"] = float(rollout_outcome_reward(rollout) or 0.0)
            detail["llm_call_count"] = float(rollout_llm_call_count(rollout) or 0.0)
            detail["achievements"] = rollout_achievements(rollout)
            rewards.append(float(detail["outcome_reward"]))
            llm_calls.append(float(detail["llm_call_count"]))
            for achievement in detail.get("achievements", []) or []:
                name = str(achievement).strip()
                if not name:
                    continue
                achievement_names.add(name)
                achievement_counts[name] = achievement_counts.get(name, 0) + 1
        else:
            detail["error"] = str(rollout.get("error") or "rollout failed") if isinstance(rollout, dict) else "rollout failed"
            detail["achievements"] = []
            detail["outcome_reward"] = 0.0
            detail["llm_call_count"] = 0.0
        details.append(detail)

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
        "achievement_frequencies": summarize_achievement_frequencies(
            rollouts,
            achievement_names=sorted(achievement_names) if achievement_names else None,
            denominator=requested,
        ),
        "rollout_summary": rollout_summary,
        "details": details,
        "seeds": seeds,
        "checkpoint": checkpoint,
        "config": config,
        "container_url": container_url,
        "inference_url": inference_url,
        "request_model": request_model,
        "enable_thinking": bool(config.get("enable_thinking", True)),
        "thinking_budget_tokens": int(config.get("thinking_budget_tokens", 2000)),
        "target_action_batch_size": int(config.get("target_action_batch_size", 4)),
        "min_action_batch_size": int(config.get("min_action_batch_size", 3)),
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
