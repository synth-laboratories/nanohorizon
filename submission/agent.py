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

from nanohorizon.craftax_core.metadata import PRIMARY_TOOL_NAME
from nanohorizon.shared.common import write_json
from nanohorizon.shared.craftax_data import (
    collect_rollouts_concurrently,
    rollout_achievements,
    rollout_llm_call_count,
    rollout_outcome_reward,
    summarize_achievement_frequencies,
)
from nanohorizon.shared.train_lora import release_cuda_memory
from nanohorizon.shared.vllm_eval import LocalVLLMEvalConfig, local_vllm_server

_SEED_MANIFEST_PATH = REPO_ROOT / "data" / "craftax" / "craftax_prompt_opt_starter_seeds.json"
_PRIVATE_TODO_ITEMS = (
    "1) immediate danger or blocker",
    "2) nearest sapling or wood target",
    "3) next progression target such as a table or furnace",
    "4) loop-break fallback if progress stalls",
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


def _system_prompt() -> str:
    todo_items = "\n".join(_PRIVATE_TODO_ITEMS)
    return (
        "You are a Craftax policy.\n"
        "Keep a tiny private four-step todo list before deciding:\n"
        f"{todo_items}\n"
        "Refresh the todo every turn and replace stale targets instead of repeating an old plan.\n"
        "Follow this priority order every turn:\n"
        "1. If a gatherable or placeable target is adjacent, use `do` immediately.\n"
        "2. If no useful target is adjacent, move toward the nearest wood source or sapling first.\n"
        "3. If you already have enough wood and no table is down, place a table as soon as the inventory allows it.\n"
        "4. If you have a table and the inventory supports the next crafting step, place a furnace.\n"
        "5. If the last batch did not unlock a new achievement or obvious progress, change direction or target category instead of repeating the same pattern.\n"
        "Prioritize collecting saplings and wood first.\n"
        "Use `do` only when the current state makes it obvious that you are adjacent to a gatherable, craftable, or placeable target.\n"
        "When a loop is forming, replace the stale target with a different direction or resource goal and keep moving.\n"
        "If nothing useful is adjacent, explore toward new information rather than immediately backtracking; prefer a new axis or direction over undoing the previous move.\n"
        "Avoid repeated no-op movement loops.\n"
        "Choose a short 3 or 4 action batch that follows the first todo item and, when safe, ends next to a useful target for the next turn.\n"
        "Return only valid full-Craftax actions, with no invented action names.\n"
        f"Use the provided `{PRIMARY_TOOL_NAME}` tool exactly once for the final answer.\n"
        "Do not reveal the todo list or scratchpad.\n"
        "Do not return JSON or plain text actions."
    )


def define() -> dict[str, Any]:
    return {
        "name": "craftax_submission_agent",
        "description": "Single-file NanoHorizon submission surface for prompt-first Craftax agents.",
        "candidate_label": _env_str("NANOHORIZON_SUBMISSION_CANDIDATE_LABEL", "codex-20260418T072236Z"),
        "base_model": _env_str("NANOHORIZON_SUBMISSION_BASE_MODEL", "Qwen/Qwen3.5-4B"),
        "train_seeds": _default_train_seeds(),
        "max_steps": _env_int("NANOHORIZON_SUBMISSION_MAX_STEPS", 10),
        "max_concurrent_rollouts": 1,
        "max_length": 8192,
        "max_new_tokens": _env_int("NANOHORIZON_SUBMISSION_MAX_NEW_TOKENS", 512),
        "thinking_budget_tokens": _env_int("NANOHORIZON_SUBMISSION_THINKING_BUDGET_TOKENS", 2048),
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
        "candidate_label": _env_str("NANOHORIZON_SUBMISSION_CANDIDATE_LABEL", "codex-20260418T072236Z"),
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
    release_cuda_memory()
    container_url = str(os.getenv("NANOHORIZON_CRAFTAX_CONTAINER_URL", "direct://local"))
    worker_token = str(os.getenv("NANOHORIZON_CRAFTAX_CONTAINER_WORKER_TOKEN", ""))
    inference_url = str(
        os.getenv("NANOHORIZON_EVAL_INFERENCE_URL")
        or os.getenv("NANOHORIZON_EVAL_INFERENCE_BASE_URL")
        or ""
    ).strip()
    inference_api_key = str(
        os.getenv("NANOHORIZON_EVAL_API_KEY")
        or os.getenv("OPENAI_API_KEY")
        or ""
    ).strip()
    base_model = str(config.get("base_model", "Qwen/Qwen3.5-4B"))
    request_model = str(os.getenv("NANOHORIZON_EVAL_REQUEST_MODEL") or base_model).strip() or base_model
    max_steps = int(config.get("max_steps", 10))
    max_length = int(config.get("max_length", 8192))
    max_new_tokens = int(config.get("max_new_tokens", 512))
    enable_thinking = bool(config.get("enable_thinking", True))
    thinking_budget_tokens = int(config.get("thinking_budget_tokens", 2048))
    system_prompt = str(config.get("system_prompt", ""))
    target_action_batch_size = int(config.get("target_action_batch_size", 4))
    min_action_batch_size = int(config.get("min_action_batch_size", 3))
    capture_video = _can_capture_video()

    async def _run_rollouts(resolved_inference_url: str) -> list[dict[str, Any]]:
        return await collect_rollouts_concurrently(
            container_url=container_url,
            container_worker_token=worker_token,
            inference_url=resolved_inference_url,
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
            max_concurrent_rollouts=int(config.get("max_concurrent_rollouts", 1)),
            trace_prefix="submission_eval",
            video_capture_rollout_index=0 if capture_video else None,
            video_capture_output_dir=str(rollout_root / "video") if capture_video else "",
            request_logprobs=False,
        )

    if inference_url:
        rollouts = asyncio.run(_run_rollouts(inference_url))
    else:
        vllm_config = LocalVLLMEvalConfig(
            model=base_model,
            served_model_name=base_model,
            max_model_len=max_length,
            max_new_tokens=max_new_tokens,
            enable_thinking=enable_thinking,
            enforce_eager=False,
        )
        with local_vllm_server(
            config=vllm_config,
            log_path=out_dir / "eval_summary_vllm_server.log",
        ) as server:
            rollouts = asyncio.run(_run_rollouts(f"{str(server['base_url']).rstrip('/')}/chat/completions"))
    details: list[dict[str, Any]] = []
    rewards: list[float] = []
    llm_calls: list[float] = []
    achievement_names: set[str] = set()

    for index, rollout in enumerate(rollouts):
        seed = int(rollout.get("_request_seed") or seeds[index])
        rollout_dir = rollout_root / f"{index:05d}_{seed}"
        rollout_dir.mkdir(parents=True, exist_ok=True)
        write_json(rollout_dir / "rollout.json", rollout)
        detail = {
            "seed": seed,
            "rollout_id": str(rollout.get("rollout_id") or f"rollout_{index:05d}"),
            "outcome_reward": rollout_outcome_reward(rollout),
            "llm_call_count": rollout_llm_call_count(rollout),
            "achievements": rollout_achievements(rollout),
            "success_status": rollout.get("success_status"),
            "error": rollout.get("error"),
        }
        if not detail["error"]:
            rewards.append(float(detail["outcome_reward"] or 0.0))
            llm_calls.append(float(detail["llm_call_count"] or 0.0))
        for achievement in detail["achievements"]:
            name = str(achievement).strip()
            if name:
                achievement_names.add(name)
        details.append(detail)

    requested = len(seeds)
    result = {
        "primary_score": mean(rewards) if rewards else 0.0,
        "requested_num_eval_rollouts": requested,
        "num_eval_rollouts": len([detail for detail in details if not detail.get("error")]),
        "num_rollout_errors": len([detail for detail in details if detail.get("error")]),
        "mean_outcome_reward": mean(rewards) if rewards else 0.0,
        "mean_outcome_reward_over_requested_rollouts": (
            (sum(rewards) / float(requested)) if requested else 0.0
        ),
        "max_outcome_reward": max(rewards) if rewards else 0.0,
        "mean_llm_calls_per_rollout": mean(llm_calls) if llm_calls else 0.0,
        "achievement_names": sorted(achievement_names),
        "achievement_frequencies": summarize_achievement_frequencies(
            rollouts,
            achievement_names=sorted(achievement_names),
            denominator=requested,
        ),
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
