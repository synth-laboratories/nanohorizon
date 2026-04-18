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

from nanohorizon.shared.common import write_json
from nanohorizon.craftax_core.metadata import PRIMARY_TOOL_NAME
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
_CANDIDATE_NAME = "craftax_prompt_opt_codex_durable_intent_fix"

_CRAFTAX_SYSTEM_PROMPT = (
    "You are a Craftax policy agent.\n"
    "Before choosing actions, keep a tiny private todo list with exactly three items: "
    "(1) the best immediate survival or resource need, (2) the next tile or object you need to reach, "
    "and (3) the next unlock that progress would enable.\n"
    "Refresh completed items every turn.\n"
    "If you repeat the same move pattern without progress, replace the target item instead of continuing the loop.\n"
    "Do not reveal the todo list to the user.\n"
    "Prefer early-game progression: move toward trees or other gatherable resources, use `do` only when adjacent "
    "to a useful target, and avoid sleep, crafting, or inventory-only actions unless the local state clearly "
    "supports them.\n"
    f"Think carefully, then use the `{PRIMARY_TOOL_NAME}` tool exactly once.\n"
    "Return 3 or 4 valid full-Craftax actions unless the episode is already done.\n"
    "Use only the tool call as the final answer.\n"
    "Do not output JSON, prose, or a plain-text action list."
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
        "candidate_name": _CANDIDATE_NAME,
        "base_model": _env_str("NANOHORIZON_SUBMISSION_BASE_MODEL", "Qwen/Qwen3.5-4B"),
        "train_seeds": _default_train_seeds(),
        "max_steps": _env_int("NANOHORIZON_SUBMISSION_MAX_STEPS", 10),
        "max_concurrent_rollouts": 1,
        "max_length": 8192,
        "max_new_tokens": _env_int("NANOHORIZON_SUBMISSION_MAX_NEW_TOKENS", 3072),
        "thinking_budget_tokens": _env_int("NANOHORIZON_SUBMISSION_THINKING_BUDGET_TOKENS", 2000),
        "enable_thinking": True,
        "target_action_batch_size": _env_int("NANOHORIZON_SUBMISSION_TARGET_ACTION_BATCH_SIZE", 4),
        "min_action_batch_size": _env_int("NANOHORIZON_SUBMISSION_MIN_ACTION_BATCH_SIZE", 3),
        "system_prompt": _CRAFTAX_SYSTEM_PROMPT,
    }


def train(data_dir: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "candidate_name": _CANDIDATE_NAME,
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
    release_cuda_memory()
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
    eval_limit = _env_int("NANOHORIZON_SUBMISSION_EVAL_LIMIT", 0)
    if eval_limit > 0:
        seeds = seeds[:eval_limit]
    rollout_root = out_dir / "rollouts"
    rollout_root.mkdir(parents=True, exist_ok=True)
    base_model = str(config.get("base_model", "Qwen/Qwen3.5-4B"))
    resolved_inference_url = str(
        os.getenv("NANOHORIZON_EVAL_INFERENCE_URL", os.getenv("NANOHORIZON_EVAL_INFERENCE_BASE_URL", ""))
    ).strip()
    resolved_inference_api_key = str(os.getenv("NANOHORIZON_EVAL_API_KEY", "")).strip()
    resolved_request_model = str(os.getenv("NANOHORIZON_EVAL_REQUEST_MODEL", "")).strip() or base_model
    resolved_container_url = str(os.getenv("NANOHORIZON_CRAFTAX_CONTAINER_URL", "direct://local"))
    resolved_container_worker_token = str(os.getenv("NANOHORIZON_CRAFTAX_CONTAINER_WORKER_TOKEN", "")).strip()
    max_steps = int(config.get("max_steps", 12))
    max_length = int(config.get("max_length", 8192))
    max_new_tokens = int(config.get("max_new_tokens", 384))
    thinking_budget_tokens = int(config.get("thinking_budget_tokens", 1024))
    enable_thinking = bool(config.get("enable_thinking", True))
    target_action_batch_size = int(config.get("target_action_batch_size", 4))
    min_action_batch_size = int(config.get("min_action_batch_size", 3))
    capture_video = _can_capture_video()
    system_prompt = str(config.get("system_prompt", _CRAFTAX_SYSTEM_PROMPT))

    if resolved_inference_url:
        rollouts = asyncio.run(
            collect_rollouts_concurrently(
                container_url=resolved_container_url,
                container_worker_token=resolved_container_worker_token,
                inference_url=resolved_inference_url,
                model=resolved_request_model,
                api_key=resolved_inference_api_key,
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
                trace_prefix="submission_eval",
                video_capture_rollout_index=0 if capture_video else None,
                video_capture_output_dir=str(rollout_root) if capture_video else "",
            )
        )
    else:
        local_config = LocalVLLMEvalConfig(
            model=base_model,
            served_model_name=base_model,
            lora_name="",
            lora_path="",
            max_lora_rank=16,
            max_model_len=max_length,
            max_new_tokens=max_new_tokens,
            enable_thinking=enable_thinking,
            enforce_eager=False,
        )
        try:
            with local_vllm_server(
                config=local_config,
                log_path=out_dir / "submission_eval_vllm_server.log",
            ) as server:
                rollouts = asyncio.run(
                    collect_rollouts_concurrently(
                        container_url=resolved_container_url,
                        container_worker_token=resolved_container_worker_token,
                        inference_url=f"{str(server['base_url']).rstrip('/')}/chat/completions",
                        model=resolved_request_model,
                        api_key=resolved_inference_api_key,
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
                        trace_prefix="submission_eval",
                        video_capture_rollout_index=0 if capture_video else None,
                        video_capture_output_dir=str(rollout_root) if capture_video else "",
                    )
                )
        except FileNotFoundError as exc:
            raise RuntimeError(
                "Local honest eval could not start because the expected vllm binary is missing. "
                "Set NANOHORIZON_EVAL_INFERENCE_URL to a reachable OpenAI-compatible endpoint or run "
                "in an environment with the local teacher vllm binary installed."
            ) from exc

    details: list[dict[str, Any]] = []
    rewards: list[float] = []
    llm_calls: list[float] = []
    achievement_names: set[str] = set()
    for index, rollout in enumerate(rollouts):
        seed = int(rollout.get("_request_seed") or (seeds[index] if index < len(seeds) else 0))
        rollout_dir = rollout_root / f"{index:05d}_{seed}"
        rollout_dir.mkdir(parents=True, exist_ok=True)
        detail = {
            "seed": seed,
            "rollout_id": str(rollout.get("rollout_id") or f"rollout_{index:05d}"),
            "trace_correlation_id": str(rollout.get("trace_correlation_id") or ""),
            "outcome_reward": rollout_outcome_reward(rollout),
            "llm_call_count": rollout_llm_call_count(rollout),
            "achievements": rollout_achievements(rollout),
            "success_status": rollout.get("success_status"),
            "error": rollout.get("error"),
        }
        if not detail.get("error"):
            candidate = rollout_dir / "rollout.mp4"
            if candidate.exists():
                detail["mp4_path"] = str(candidate)
        details.append(detail)
        if not detail.get("error"):
            rewards.append(float(detail.get("outcome_reward", 0.0) or 0.0))
            llm_calls.append(float(detail.get("llm_call_count", 0.0) or 0.0))
        for achievement in detail.get("achievements", []) or []:
            name = str(achievement).strip()
            if name:
                achievement_names.add(name)

    requested = len(seeds)
    result = {
        "candidate_name": _CANDIDATE_NAME,
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
            achievement_names=sorted(achievement_names) or None,
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
