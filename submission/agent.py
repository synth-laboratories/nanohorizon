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
from nanohorizon.shared.craftax_data import collect_rollouts_concurrently_with_summary

_SEED_MANIFEST_PATH = REPO_ROOT / "data" / "craftax" / "craftax_prompt_opt_starter_seeds.json"
_ACHIEVEMENT_LADDER = (
    "wake_up",
    "collect_wood",
    "place_table",
    "make_wood_pickaxe",
    "collect_stone",
    "make_stone_pickaxe",
    "collect_coal",
    "place_furnace",
    "collect_iron",
    "make_iron_pickaxe",
    "collect_sapling",
    "place_plant",
    "collect_drink",
    "eat_plant",
    "eat_cow",
    "make_wood_sword",
    "make_stone_sword",
    "defeat_zombie",
    "defeat_skeleton",
    "make_arrow",
    "make_torch",
    "open_chest",
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


def _env_bool(name: str, default: bool) -> bool:
    raw = str(os.getenv(name, "")).strip().lower()
    if not raw:
        return bool(default)
    if raw in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if raw in {"0", "false", "f", "no", "n", "off"}:
        return False
    return bool(default)


def _normalize_inference_url(raw_url: str) -> str:
    url = str(raw_url or "").strip().rstrip("/")
    if not url:
        return ""
    if url.endswith("/chat/completions"):
        return url
    if url.endswith("/v1beta/openai"):
        return f"{url}/chat/completions"
    if url.endswith("/v1"):
        return f"{url}/chat/completions"
    return f"{url}/v1/chat/completions"


def _default_train_seeds() -> list[int]:
    if _SEED_MANIFEST_PATH.exists():
        payload = json.loads(_SEED_MANIFEST_PATH.read_text(encoding="utf-8"))
        values = payload.get("train_seeds") if isinstance(payload, dict) else None
        if isinstance(values, list) and values:
            return [int(item) for item in values]
    return [seed for seed in range(0, 20)]


def _system_prompt() -> str:
    ladder = ", ".join(_ACHIEVEMENT_LADDER)
    return (
        "You are a Craftax policy agent.\n"
        "Optimize for the next new achievement, not for long prose.\n"
        "Use thinking to inspect the scene, then call the action tool exactly once.\n"
        "Keep the action batch short and coherent, usually one action unless a tiny\n"
        "sequence is clearly necessary.\n"
        "Prefer the earliest missing achievement in this ladder: "
        f"{ladder}.\n"
        "If a useful resource or enemy is directly adjacent, use `do`.\n"
        "Otherwise move toward the nearest promising resource, crafting setup, or\n"
        "safe exploration frontier.\n"
        "Do not repeat loops. Do not mix unrelated goals in the same batch.\n"
        "Return only the tool call."
    )


def _resolve_inference_backend(config: dict[str, Any]) -> tuple[str, str, str]:
    inference_url = _normalize_inference_url(
        _env_str("NANOHORIZON_EVAL_INFERENCE_URL", _env_str("NANOHORIZON_EVAL_INFERENCE_BASE_URL", ""))
    )
    if not inference_url:
        if str(os.getenv("GEMINI_API_KEY", "")).strip():
            inference_url = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
        elif str(os.getenv("OPENAI_API_KEY", "")).strip():
            inference_url = "https://api.openai.com/v1/chat/completions"
    api_key = _env_str(
        "NANOHORIZON_EVAL_API_KEY",
        _env_str("GEMINI_API_KEY", _env_str("OPENAI_API_KEY", "")),
    )
    request_model = _env_str(
        "NANOHORIZON_EVAL_REQUEST_MODEL",
        str(config.get("base_model", "Qwen/Qwen3.5-4B")),
    )
    return inference_url, api_key, request_model


def define() -> dict[str, Any]:
    return {
        "name": "craftax_submission_agent",
        "description": "Thinking-enabled Craftax submission candidate with an achievement-ladder prompt.",
        "base_model": _env_str("NANOHORIZON_SUBMISSION_BASE_MODEL", "Qwen/Qwen3.5-4B"),
        "train_seeds": _default_train_seeds(),
        "max_steps": _env_int("NANOHORIZON_SUBMISSION_MAX_STEPS", 8),
        "max_concurrent_rollouts": 1,
        "max_length": 8192,
        "max_new_tokens": _env_int("NANOHORIZON_SUBMISSION_MAX_NEW_TOKENS", 3072),
        "thinking_budget_tokens": _env_int("NANOHORIZON_SUBMISSION_THINKING_BUDGET_TOKENS", 2000),
        "enable_thinking": _env_bool("NANOHORIZON_SUBMISSION_ENABLE_THINKING", True),
        "target_action_batch_size": _env_int("NANOHORIZON_SUBMISSION_TARGET_ACTION_BATCH_SIZE", 1),
        "min_action_batch_size": _env_int("NANOHORIZON_SUBMISSION_MIN_ACTION_BATCH_SIZE", 1),
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
    capture_video = _can_capture_video()
    inference_url, inference_api_key, request_model = _resolve_inference_backend(config)
    rollout_concurrency = max(1, int(config.get("max_concurrent_rollouts", 1)))
    rollout_semaphore_limit = rollout_concurrency
    rollouts, summary = asyncio.run(
        collect_rollouts_concurrently_with_summary(
            container_url=str(os.getenv("NANOHORIZON_CRAFTAX_CONTAINER_URL", "direct://local")),
            container_worker_token=str(os.getenv("NANOHORIZON_CRAFTAX_CONTAINER_WORKER_TOKEN", "")),
            environment_api_key=str(os.getenv("NANOHORIZON_CRAFTAX_CONTAINER_ENVIRONMENT_API_KEY", "")),
            inference_url=inference_url,
            model=request_model,
            api_key=inference_api_key,
            seeds=seeds,
            max_steps=int(config.get("max_steps", 8)),
            system_prompt=str(config.get("system_prompt", "")),
            temperature=0.0,
            max_tokens=int(config.get("max_new_tokens", 3072)),
            enable_thinking=bool(config.get("enable_thinking", True)),
            thinking_budget_tokens=int(config.get("thinking_budget_tokens", 2000)),
            policy_version="submission-agent",
            target_action_batch_size=int(config.get("target_action_batch_size", 1)),
            min_action_batch_size=int(config.get("min_action_batch_size", 1)),
            request_timeout_seconds=float(os.getenv("NANOHORIZON_EVAL_REQUEST_TIMEOUT_SECONDS", "300")),
            max_concurrent_rollouts=rollout_concurrency,
            trace_prefix="submission_eval",
            video_capture_rollout_index=0 if capture_video else None,
            video_capture_output_dir=str(rollout_root / "video") if capture_video else "",
            rollout_concurrency=rollout_concurrency,
            rollout_semaphore_limit=rollout_semaphore_limit,
            request_logprobs=False,
        )
    )

    details: list[dict[str, Any]] = []
    rewards: list[float] = []
    llm_calls: list[float] = []
    achievement_counts: dict[str, int] = {}
    achievement_names: set[str] = set()

    for index, rollout in enumerate(rollouts):
        rollout_dir = rollout_root / f"{index:05d}_{int(rollout.get('seed', seeds[index] if index < len(seeds) else index))}"
        rollout_dir.mkdir(parents=True, exist_ok=True)
        detail = dict(rollout)
        detail.setdefault("seed", int(detail.get("seed", seeds[index] if index < len(seeds) else index)))
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
        "rollout_summary": summary,
        "inference_backend": {
            "inference_url": inference_url,
            "request_model": request_model,
            "policy_model": str(config.get("base_model", "Qwen/Qwen3.5-4B")),
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
