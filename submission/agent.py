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
from nanohorizon.craftax_core.rollout import RenderMode, run_rollout
from nanohorizon.shared.craftax_data import (
    is_rollout_payload,
    rollout_achievements,
    rollout_llm_call_count,
    rollout_outcome_reward,
    summarize_achievement_frequencies,
)

_SEED_MANIFEST_PATH = REPO_ROOT / "data" / "craftax" / "craftax_prompt_opt_starter_seeds.json"
_DEFAULT_BASE_MODEL = "Qwen/Qwen3.5-4B"
_DEFAULT_REQUEST_MODEL = "gpt-5.4-mini"
_DEFAULT_INFERENCE_URL = "https://api.openai.com/v1/chat/completions"


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


def _normalize_chat_url(raw_url: str) -> str:
    url = str(raw_url or "").strip()
    if not url:
        return _DEFAULT_INFERENCE_URL
    if url.endswith("/chat/completions"):
        return url
    if url.endswith("/v1"):
        return f"{url}/chat/completions"
    if url.endswith("/v1/"):
        return f"{url}chat/completions"
    return url.rstrip("/") + "/v1/chat/completions"


def _default_inference_url() -> str:
    raw = _env_str("NANOHORIZON_EVAL_INFERENCE_URL", _env_str("NANOHORIZON_EVAL_INFERENCE_BASE_URL", ""))
    if raw:
        return _normalize_chat_url(raw)
    raw = _env_str("OPENAI_BASE_URL", "")
    if raw:
        return _normalize_chat_url(raw)
    return _DEFAULT_INFERENCE_URL


def _default_request_model() -> str:
    return _env_str("NANOHORIZON_EVAL_REQUEST_MODEL", _env_str("SMR_COMPUTE_MODEL", _DEFAULT_REQUEST_MODEL))


def _default_inference_api_key() -> str:
    return _env_str("NANOHORIZON_EVAL_API_KEY", _env_str("OPENAI_API_KEY", ""))


def _build_system_prompt() -> str:
    return (
        "You are a Craftax policy agent.\n"
        "Before choosing actions, keep a tiny private todo list with exactly three items: "
        "(1) the immediate danger or blocker, (2) the next tile, object, or resource target, "
        "and (3) the loop-break or fallback progress action.\n"
        "Refresh completed items every turn. If you repeat the same movement pattern without "
        "progress or new information, replace the stale target item instead of continuing the loop.\n"
        "Do not reveal the todo list to the user.\n"
        "Prefer early-game progression: move toward nearby trees or other gatherable resources, "
        "use `do` only when adjacent to a useful target, and avoid sleep, crafting, or inventory-only "
        "actions unless the local state clearly supports them.\n"
        "Think briefly, then use the `craftax_interact` tool exactly once.\n"
        "Return a short useful macro-action with 3-4 valid full-Craftax actions unless the episode is already done.\n"
        "Use only the tool call as the final answer. Do not output JSON, prose, or a plain-text action list."
    )


def define() -> dict[str, Any]:
    return {
        "name": "craftax_submission_agent",
        "description": "Single-file NanoHorizon Craftax submission surface with a loop-breaking prompt candidate.",
        "base_model": _env_str("NANOHORIZON_SUBMISSION_BASE_MODEL", _DEFAULT_BASE_MODEL),
        "train_seeds": _default_train_seeds(),
        "max_steps": _env_int("NANOHORIZON_SUBMISSION_MAX_STEPS", 10),
        "max_concurrent_rollouts": 1,
        "max_length": 8192,
        "max_new_tokens": _env_int("NANOHORIZON_SUBMISSION_MAX_NEW_TOKENS", 512),
        "thinking_budget_tokens": _env_int("NANOHORIZON_SUBMISSION_THINKING_BUDGET_TOKENS", 2000),
        "enable_thinking": True,
        "target_action_batch_size": _env_int("NANOHORIZON_SUBMISSION_TARGET_ACTION_BATCH_SIZE", 4),
        "min_action_batch_size": _env_int("NANOHORIZON_SUBMISSION_MIN_ACTION_BATCH_SIZE", 3),
        "inference_url": _default_inference_url(),
        "inference_api_key": _default_inference_api_key(),
        "request_model": _default_request_model(),
        "system_prompt": _build_system_prompt(),
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
    achievement_names: set[str] = set()

    inference_url = str(config.get("inference_url") or _default_inference_url())
    request_model = str(config.get("request_model") or _default_request_model())
    inference_api_key = str(config.get("inference_api_key") or _default_inference_api_key())
    system_prompt = str(config.get("system_prompt") or _build_system_prompt())
    max_steps = int(config.get("max_steps", 10))
    max_new_tokens = int(config.get("max_new_tokens", 512))
    thinking_budget_tokens = int(config.get("thinking_budget_tokens", 2000))
    enable_thinking = bool(config.get("enable_thinking", True))
    target_action_batch_size = int(config.get("target_action_batch_size", 4))
    min_action_batch_size = int(config.get("min_action_batch_size", 3))

    rollouts = []
    for index, seed in enumerate(seeds):
        rollout = run_rollout(
            inference_url=inference_url,
            model=request_model,
            api_key=inference_api_key,
            seed=int(seed),
            max_steps=max_steps,
            trace_correlation_id=f"submission_candidate_{index:05d}_{int(seed)}",
            system_prompt=system_prompt,
            temperature=0.0,
            max_tokens=max_new_tokens,
            enable_thinking=enable_thinking,
            thinking_budget_tokens=thinking_budget_tokens,
            policy_version="submission-candidate",
            target_action_batch_size=target_action_batch_size,
            min_action_batch_size=min_action_batch_size,
            timeout_s=900,
            render_mode=RenderMode.NONE,
            media=None,
            env_kind="full",
            request_logprobs=False,
        )
        rollouts.append(rollout)

    for index, rollout in enumerate(rollouts):
        seed = int(seeds[index]) if index < len(seeds) else index
        rollout_dir = rollout_root / f"{index:05d}_{seed}"
        rollout_dir.mkdir(parents=True, exist_ok=True)
        write_json(rollout_dir / "rollout.json", rollout)
        detail = {
            "seed": seed,
            "rollout_id": str(rollout.get("rollout_id") or f"rollout_{index:05d}"),
            "trace_correlation_id": str(rollout.get("trace_correlation_id") or ""),
            "success_status": rollout.get("success_status"),
            "error": rollout.get("error"),
            "outcome_reward": rollout_outcome_reward(rollout),
            "llm_call_count": rollout_llm_call_count(rollout),
            "achievements": rollout_achievements(rollout),
        }
        details.append(detail)
        if not detail.get("error") and is_rollout_payload(rollout):
            rewards.append(float(detail["outcome_reward"]))
            llm_calls.append(float(detail["llm_call_count"]))
            for achievement in detail["achievements"]:
                name = str(achievement).strip()
                if name:
                    achievement_names.add(name)

    requested = len(seeds)
    achievement_frequencies = summarize_achievement_frequencies(
        rollouts,
        achievement_names=sorted(achievement_names) if achievement_names else None,
        denominator=requested,
    )
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
        "achievement_frequencies": achievement_frequencies,
        "details": details,
        "seeds": seeds,
        "checkpoint": checkpoint,
        "system_prompt": system_prompt,
        "inference_url": inference_url,
        "request_model": request_model,
        "target_action_batch_size": target_action_batch_size,
        "min_action_batch_size": min_action_batch_size,
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
