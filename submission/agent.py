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
from nanohorizon.shared.eval_model import evaluate_model
from nanohorizon.craftax_core.metadata import PRIMARY_TOOL_NAME

_SEED_MANIFEST_PATH = REPO_ROOT / "data" / "craftax" / "craftax_prompt_opt_starter_seeds.json"
_CANDIDATE_LABEL = "codex-20260418T103225Z"
_ACHIEVEMENT_LADDER = (
    "collect_wood",
    "place_table",
    "collect_stone",
    "make_wood_pickaxe",
    "place_furnace",
    "collect_coal",
    "collect_iron",
    "make_stone_pickaxe",
    "make_iron_pickaxe",
    "make_iron_sword",
    "make_iron_armour",
    "find_bow",
    "make_arrow",
    "make_torch",
    "enter_gnomish_mines",
    "enter_dungeon",
    "enter_sewers",
    "enter_vault",
    "enter_troll_mines",
    "enter_fire_realm",
    "enter_ice_realm",
    "defeat_zombie",
    "defeat_skeleton",
    "defeat_orc_soldier",
    "defeat_orc_mage",
    "defeat_troll",
    "defeat_necromancer",
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


def _build_system_prompt(target_action_batch_size: int) -> str:
    ladder = " -> ".join(_ACHIEVEMENT_LADDER)
    return (
        "You are a Craftax policy optimizing unique achievements.\n"
        "Read the observation text and recent action history, then choose the next macro-action batch.\n"
        "Priorities:\n"
        f"1. Progress along this achievement ladder when possible: {ladder}.\n"
        "2. If a needed resource, tool, crafting station, hostile, or portal is visible or adjacent, act on it immediately.\n"
        "3. If nothing actionable is visible, use a deterministic clockwise sweep: move_right, move_right, move_down, move_down, move_left, move_left, move_up, move_up.\n"
        "   Continue the current sweep from the action history instead of restarting it.\n"
        "4. Prefer movement over noop, sleep, or rest unless the observation explicitly says to recover.\n"
        "5. Fight only when armed or when combat clearly unlocks a new achievement; otherwise reposition first.\n"
        "6. Do not repeat an unchanged unproductive batch if the last turn made no progress.\n"
        f"Output contract: use {PRIMARY_TOOL_NAME} exactly once and return exactly {int(target_action_batch_size)} valid full-Craftax actions.\n"
        "No JSON, no prose, no extra tool calls."
    )


def define() -> dict[str, Any]:
    target_action_batch_size = _env_int("NANOHORIZON_SUBMISSION_TARGET_ACTION_BATCH_SIZE", 8)
    return {
        "name": "craftax_submission_agent",
        "description": "Single-file NanoHorizon submission surface for prompt-first Craftax agents.",
        "base_model": _env_str("NANOHORIZON_SUBMISSION_BASE_MODEL", "Qwen/Qwen3.5-4B"),
        "train_seeds": _default_train_seeds(),
        "max_steps": _env_int("NANOHORIZON_SUBMISSION_MAX_STEPS", 12),
        "max_concurrent_rollouts": 1,
        "max_length": 8192,
        "max_new_tokens": _env_int("NANOHORIZON_SUBMISSION_MAX_NEW_TOKENS", 256),
        "thinking_budget_tokens": _env_int("NANOHORIZON_SUBMISSION_THINKING_BUDGET_TOKENS", 768),
        "enable_thinking": True,
        "target_action_batch_size": target_action_batch_size,
        "min_action_batch_size": _env_int("NANOHORIZON_SUBMISSION_MIN_ACTION_BATCH_SIZE", 5),
        "system_prompt": _build_system_prompt(target_action_batch_size),
        "candidate_label": _CANDIDATE_LABEL,
    }


def train(data_dir: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "candidate_label": _CANDIDATE_LABEL,
        "define": define(),
        "train_data_dir": str(data_dir),
        "train_seed_count": len(_default_train_seeds()),
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
    target_action_batch_size = int(config.get("target_action_batch_size", 8))
    rollout_root = out_dir / "rollouts"
    rollout_root.mkdir(parents=True, exist_ok=True)
    details: list[dict[str, Any]] = []
    rewards: list[float] = []
    llm_calls: list[float] = []
    achievement_counts: dict[str, int] = {}
    achievement_names: set[str] = set()

    for index, seed in enumerate(seeds):
        rollout_dir = rollout_root / f"{index:05d}_{seed}"
        rollout_dir.mkdir(parents=True, exist_ok=True)
        capture_video = _can_capture_video()
        summary = evaluate_model(
            base_model=str(config.get("base_model", "Qwen/Qwen3.5-4B")),
            output_dir=rollout_dir,
            container_url=str(os.getenv("NANOHORIZON_CRAFTAX_CONTAINER_URL", "direct://local")),
            seed_start=int(seed),
            num_rollouts=1,
            max_steps=int(config.get("max_steps", 12)),
            max_concurrent_rollouts=1,
            max_length=int(config.get("max_length", 8192)),
            max_new_tokens=int(config.get("max_new_tokens", 256)),
            thinking_budget_tokens=int(config.get("thinking_budget_tokens", 768)),
            enable_thinking=bool(config.get("enable_thinking", True)),
            system_prompt=str(config.get("system_prompt") or _build_system_prompt(target_action_batch_size)),
            inference_url=str(os.getenv("NANOHORIZON_EVAL_INFERENCE_URL", os.getenv("NANOHORIZON_EVAL_INFERENCE_BASE_URL", ""))),
            inference_api_key=str(os.getenv("NANOHORIZON_EVAL_API_KEY", "")),
            request_model=str(os.getenv("NANOHORIZON_EVAL_REQUEST_MODEL", "")),
            video_capture_rollout_index=0 if capture_video else None,
            video_capture_output_dir=str(rollout_dir) if capture_video else "",
            summary_name=f"rollout_{index:05d}_{seed}.json",
        )
        detail = dict((summary.get("details") or [{}])[0])
        detail.setdefault("seed", int(seed))
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
        "candidate_label": _CANDIDATE_LABEL,
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
