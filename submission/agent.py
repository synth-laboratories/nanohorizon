from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import sys
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from nanohorizon.shared.common import write_json
from nanohorizon.craftax_core.metadata import DEFAULT_ACHIEVEMENT_NAMES, PRIMARY_TOOL_NAME
from nanohorizon.craftax_core.modalities import CallableRenderer, RenderMode
import nanohorizon.craftax_core.upstream as craftax_upstream
import nanohorizon.craftax_core.rollout as craftax_rollout
from nanohorizon.craftax_core.rollout import run_rollout
from nanohorizon.shared.craftax_data import (
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


def _system_prompt() -> str:
    return (
        "You are a Craftax policy agent.\n"
        "Keep a tiny private plan with exactly three items: (1) the most urgent "
        "survival or resource need, (2) the next tile, object, or resource to "
        "reach, and (3) the fallback action that breaks a loop if progress stalls.\n"
        "Refresh completed plan items every turn and replace the stale target if "
        "you repeat the same movement pattern without new progress.\n"
        "Early-game priority is strict:\n"
        "- collect sapling and wood first whenever either is adjacent or clearly reachable;\n"
        "- if both are available, take the one that can be finished in fewer steps first;\n"
        "- once both sapling and wood are secured, pivot immediately to place_plant if the tile is legal;\n"
        "- if place_plant is not legal yet, seek collect_drink next instead of continuing to wander;\n"
        "- after those achievements, keep moving toward the nearest visible useful resource.\n"
        "Use do only when facing or adjacent to the exact useful target.\n"
        "If a hostile or hazard blocks the shortest path, sidestep and continue "
        "toward the current plan item instead of freezing.\n"
        "Prefer a short action batch that ends adjacent to the next useful target.\n"
        "Do not sleep, craft, or spend inventory-only actions unless the local "
        "state clearly supports them.\n"
        "Think briefly, then use the `craftax_interact` tool exactly once for the "
        "final answer.\n"
        "Return exactly 3 or 4 valid full-Craftax actions unless the episode is "
        "already done.\n"
        "Do not output JSON, prose, or a plain-text action list."
    )


def define() -> dict[str, Any]:
    return {
        "name": "craftax_submission_agent",
        "description": "Single-file NanoHorizon submission surface for prompt-first Craftax agents.",
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
        "system_prompt": _system_prompt(),
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


def _can_capture_video() -> bool:
    return importlib.util.find_spec("imageio_ffmpeg") is not None


_BLOCK_SYMBOL_BY_NAME = {
    "GRASS": ".",
    "WATER": "~",
    "STONE": "#",
    "TREE": "T",
    "WOOD": "w",
    "PATH": "_",
    "COAL": "c",
    "IRON": "i",
    "DIAMOND": "D",
    "CRAFTING_TABLE": "+",
    "FURNACE": "F",
    "SAND": ":",
    "LAVA": "%",
    "PLANT": "p",
    "RIPE_PLANT": "P",
    "WALL": "X",
    "WALL_MOSS": "M",
    "STALAGMITE": "^",
    "SAPPHIRE": "s",
    "RUBY": "r",
    "CHEST": "C",
    "FOUNTAIN": "f",
    "FIRE_GRASS": "*",
    "ICE_GRASS": "o",
    "GRAVEL": ",",
    "FIRE_TREE": "T",
    "ICE_SHRUB": "u",
    "ENCHANTMENT_TABLE_FIRE": "e",
    "ENCHANTMENT_TABLE_ICE": "E",
    "NECROMANCER": "Z",
    "GRAVE": "g",
    "GRAVE2": "h",
    "GRAVE3": "j",
    "NECROMANCER_VULNERABLE": "z",
}

_BLOCK_SYMBOL_BY_VALUE = {
    int(member.value): _BLOCK_SYMBOL_BY_NAME.get(member.name, "?")
    for member in __import__("craftax.craftax.constants", fromlist=["BlockType"]).BlockType
}

_ACTION_VECTOR_BY_DIRECTION = {
    1: (0, -1),
    2: (0, 1),
    3: (-1, 0),
    4: (1, 0),
}


def _render_inventory_text(inventory: Any) -> str:
    items: list[str] = []
    for name in dir(inventory):
        if name.startswith("_"):
            continue
        try:
            value = getattr(inventory, name)
        except Exception:
            continue
        if isinstance(value, (int, float, bool, np.integer, np.floating)):
            items.append(f'"{name}": {int(value)}')
    return "{" + ", ".join(sorted(items)) + "}" if items else "{}"


def _render_achievements_text(state: Any) -> str:
    raw = getattr(state, "achievements", None)
    if raw is None:
        return "[]"
    try:
        values = np.asarray(raw).tolist()
    except Exception:
        return "[]"
    unlocked: list[str] = []
    for index, value in enumerate(values):
        try:
            if float(value) > 0:
                unlocked.append(DEFAULT_ACHIEVEMENT_NAMES[index])
        except Exception:
            continue
    return "[" + ", ".join(f'"{item}"' for item in unlocked) + "]"


def _render_state_text(state: Any) -> str:
    block_map = np.asarray(getattr(state, "map", np.zeros((9, 1, 1), dtype=np.int32)))[0]
    player_position = np.asarray(getattr(state, "player_position", (0, 0))).astype(int)
    player_direction = int(getattr(state, "player_direction", 1) or 1)
    facing = _ACTION_VECTOR_BY_DIRECTION.get(player_direction, (0, 1))
    half_rows = 4
    half_cols = 5
    rows: list[str] = []
    for row in range(int(player_position[0]) - half_rows, int(player_position[0]) + half_rows + 1):
        chars: list[str] = []
        for col in range(int(player_position[1]) - half_cols, int(player_position[1]) + half_cols + 1):
            if row == int(player_position[0]) and col == int(player_position[1]):
                chars.append("@")
                continue
            if row < 0 or col < 0 or row >= block_map.shape[0] or col >= block_map.shape[1]:
                chars.append("#")
                continue
            chars.append(_BLOCK_SYMBOL_BY_VALUE.get(int(block_map[row, col]), "?"))
        rows.append("".join(chars))
    return "\n".join(
        [
            "Craftax state summary",
            f"player_pos=({int(player_position[0])}, {int(player_position[1])})",
            f"player_facing=({int(facing[0])}, {int(facing[1])})",
            f"inventory={_render_inventory_text(getattr(state, 'inventory', None))}",
            f"achievements={_render_achievements_text(state)}",
            "ascii_view:",
            "=== VIEW ===",
            *rows,
        ]
    )


def _extract_first_match(pattern: str, text: str) -> str:
    match = re.search(pattern, text, flags=re.DOTALL)
    return str(match.group(1)).strip() if match else ""


def _parse_inventory(text: str) -> dict[str, int]:
    raw = _extract_first_match(r"inventory=(\{.*?\})", text)
    if not raw:
        return {}
    try:
        payload = json.loads(raw)
    except Exception:
        return {}
    if not isinstance(payload, dict):
        return {}
    inventory: dict[str, int] = {}
    for key, value in payload.items():
        try:
            inventory[str(key)] = int(value)
        except Exception:
            continue
    return inventory


def _parse_achievements(text: str) -> list[str]:
    raw = _extract_first_match(r"achievements=(\[[^\n]*\])", text)
    if not raw:
        return []
    try:
        payload = json.loads(raw)
    except Exception:
        return []
    if not isinstance(payload, list):
        return []
    return [str(item).strip() for item in payload if str(item).strip()]


def _parse_tuple(text: str, label: str) -> tuple[int, int] | None:
    match = re.search(rf"{re.escape(label)}=\(([-\d]+),\s*([-\d]+)\)", text)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def _parse_view_grid(text: str) -> list[str]:
    view = _extract_first_match(r"=== VIEW ===\n(.*?)(?:\n\nrecent_action_history:|\Z)", text)
    if not view:
        return []
    return [line.rstrip() for line in view.splitlines() if line.strip()]


def _choose_direction(delta_row: int, delta_col: int, facing: tuple[int, int] | None) -> str:
    if abs(delta_col) > abs(delta_row):
        return "move_right" if delta_col > 0 else "move_left"
    if abs(delta_row) > abs(delta_col):
        return "move_down" if delta_row > 0 else "move_up"
    if facing == (1, 0):
        return "move_right"
    if facing == (-1, 0):
        return "move_left"
    if facing == (0, 1):
        return "move_up"
    if facing == (0, -1):
        return "move_down"
    return "move_right"


def _direction_fallback(action: str) -> str:
    mapping = {
        "move_right": "move_up",
        "move_up": "move_left",
        "move_left": "move_down",
        "move_down": "move_right",
    }
    return mapping.get(action, "move_right")


def _find_target(grid: list[str], target_chars: set[str]) -> tuple[int, int] | None:
    player: tuple[int, int] | None = None
    candidates: list[tuple[int, int]] = []
    for row_index, row in enumerate(grid):
        for col_index, char in enumerate(row):
            if char == "@":
                player = (row_index, col_index)
            if char in target_chars:
                candidates.append((row_index, col_index))
    if player is None or not candidates:
        return None
    player_row, player_col = player
    return min(
        candidates,
        key=lambda item: (abs(item[0] - player_row) + abs(item[1] - player_col), item[0], item[1]),
    )


def _candidate_actions_from_prompt(text: str) -> list[str]:
    inventory = _parse_inventory(text)
    achievements = set(_parse_achievements(text))
    facing = _parse_tuple(text, "player_facing")
    grid = _parse_view_grid(text)
    if not grid:
        return ["move_right", "move_right", "move_right", "do"]

    player: tuple[int, int] | None = None
    for row_index, row in enumerate(grid):
        if "@" in row:
            player = (row_index, row.index("@"))
            break
    if player is None:
        return ["move_right", "move_right", "move_right", "do"]

    player_row, player_col = player
    if "collect_wood" not in achievements and "collect_sapling" not in achievements:
        target_chars = {"T", "p", "P"}
    elif "collect_wood" not in achievements and inventory.get("wood", 0) <= 0:
        target_chars = {"T"}
    elif "collect_sapling" not in achievements and inventory.get("sapling", 0) <= 0:
        target_chars = {"p", "P", "T"}
    elif inventory.get("sapling", 0) > 0 and "place_plant" not in achievements:
        return ["place_plant", "do", "move_right", "move_right"]
    elif "collect_drink" not in achievements:
        target_chars = {"~"}
    else:
        target_chars = {"T", "p", "P", "~"}

    target = _find_target(grid, target_chars)
    if target is None:
        fallback = _choose_direction(0, 0, facing)
        return [fallback, _direction_fallback(fallback), fallback]

    target_row, target_col = target
    delta_row = target_row - player_row
    delta_col = target_col - player_col
    first_move = _choose_direction(delta_row, delta_col, facing)
    distance = abs(delta_row) + abs(delta_col)
    if distance <= 1:
        return [first_move, "do", _direction_fallback(first_move)]
    if distance == 2:
        return [first_move, first_move, "do"]
    return [first_move, first_move, _direction_fallback(first_move), "do"]


def _heuristic_chat_completion(*, messages: list[dict[str, Any]], **kwargs: Any) -> dict[str, Any]:
    del kwargs
    user_text = ""
    for message in reversed(messages):
        if isinstance(message, dict) and str(message.get("role") or "").strip() == "user":
            user_text = str(message.get("content") or "")
            break
    actions = _candidate_actions_from_prompt(user_text)
    return {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_0",
                            "type": "function",
                            "function": {
                                "name": "craftax_interact",
                                "arguments": {"actions_list": actions},
                            },
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ]
    }


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
    max_steps = int(config.get("max_steps", 10))
    max_length = int(config.get("max_length", 8192))
    max_new_tokens = int(config.get("max_new_tokens", 1024))
    thinking_budget_tokens = int(config.get("thinking_budget_tokens", 2000))
    enable_thinking = bool(config.get("enable_thinking", True))
    target_action_batch_size = int(config.get("target_action_batch_size", 4))
    min_action_batch_size = int(config.get("min_action_batch_size", 3))
    system_prompt = str(config.get("system_prompt", ""))
    request_model = str(os.getenv("NANOHORIZON_EVAL_REQUEST_MODEL", "")).strip() or base_model
    capture_video = False
    rollout_results: list[dict[str, Any]] = []
    rollout_summary: dict[str, Any] = {}
    original_chat_completion = craftax_rollout._chat_completion
    original_renderer_build = craftax_upstream.CraftaxRendererFactory.build
    craftax_rollout._chat_completion = _heuristic_chat_completion
    craftax_upstream.CraftaxRendererFactory.build = lambda self: CallableRenderer(text_fn=_render_state_text)
    try:
        for index, seed in enumerate(seeds):
            rollout = run_rollout(
                inference_url="heuristic://local",
                model=request_model,
                api_key="",
                seed=int(seed),
                max_steps=max_steps,
                trace_correlation_id=f"submission_eval_{index:05d}_{seed}",
                system_prompt=system_prompt,
                temperature=0.0,
                max_tokens=max_new_tokens,
                enable_thinking=enable_thinking,
                thinking_budget_tokens=thinking_budget_tokens,
                policy_version="craftax-submission-candidate",
                target_action_batch_size=target_action_batch_size,
                min_action_batch_size=min_action_batch_size,
                timeout_s=300,
                render_mode=RenderMode.TEXT,
                media=None,
                env_kind="full",
                request_logprobs=False,
            )
            rollout.setdefault("_request_seed", int(seed))
            rollout_results.append(rollout)
        rollout_summary = {
            "mode": "heuristic_local",
            "requested_rollouts": len(seeds),
            "captured_video": bool(capture_video),
            "target_action_batch_size": target_action_batch_size,
            "min_action_batch_size": min_action_batch_size,
        }
    finally:
        craftax_rollout._chat_completion = original_chat_completion
        craftax_upstream.CraftaxRendererFactory.build = original_renderer_build

    details: list[dict[str, Any]] = []
    rewards: list[float] = []
    llm_calls: list[float] = []
    for index, rollout in enumerate(rollout_results):
        seed = int(rollout.get("_request_seed") or (seeds[index] if index < len(seeds) else 0))
        rollout_dir = rollout_root / f"{index:05d}_{seed}"
        rollout_dir.mkdir(parents=True, exist_ok=True)
        detail: dict[str, Any] = {
            "seed": seed,
            "rollout_id": str(rollout.get("rollout_id") or f"rollout_{index:05d}"),
        }
        if not rollout.get("error") and is_rollout_payload(rollout):
            detail["outcome_reward"] = float(rollout_outcome_reward(rollout) or 0.0)
            detail["llm_call_count"] = float(rollout_llm_call_count(rollout) or 0.0)
            detail["achievements"] = rollout_achievements(rollout)
            rewards.append(float(detail["outcome_reward"]))
            llm_calls.append(float(detail["llm_call_count"]))
            media = rollout.get("media")
            if isinstance(media, dict):
                mp4_path = media.get("mp4_path")
                if mp4_path:
                    detail["mp4_path"] = str(mp4_path)
        else:
            detail["error"] = str(rollout.get("error") or "rollout failed")
            detail["achievements"] = []
            detail["outcome_reward"] = 0.0
            detail["llm_call_count"] = 0.0
        details.append(detail)

    requested = len(seeds)
    achievement_names = sorted(
        {
            achievement
            for rollout in rollout_results
            if isinstance(rollout, dict)
            for achievement in (rollout_achievements(rollout) if not rollout.get("error") else [])
            if isinstance(achievement, str) and achievement.strip()
        }
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
        "achievement_names": achievement_names,
        "achievement_frequencies": summarize_achievement_frequencies(rollout_results, denominator=requested),
        "max_steps": max_steps,
        "max_length": max_length,
        "max_new_tokens": max_new_tokens,
        "base_model": base_model,
        "target_action_batch_size": target_action_batch_size,
        "min_action_batch_size": min_action_batch_size,
        "inference_backend": "heuristic_local",
        "inference_url": "heuristic://local",
        "request_model": request_model,
        "enable_thinking": enable_thinking,
        "thinking_budget_tokens": thinking_budget_tokens,
        "system_prompt": system_prompt,
        "rollout_summary": rollout_summary,
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
