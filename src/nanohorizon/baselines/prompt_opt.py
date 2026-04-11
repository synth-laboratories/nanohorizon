"""Craftax prompt-opt candidate for the Todo Tool / Full Auto E2E lane."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

TRACK_NAME = "prompt_opt_1usd_gpt54_family"
TASK_NAME = "craftax"
BASE_MODEL = "Qwen/Qwen3.5-4B"
MODEL_LABEL = "qwen35-4b-full-auto-e2e"
OPTIMIZER_MODELS = ["gpt-5.4", "gpt-5.4-mini", "gpt-5.4-nano"]

TODO_SCRATCHPAD_REQUIREMENTS = [
    "Keep a tiny private todo list with exactly three items before the tool call.",
    "Refresh completed todo items every turn.",
    "Replace the stale target if the same movement loop repeats without new progress.",
    "Do not reveal the todo list or scratchpad in the final answer.",
]

FULL_AUTO_E2E_SYSTEM_PROMPT = (
    "You are a Craftax policy agent. Before choosing actions, keep a tiny "
    "private todo list with exactly three items: (1) the most urgent danger "
    "or blocker, (2) the next tile, object, or resource you should reach, and "
    "(3) the fallback action that breaks a loop if progress stalls. Refresh "
    "completed todo items every turn. If you repeat the same movement pattern "
    "without new progress or information, replace the stale target item before "
    "acting. Do not reveal the todo list to the user. Prefer early-game "
    "progression: move toward nearby trees or other gatherable resources, use "
    "`do` only when adjacent to a useful target, and avoid sleep, crafting, or "
    "inventory-only actions unless the local state clearly supports them. "
    "Choose a short 3 or 4 action batch that follows the first todo item and, "
    "when safe, ends next to a useful target for the next turn. Think "
    "carefully, then use the `craftax_interact` tool exactly once. Return 3 or "
    "4 valid full-Craftax actions unless the episode is already done. Use only "
    "the tool call as the final answer. Do not output JSON, prose, or a "
    "plain-text action list."
)


def todo_scratchpad_directive() -> str:
    return " ".join(TODO_SCRATCHPAD_REQUIREMENTS)


def candidate_config() -> dict[str, Any]:
    return {
        "track": TRACK_NAME,
        "task": TASK_NAME,
        "candidate_label": "Full Auto E2E",
        "base_model": BASE_MODEL,
        "optimizer_budget_usd": 1.0,
        "optimizer_models": OPTIMIZER_MODELS,
        "rollout": {
            "max_steps": 10,
            "max_concurrent_rollouts": 4,
            "rollout_concurrency": 4,
            "rollout_semaphore_limit": 2,
            "request_timeout_seconds": 900,
            "target_action_batch_size": 4,
            "min_action_batch_size": 3,
            "temperature": 0.0,
            "max_tokens": 3072,
            "enable_thinking": True,
            "thinking_budget_tokens": 2000,
        },
        "train_seeds": [10007, 10008, 10011, 10014],
        "eval_seeds": [10001, 10010, 10017, 10019],
        "prompt": {
            "component_name": "system_prompt",
            "seed_prompt": FULL_AUTO_E2E_SYSTEM_PROMPT,
            "todo_contract": todo_scratchpad_directive(),
        },
        "output": {
            "root_dir": "records/prompt_opt_1usd_gpt54_family/2026-04-11_full_auto_e2e"
        },
    }


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    text = config_path.read_text(encoding="utf-8")
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError(f"config must decode to an object: {config_path}")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/craftax_prompt_opt_qwen35_4b_full_auto_e2e.yaml",
        help="Path to the prompt-opt config file.",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="Optional output directory to record in the smoke summary.",
    )
    args = parser.parse_args(argv)

    loaded = load_config(args.config)
    summary = {
        "config_path": str(Path(args.config).expanduser()),
        "output_dir": args.output_dir,
        "loaded_config": loaded,
        "candidate_config": candidate_config(),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
