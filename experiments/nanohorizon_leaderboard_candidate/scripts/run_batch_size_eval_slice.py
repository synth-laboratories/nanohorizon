#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from nanohorizon.craftax_core.metadata import PRIMARY_TOOL_NAME
from nanohorizon.craftax_core import rollout as rollout_module


ACTION_NAME_TO_INDEX = {
    "move_right": 1,
    "move_left": 2,
    "move_up": 3,
    "move_down": 4,
    "do": 5,
    "noop": 0,
}

CURRENT_ACTION_COUNT = 4


@dataclass
class FakeStep:
    done: bool
    reward: float
    position: int

    @property
    def render(self) -> SimpleNamespace:
        return SimpleNamespace(text=f"position={self.position}", pixels=None)


class FakeRunner:
    def __init__(self, seed: int) -> None:
        self.seed = int(seed)
        self.position = 0
        self.state = {"position": 0, "seed": self.seed}
        self.action_history: list[int] = []

    def reset(self) -> FakeStep:
        return FakeStep(done=False, reward=0.0, position=self.position)

    def step_many(self, actions: list[int]) -> list[FakeStep]:
        outputs: list[FakeStep] = []
        for action in actions:
            self.action_history.append(int(action))
            if int(action) == ACTION_NAME_TO_INDEX["move_right"]:
                self.position += 1
            self.state["position"] = self.position
            self.state["last_batch_size"] = len(actions)
            reward = 1.0 if len(actions) == 4 and action == actions[-1] else 0.0
            outputs.append(FakeStep(done=False, reward=reward, position=self.position))
        return outputs


def _fake_make_runner(*, kind: str, seed: int, render_mode: Any) -> FakeRunner:
    del kind, render_mode
    return FakeRunner(seed=seed)


def _fake_achievement_names_from_state(state: Any) -> list[str]:
    batch_size = state.get("last_batch_size", 0) if isinstance(state, dict) else getattr(state, "last_batch_size", 0)
    return ["collect_wood"] if int(batch_size) == 4 else []


def _fake_chat_completion(**kwargs: Any) -> dict[str, Any]:
    del kwargs
    action_bank = [
        "move_right",
        "move_left",
        "move_up",
        "move_down",
        "do",
        "noop",
        "sleep",
        "place_stone",
        "place_table",
    ]
    actions = action_bank[: max(1, int(CURRENT_ACTION_COUNT))]
    return {
        "choices": [
            {
                "message": {
                    "content": "",
                    "tool_calls": [
                        {
                            "function": {
                                "name": PRIMARY_TOOL_NAME,
                                "arguments": {"actions_list": actions},
                            }
                        }
                    ],
                }
            }
        ]
    }


def _run_case(*, target_action_batch_size: int, min_action_batch_size: int, seeds: list[int]) -> dict[str, Any]:
    global CURRENT_ACTION_COUNT
    original_make_runner = rollout_module.make_runner
    original_chat_completion = rollout_module._chat_completion
    original_achievement_names_from_state = rollout_module.achievement_names_from_state
    try:
        CURRENT_ACTION_COUNT = target_action_batch_size
        rollout_module.make_runner = _fake_make_runner  # type: ignore[assignment]
        rollout_module._chat_completion = _fake_chat_completion  # type: ignore[assignment]
        rollout_module.achievement_names_from_state = _fake_achievement_names_from_state  # type: ignore[assignment]
        rollouts = [
            rollout_module.run_rollout_request(
                {
                    "trace_correlation_id": f"batch_{target_action_batch_size}_{seed}",
                    "env": {"seed": seed, "config": {"env_kind": "full", "max_steps": 1, "episode_max_steps": 1}},
                    "policy": {
                        "config": {
                            "inference_url": "http://example.test/v1/chat/completions",
                            "model": "demo",
                            "api_key": "",
                            "system_prompt": "demo",
                            "target_action_batch_size": target_action_batch_size,
                            "min_action_batch_size": min_action_batch_size,
                        }
                    },
                }
            )
            for seed in seeds
        ]
    finally:
        rollout_module.make_runner = original_make_runner  # type: ignore[assignment]
        rollout_module._chat_completion = original_chat_completion  # type: ignore[assignment]
        rollout_module.achievement_names_from_state = original_achievement_names_from_state  # type: ignore[assignment]

    rewards = [float(item["reward_info"]["outcome_objectives"]["unique_achievements"]) for item in rollouts]
    llm_calls = [int(item["metadata"]["llm_call_count"]) for item in rollouts]
    invalid_parses = sum(
        1
        for item in rollouts
        for turn in item["trace"]["inference"]["turns"]
        if bool(turn.get("invalid_parse"))
    )
    return {
        "target_action_batch_size": target_action_batch_size,
        "min_action_batch_size": min_action_batch_size,
        "seeds": seeds,
        "mean_outcome_reward": sum(rewards) / len(rewards),
        "mean_llm_call_count": sum(llm_calls) / len(llm_calls),
        "invalid_parse_turns": invalid_parses,
        "per_seed": [
            {
                "seed": seed,
                "outcome_reward": float(item["reward_info"]["outcome_objectives"]["unique_achievements"]),
                "llm_call_count": int(item["metadata"]["llm_call_count"]),
                "actions": item["metadata"]["action_history"],
                "achievements": item["metadata"]["achievements"],
            }
            for seed, item in zip(seeds, rollouts, strict=True)
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[101, 102, 103])
    args = parser.parse_args()

    baseline = _run_case(target_action_batch_size=8, min_action_batch_size=5, seeds=args.seeds)
    candidate = _run_case(target_action_batch_size=4, min_action_batch_size=3, seeds=args.seeds)
    comparison = {
        "reward_delta": candidate["mean_outcome_reward"] - baseline["mean_outcome_reward"],
        "llm_call_delta": candidate["mean_llm_call_count"] - baseline["mean_llm_call_count"],
        "invalid_parse_delta": candidate["invalid_parse_turns"] - baseline["invalid_parse_turns"],
    }
    payload = {
        "baseline": baseline,
        "candidate": candidate,
        "comparison": comparison,
    }
    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
