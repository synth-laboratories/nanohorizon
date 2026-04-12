from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from nanohorizon.craftax_core.metadata import PRIMARY_TOOL_NAME
import nanohorizon.craftax_core.rollout as rollout


@dataclass(frozen=True)
class FakeRender:
    text: str
    pixels: object | None = None


@dataclass(frozen=True)
class FakeState:
    achievements: tuple[str, ...]


@dataclass
class FakeStep:
    text: str
    done: bool
    reward: float = 0.0

    def __post_init__(self) -> None:
        self.render = FakeRender(self.text)


class FakeRunner:
    def __init__(self, seed: int) -> None:
        self.seed = int(seed)
        self.state = FakeState(())
        self.action_history: list[int] = []
        self.turn_index = 0

    def reset(self):
        self.state = FakeState(())
        self.turn_index = 0
        self.action_history = []
        return FakeStep(text=f"obs0 seed={self.seed}", done=False)

    def step_many(self, actions):
        self.action_history.extend(int(action) for action in actions)
        self.turn_index += 1
        action = int(actions[0]) if actions else 0
        achievements = list(self.state.achievements)
        if self.turn_index == 1 and action == 2 and "collect_wood" not in achievements:
            achievements.append("collect_wood")
        if self.turn_index >= 2 and action == 3 and "collect_sapling" not in achievements:
            achievements.append("collect_sapling")
        if "collect_wood" not in achievements:
            achievements.append("collect_wood")
        self.state = FakeState(tuple(achievements))
        return [
            FakeStep(
                text=f"obs{self.turn_index} seed={self.seed}",
                done=self.turn_index >= 2,
                reward=float(action),
            )
        ]


def _fake_chat_completion(*, messages, **kwargs):  # type: ignore[no-untyped-def]
    del kwargs
    user_prompt = str(messages[1]["content"])
    if "Recent turn history" in user_prompt and "turn 0" in user_prompt:
        actions = ["move_up"]
    else:
        actions = ["move_right"]
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


def _achievement_names_from_state(state: FakeState) -> list[str]:
    return [str(item) for item in state.achievements]


def evaluate_variant(*, use_history: bool, seeds: list[int]) -> dict[str, object]:
    original_make_runner = rollout.make_runner
    original_chat_completion = rollout._chat_completion
    original_history_prompt = rollout._recent_turn_history_prompt
    original_achievement_names_from_state = rollout.achievement_names_from_state
    try:
        rollout.make_runner = lambda **kwargs: FakeRunner(int(kwargs.get("seed", 0)))  # type: ignore[assignment]
        rollout._chat_completion = _fake_chat_completion  # type: ignore[assignment]
        if not use_history:
            rollout._recent_turn_history_prompt = lambda turns, window_size=4: ""  # type: ignore[assignment]
        rollout.achievement_names_from_state = _achievement_names_from_state  # type: ignore[assignment]

        details: list[dict[str, object]] = []
        rewards: list[float] = []
        native_rewards: list[float] = []
        llm_calls: list[int] = []
        for seed in seeds:
            result = rollout.run_rollout(
                inference_url="http://example.test/v1/chat/completions",
                model="demo",
                api_key="",
                seed=int(seed),
                max_steps=2,
                trace_correlation_id=f"{'candidate' if use_history else 'baseline'}_{seed}",
                system_prompt="system",
                target_action_batch_size=1,
                min_action_batch_size=1,
                request_logprobs=False,
            )
            reward_info = result["reward_info"]
            details.append(
                {
                    "seed": int(seed),
                    "outcome_reward": float(reward_info["outcome_reward"]),
                    "native_env_reward_total": float(
                        reward_info["outcome_objectives"]["native_env_reward_total"]
                    ),
                    "llm_call_count": int(reward_info["details"]["llm_call_count"]),
                    "achievements": list(reward_info["details"]["achievements"]),
                }
            )
            rewards.append(float(reward_info["outcome_reward"]))
            native_rewards.append(float(reward_info["outcome_objectives"]["native_env_reward_total"]))
            llm_calls.append(int(reward_info["details"]["llm_call_count"]))

        return {
            "use_history": bool(use_history),
            "seeds": list(seeds),
            "mean_outcome_reward": sum(rewards) / len(rewards) if rewards else 0.0,
            "mean_native_env_reward_total": (
                sum(native_rewards) / len(native_rewards) if native_rewards else 0.0
            ),
            "mean_llm_calls": sum(llm_calls) / len(llm_calls) if llm_calls else 0.0,
            "details": details,
        }
    finally:
        rollout.make_runner = original_make_runner  # type: ignore[assignment]
        rollout._chat_completion = original_chat_completion  # type: ignore[assignment]
        rollout._recent_turn_history_prompt = original_history_prompt  # type: ignore[assignment]
        rollout.achievement_names_from_state = original_achievement_names_from_state  # type: ignore[assignment]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[10000, 10001, 10002, 10003])
    args = parser.parse_args()

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline = evaluate_variant(use_history=False, seeds=list(args.seeds))
    candidate = evaluate_variant(use_history=True, seeds=list(args.seeds))
    comparison = {
        "baseline": baseline,
        "candidate": candidate,
        "delta": {
            "mean_outcome_reward": candidate["mean_outcome_reward"] - baseline["mean_outcome_reward"],
            "mean_native_env_reward_total": (
                candidate["mean_native_env_reward_total"] - baseline["mean_native_env_reward_total"]
            ),
            "mean_llm_calls": candidate["mean_llm_calls"] - baseline["mean_llm_calls"],
        },
    }
    (output_dir / "comparison.json").write_text(json.dumps(comparison, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(comparison, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
