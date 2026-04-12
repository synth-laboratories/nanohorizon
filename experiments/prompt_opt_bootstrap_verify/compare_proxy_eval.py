from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from nanohorizon.craftax_core.metadata import PRIMARY_TOOL_NAME
from nanohorizon.craftax_core.rollout import run_rollout
import nanohorizon.craftax_core.rollout as rollout_module


@dataclass(frozen=True)
class FakeState:
    position: int
    achievements: tuple[str, ...] = ()


@dataclass(frozen=True)
class FakeRender:
    text: str
    pixels: Any = None


@dataclass(frozen=True)
class FakeStep:
    render: FakeRender
    reward: float
    done: bool
    info: dict[str, Any]


class FakeRunner:
    def __init__(self, *, seed: int) -> None:
        self.seed = int(seed)
        self.state = FakeState(position=int(seed % 2))
        self.action_history: list[int] = []
        self._done = False

    def reset(self) -> FakeStep:
        self.state = FakeState(position=int(self.seed % 2))
        self.action_history = []
        self._done = False
        return FakeStep(
            render=FakeRender(text=f"position={self.state.position}|achievements={list(self.state.achievements)}"),
            reward=0.0,
            done=False,
            info={"position": self.state.position},
        )

    def step_many(self, actions: list[int]) -> list[FakeStep]:
        outputs: list[FakeStep] = []
        for action in actions:
            if self._done:
                break
            self.action_history.append(int(action))
            delta = 1 if int(action) != 0 else 0
            next_position = self.state.position + delta
            achievements = list(self.state.achievements)
            if next_position >= 2 and "collect_wood" not in achievements:
                achievements.append("collect_wood")
            if next_position >= 4 and "collect_sapling" not in achievements:
                achievements.append("collect_sapling")
            self.state = FakeState(position=next_position, achievements=tuple(achievements))
            self._done = next_position >= 5
            outputs.append(
                FakeStep(
                    render=FakeRender(
                        text=f"position={self.state.position}|achievements={list(self.state.achievements)}"
                    ),
                    reward=float(len(achievements)),
                    done=self._done,
                    info={"position": self.state.position},
                )
            )
        return outputs


def load_prompt(path: Path) -> str:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    return str(payload["prompt"]["seed_prompt"])


def make_proxy_chat_completion(system_prompt: str):
    def _chat_completion(**kwargs):  # type: ignore[no-untyped-def]
        messages = kwargs.get("messages") or []
        prompt_text = "\n".join(
            str(message.get("content") or "") for message in messages if isinstance(message, dict)
        )
        if "Prefer 3-action batches by default" in system_prompt:
            actions = ["move_right", "move_up", "do", "move_down"]
        elif "follow the first todo item" in system_prompt:
            actions = ["move_right", "move_up", "noop"]
        else:
            actions = ["noop", "move_left", "move_right"]
        if "invalid" in prompt_text.lower():
            actions = ["move_right", "move_right", "move_right"]
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

    return _chat_completion


def run_slice(prompt_path: Path, seeds: list[int]) -> dict[str, Any]:
    system_prompt = load_prompt(prompt_path)
    original_make_runner = rollout_module.make_runner
    original_chat_completion = rollout_module._chat_completion
    original_achievements = rollout_module.achievement_names_from_state

    def fake_achievements(state):  # type: ignore[no-untyped-def]
        return list(getattr(state, "achievements", ()))

    rollout_module._chat_completion = make_proxy_chat_completion(system_prompt)
    rollout_module.achievement_names_from_state = fake_achievements

    rewards: list[float] = []
    details: list[dict[str, Any]] = []
    try:
        for seed in seeds:
            rollout_module.make_runner = lambda **kwargs: FakeRunner(seed=int(kwargs.get("seed", seed)))
            result = run_rollout(
                inference_url="http://proxy.test/v1/chat/completions",
                model="proxy",
                api_key="",
                seed=int(seed),
                max_steps=1,
                trace_correlation_id=f"proxy-{seed}",
                system_prompt=system_prompt,
                target_action_batch_size=4,
                min_action_batch_size=3,
                request_logprobs=False,
            )
            reward = float(result["reward_info"]["outcome_reward"])
            rewards.append(reward)
            details.append(
                {
                    "seed": int(seed),
                    "reward": reward,
                    "achievements": result["reward_info"]["details"]["achievements"],
                    "actions": result["metadata"]["action_history"],
                }
            )
    finally:
        rollout_module.make_runner = original_make_runner
        rollout_module._chat_completion = original_chat_completion
        rollout_module.achievement_names_from_state = original_achievements

    return {
        "prompt_path": str(prompt_path),
        "mean_outcome_reward": sum(rewards) / len(rewards) if rewards else 0.0,
        "rewards": rewards,
        "details": details,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--baseline",
        default="configs/craftax_prompt_opt_qwen35_4b_codex_todo_refresh_gate.yaml",
    )
    parser.add_argument(
        "--candidate",
        default="configs/craftax_prompt_opt_qwen35_4b_codex_bootstrap_verify.yaml",
    )
    parser.add_argument(
        "--seeds",
        default="10000,10001,10002,10003,10004,10005",
    )
    parser.add_argument("--output", default="experiments/prompt_opt_bootstrap_verify/results/proxy_eval_summary.json")
    args = parser.parse_args()

    seeds = [int(item.strip()) for item in str(args.seeds).split(",") if item.strip()]
    baseline = run_slice(Path(args.baseline), seeds)
    candidate = run_slice(Path(args.candidate), seeds)
    summary = {
        "baseline": baseline,
        "candidate": candidate,
        "delta_mean_outcome_reward": candidate["mean_outcome_reward"] - baseline["mean_outcome_reward"],
        "seeds": seeds,
        "note": "Proxy only. Uses existing run_rollout path with a deterministic fake runner and chat stub because no live Craftax model endpoint was available in this workspace.",
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
