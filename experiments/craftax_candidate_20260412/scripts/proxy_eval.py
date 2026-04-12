from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nanohorizon.craftax_core.metadata import PRIMARY_TOOL_NAME
from nanohorizon.craftax_core.modalities import RenderBundle, RenderMode
from nanohorizon.craftax_core.rollout import run_rollout


@dataclass(frozen=True)
class FakeState:
    position: int
    ticks: tuple[int, ...]
    achievements: tuple[str, ...] = ()


class FakeRender:
    def __init__(self, text: str, state_view: dict[str, Any]):
        self.text = text
        self.pixels = None
        self.state_view = state_view


class FakeStep:
    def __init__(self, *, position: int, ticks: tuple[int, ...], achievements: tuple[str, ...], done: bool, reward: float):
        self.done = done
        self.reward = reward
        self.render = FakeRender(
            text=f"position={position}|ticks={list(ticks)}|achievements={list(achievements)}",
            state_view={"position": position, "ticks": list(ticks), "achievements": list(achievements)},
        )


class ProxyRunner:
    default_params = {"terminal_position": 70}

    def __init__(self, *, seed: int):
        self.seed = int(seed)
        self.state = FakeState(position=int(seed) % 3, ticks=(int(seed) % 3,))
        self.action_history: list[int] = []
        self._step_index = 0

    def reset(self):
        self.state = FakeState(position=int(self.seed) % 3, ticks=(int(self.seed) % 3,))
        self.action_history = []
        self._step_index = 0
        return FakeStep(position=self.state.position, ticks=self.state.ticks, achievements=self.state.achievements, done=False, reward=0.0)

    def step_many(self, actions):
        outputs: list[FakeStep] = []
        for action in actions:
            self.action_history.append(int(action))
            self._step_index += 1
            tick = 1 + ((self.seed + self._step_index) % 2)
            next_position = int(self.state.position) + int(action) + tick
            achievements = list(self.state.achievements)
            if next_position >= 8 and "collect_wood" not in achievements:
                achievements.append("collect_wood")
            if next_position >= 35 and "collect_sapling" not in achievements:
                achievements.append("collect_sapling")
            self.state = FakeState(position=next_position, ticks=(*self.state.ticks, tick), achievements=tuple(achievements))
            done = next_position >= int(self.default_params["terminal_position"])
            outputs.append(
                FakeStep(
                    position=next_position,
                    ticks=self.state.ticks,
                    achievements=self.state.achievements,
                    done=done,
                    reward=float(next_position),
                )
            )
            if done:
                break
        return outputs


def _tool_call(actions: list[str]) -> dict[str, Any]:
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


def _make_policy_response(messages: list[dict[str, Any]]) -> dict[str, Any]:
    user_prompt = ""
    if messages:
        last = messages[-1]
        if isinstance(last, dict):
            user_prompt = str(last.get("content") or "")
    if "Recent action history:" in user_prompt:
        actions = ["enchant_bow"] * 5
    else:
        actions = ["move_right"] * 5
    return _tool_call(actions)


def _run_slice(*, baseline: bool, seeds: list[int]) -> dict[str, Any]:
    import nanohorizon.craftax_core.rollout as rollout_mod

    original_observation_prompt = rollout_mod._observation_prompt

    def fake_make_runner(*, seed: int, render_mode: RenderMode, **kwargs):  # type: ignore[no-untyped-def]
        del render_mode, kwargs
        return ProxyRunner(seed=seed)

    def fake_chat_completion(**kwargs):  # type: ignore[no-untyped-def]
        return _make_policy_response(list(kwargs["messages"]))

    def baseline_prompt(*, observation_text: str, target_action_batch_size: int, state_view: Any | None = None, reward_history: Any | None = None) -> str:
        del reward_history
        return original_observation_prompt(
            observation_text=observation_text,
            target_action_batch_size=target_action_batch_size,
            state_view=state_view,
            reward_history=None,
        )

    rollout_mod.make_runner = fake_make_runner  # type: ignore[assignment]
    rollout_mod._chat_completion = fake_chat_completion  # type: ignore[assignment]
    rollout_mod.achievement_names_from_state = lambda state: list(getattr(state, "achievements", ()))  # type: ignore[assignment]
    if baseline:
        rollout_mod._observation_prompt = baseline_prompt  # type: ignore[assignment]

    results = []
    for seed in seeds:
        result = run_rollout(
            inference_url="http://mock.local/v1/chat/completions",
            model="mock-policy",
            api_key="",
            seed=seed,
            max_steps=3,
            trace_correlation_id=f"proxy-{seed}",
            system_prompt="system",
            target_action_batch_size=5,
            min_action_batch_size=1,
            request_logprobs=False,
        )
        results.append(result)

    rollout_mod._observation_prompt = original_observation_prompt  # type: ignore[assignment]
    return {
        "mean_outcome_reward": sum(float(item["reward_info"]["outcome_reward"]) for item in results) / len(results),
        "mean_native_env_reward_total": sum(float(item["reward_info"]["outcome_objectives"]["native_env_reward_total"]) for item in results) / len(results),
        "mean_llm_calls_per_rollout": sum(float(item["metadata"]["llm_call_count"]) for item in results) / len(results),
        "details": [
            {
                "seed": seed,
                "outcome_reward": float(result["reward_info"]["outcome_reward"]),
                "native_env_reward_total": float(result["reward_info"]["outcome_objectives"]["native_env_reward_total"]),
                "achievements": list(result["metadata"]["achievements"]),
                "first_prompt": str(result["trace"]["inference"]["turns"][0]["prompt_messages"][1]["content"]),
            }
            for seed, result in zip(seeds, results, strict=True)
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--num-seeds", type=int, default=4)
    args = parser.parse_args()

    seeds = [int(args.seed_start) + idx for idx in range(int(args.num_seeds))]
    baseline = _run_slice(baseline=True, seeds=seeds)
    candidate = _run_slice(baseline=False, seeds=seeds)
    comparison = {
        "seeds": seeds,
        "baseline": baseline,
        "candidate": candidate,
        "delta_mean_outcome_reward": candidate["mean_outcome_reward"] - baseline["mean_outcome_reward"],
        "delta_mean_native_env_reward_total": candidate["mean_native_env_reward_total"] - baseline["mean_native_env_reward_total"],
        "delta_mean_llm_calls_per_rollout": candidate["mean_llm_calls_per_rollout"] - baseline["mean_llm_calls_per_rollout"],
        "decision": "retain" if candidate["mean_outcome_reward"] >= baseline["mean_outcome_reward"] else "revert",
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "proxy_eval_comparison.json").write_text(json.dumps(comparison, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(comparison, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
