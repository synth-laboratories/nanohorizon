#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import hashlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

from nanohorizon.craftax_core.modalities import CallableRenderer, RenderMode
from nanohorizon.craftax_core.metadata import PRIMARY_TOOL_NAME
from nanohorizon.craftax_core.runner import DeterministicCraftaxRunner


class FakeRandom:
    @staticmethod
    def PRNGKey(seed: int) -> tuple[int, int]:
        return (int(seed), 0)

    @staticmethod
    def split(key: tuple[int, int]) -> tuple[tuple[int, int], tuple[int, int]]:
        seed, counter = key
        return (seed, counter + 1), (seed, counter + 2)


class FakeTreeUtil:
    @staticmethod
    def tree_flatten(tree):
        if isinstance(tree, dict):
            keys = sorted(tree.keys())
            return [tree[key] for key in keys], tuple(keys)
        return [tree], type(tree).__name__


class FakeJax:
    random = FakeRandom()
    tree_util = FakeTreeUtil()
    Array = tuple


class ProxyEnv:
    default_params = {"terminal_position": 64}

    def reset(self, key, params=None):
        del params
        tick = int(key[1]) + int(int(key[0]) % 2 == 0)
        return None, SimpleNamespace(position=tick, rng_ticks=(tick,), achievements=())

    def step(self, key, state, action: int, params=None):
        limit = int((params or self.default_params).get("terminal_position", 64))
        tick = int(key[1]) + int(int(key[0]) % 2 == 0)
        next_position = int(state.position) + int(action) + tick
        achievements = list(state.achievements)
        if next_position >= 5 and "collect_wood" not in achievements:
            achievements.append("collect_wood")
        next_state = SimpleNamespace(
            position=next_position,
            rng_ticks=tuple([*state.rng_ticks, tick]),
            achievements=tuple(achievements),
        )
        reward = float(next_position)
        done = next_position >= limit
        info = {"position": next_position, "rng_tick": tick}
        return None, next_state, reward, done, info


def build_renderer() -> CallableRenderer:
    return CallableRenderer(
        text_fn=lambda state: f"position={state.position}|ticks={list(state.rng_ticks)}",
        pixels_fn=None,
        structured_fn=lambda state: {
            "position": state.position,
            "rng_ticks": list(state.rng_ticks),
            "achievements": list(state.achievements),
        },
    )


def build_runner(seed: int) -> DeterministicCraftaxRunner:
    import nanohorizon.craftax_core.checkpoint as checkpoint_module
    import nanohorizon.craftax_core.runner as runner_module

    fake_jax = FakeJax()
    runner_module.jax = fake_jax
    checkpoint_module.jax = fake_jax
    return DeterministicCraftaxRunner(
        env=ProxyEnv(),
        renderer=build_renderer(),
        seed=seed,
        render_mode=RenderMode.TEXT,
    )


def make_policy(mode: str) -> Callable[[list[dict[str, Any]]], dict[str, Any]]:
    def policy(messages: list[dict[str, Any]]) -> dict[str, Any]:
        user_message = ""
        for message in reversed(messages):
            if message.get("role") == "user":
                user_message = str(message.get("content") or "")
                break
        if mode == "candidate" and "reward_history" in user_message and '"reward_history":[]' not in user_message:
            action = "do"
        else:
            action = "move_right"
        return {
            "choices": [
                {
                    "message": {
                        "content": "",
                        "tool_calls": [
                            {
                                "function": {
                                    "name": PRIMARY_TOOL_NAME,
                                    "arguments": {"actions_list": [action]},
                                }
                            }
                        ],
                    }
                }
            ]
        }

    return policy


def run_rollout(seed: int, mode: str) -> dict[str, Any]:
    import nanohorizon.craftax_core.rollout as rollout

    original_make_runner = rollout.make_runner
    original_chat_completion = rollout._chat_completion
    original_achievement_names_from_state = rollout.achievement_names_from_state
    prompt_messages: list[list[dict[str, Any]]] = []

    def fake_chat_completion(**kwargs):  # type: ignore[no-untyped-def]
        prompt_messages.append(list(kwargs["messages"]))
        return make_policy(mode)(kwargs["messages"])

    try:
        rollout.make_runner = lambda **_: build_runner(seed)  # type: ignore[assignment]
        rollout._chat_completion = fake_chat_completion  # type: ignore[assignment]
        rollout.achievement_names_from_state = lambda state: list(getattr(state, "achievements", []))  # type: ignore[assignment]
        result = rollout.run_rollout(
            inference_url="http://example.test/v1/chat/completions",
            model="demo",
            api_key="",
            seed=seed,
            max_steps=2,
            trace_correlation_id=f"{mode}-{seed}",
            system_prompt="system",
            target_action_batch_size=1,
            min_action_batch_size=1,
            request_logprobs=False,
        )
    finally:
        rollout.make_runner = original_make_runner  # type: ignore[assignment]
        rollout._chat_completion = original_chat_completion  # type: ignore[assignment]
        rollout.achievement_names_from_state = original_achievement_names_from_state  # type: ignore[assignment]

    first_prompt = prompt_messages[0][1]["content"] if prompt_messages else ""
    second_prompt = prompt_messages[1][1]["content"] if len(prompt_messages) > 1 else ""
    return {
        "seed": seed,
        "mode": mode,
        "outcome_reward": result["reward_info"]["outcome_reward"],
        "native_env_reward_total": result["reward_info"]["details"]["native_env_reward_total"],
        "action_history": result["metadata"]["action_history"],
        "llm_call_count": result["metadata"]["llm_call_count"],
        "first_prompt_contains_structured_context": "Structured prompt context:" in first_prompt,
        "second_prompt_contains_reward_history": '"reward_history":[' in second_prompt,
        "first_prompt_hash": hashlib.sha256(first_prompt.encode("utf-8")).hexdigest(),
        "second_prompt_hash": hashlib.sha256(second_prompt.encode("utf-8")).hexdigest(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    seeds = [11, 11, 29, 29]
    baseline_rows = [run_rollout(seed, "baseline") for seed in seeds]
    candidate_rows = [run_rollout(seed, "candidate") for seed in seeds]

    payload = {
        "seeds": seeds,
        "baseline": baseline_rows,
        "candidate": candidate_rows,
        "summary": {
            "baseline_mean_outcome_reward": sum(row["outcome_reward"] for row in baseline_rows) / len(baseline_rows),
            "candidate_mean_outcome_reward": sum(row["outcome_reward"] for row in candidate_rows) / len(candidate_rows),
            "baseline_mean_native_env_reward_total": sum(
                row["native_env_reward_total"] for row in baseline_rows
            )
            / len(baseline_rows),
            "candidate_mean_native_env_reward_total": sum(
                row["native_env_reward_total"] for row in candidate_rows
            )
            / len(candidate_rows),
            "baseline_repeat_seed_hash_match": baseline_rows[0]["first_prompt_hash"] == baseline_rows[1]["first_prompt_hash"]
            and baseline_rows[2]["first_prompt_hash"] == baseline_rows[3]["first_prompt_hash"],
            "candidate_repeat_seed_hash_match": candidate_rows[0]["first_prompt_hash"] == candidate_rows[1]["first_prompt_hash"]
            and candidate_rows[2]["first_prompt_hash"] == candidate_rows[3]["first_prompt_hash"],
            "candidate_structured_context_present": all(
                row["first_prompt_contains_structured_context"] for row in candidate_rows
            ),
            "candidate_reward_history_present": any(
                row["second_prompt_contains_reward_history"] for row in candidate_rows
            ),
        },
    }

    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
