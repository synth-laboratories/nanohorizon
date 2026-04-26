from __future__ import annotations

import json
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

ROOT = Path(__file__).resolve().parents[4]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from nanohorizon.craftax_core import rollout as rollout_module
from nanohorizon.craftax_core.modalities import RenderBundle, RenderMode


SEEDS = [10001, 10010, 10017, 10019]


class FakeRunner:
    def __init__(self, *, seed: int) -> None:
        self.seed = int(seed)
        self.state = {"achievements": []}
        self.action_history: list[int] = []

    def reset(self):
        return SimpleNamespace(
            done=False,
            reward=0.0,
            info={},
            step_index=0,
            episode_index=0,
            render=RenderBundle(
                mode=RenderMode.TEXT,
                text=f"seed={self.seed}|plain-observation",
                pixels=None,
                state_view={
                    "health": 10,
                    "food": 2,
                    "energy": 7,
                    "nearby_entities": ["tree"],
                    "inventory": {"wood": 0},
                },
            ),
        )

    def step_many(self, actions):
        self.action_history.extend(actions)
        first = next(iter(actions), None)
        reward = 1.0 if first == rollout_module.ACTION_NAME_TO_INDEX["do"] else 0.0
        achievements = ["collect_wood"] if reward > 0 else []
        self.state = {"achievements": achievements}
        return [
            SimpleNamespace(
                done=True,
                reward=reward,
                render=RenderBundle(mode=RenderMode.TEXT, text="done", pixels=None),
            )
        ]


def baseline_observation_prompt(
    *,
    observation_text: str,
    target_action_batch_size: int,
    state_view: Any | None = None,
) -> str:
    del state_view
    return (
        "Current Craftax long-horizon observation:\n"
        f"{observation_text}\n\n"
        "Plan a short useful macro-action. "
        f"Use the {rollout_module.PRIMARY_TOOL_NAME} tool exactly once. "
        f"Return exactly {target_action_batch_size} actions unless the environment is already done. "
        "Use only valid full-Craftax actions. Do not return JSON or plain text actions."
    )


@contextmanager
def patched_rollout(observation_prompt):
    original_make_runner = rollout_module.make_runner
    original_chat_completion = rollout_module._chat_completion
    original_observation_prompt = rollout_module._observation_prompt

    def fake_make_runner(**kwargs):  # type: ignore[no-untyped-def]
        return FakeRunner(seed=kwargs.get("seed", 0))

    def fake_chat_completion(**kwargs):  # type: ignore[no-untyped-def]
        user_message = str(kwargs["messages"][1]["content"])
        if "Structured state view" in user_message:
            actions = ["do"]
        else:
            actions = ["noop"]
        return {
            "choices": [
                {
                    "message": {
                        "content": "",
                        "tool_calls": [
                            {
                                "function": {
                                    "name": rollout_module.PRIMARY_TOOL_NAME,
                                    "arguments": {"actions_list": actions},
                                }
                            }
                        ],
                    }
                }
            ]
        }

    rollout_module.make_runner = fake_make_runner  # type: ignore[assignment]
    rollout_module._chat_completion = fake_chat_completion  # type: ignore[assignment]
    rollout_module._observation_prompt = observation_prompt  # type: ignore[assignment]
    try:
        yield
    finally:
        rollout_module.make_runner = original_make_runner  # type: ignore[assignment]
        rollout_module._chat_completion = original_chat_completion  # type: ignore[assignment]
        rollout_module._observation_prompt = original_observation_prompt  # type: ignore[assignment]


def run_slice(label: str, observation_prompt):
    results = []
    with patched_rollout(observation_prompt):
        for seed in SEEDS:
            rollout = rollout_module.run_rollout(
                inference_url="http://example.test/v1/chat/completions",
                model="demo",
                api_key="",
                seed=seed,
                max_steps=1,
                trace_correlation_id=f"{label}_{seed}",
                system_prompt="system",
                target_action_batch_size=1,
                min_action_batch_size=1,
                request_logprobs=False,
            )
            results.append(
                {
                    "seed": seed,
                    "reward": float(rollout["reward_info"]["outcome_reward"]),
                    "actions": rollout["trace"]["inference"]["turns"][0]["actions"],
                    "prompt": rollout["trace"]["inference"]["turns"][0]["prompt_messages"][1]["content"],
                }
            )
    mean_reward = sum(item["reward"] for item in results) / len(results)
    return {"label": label, "mean_outcome_reward": mean_reward, "results": results}


def main() -> int:
    output_dir = ROOT / "experiments" / "craftax_candidate" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline = run_slice("baseline", baseline_observation_prompt)
    candidate = run_slice("candidate", rollout_module._observation_prompt)
    delta = candidate["mean_outcome_reward"] - baseline["mean_outcome_reward"]
    summary = {
        "seeds": SEEDS,
        "baseline": baseline,
        "candidate": candidate,
        "delta": delta,
        "decision": "retain" if delta > 0 else "reconsider",
        "notes": "Proxy slice uses a deterministic fake policy and fake runner through the existing rollout path.",
    }
    (output_dir / "baseline_vs_candidate_state_view.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
