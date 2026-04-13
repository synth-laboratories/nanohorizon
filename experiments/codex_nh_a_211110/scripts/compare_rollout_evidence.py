from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
from types import SimpleNamespace
from unittest.mock import patch

from nanohorizon.craftax_core.modalities import RenderBundle, RenderMode
from nanohorizon.craftax_core.rollout import ACTION_NAME_TO_INDEX, run_rollout


ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = ROOT / "experiments" / "codex_nh_a_211110" / "results"


class FakeRunner:
    def __init__(self, *, seed: int) -> None:
        self.seed = int(seed)
        self.state = {"steps": 0}
        self.action_history: list[int] = []

    def reset(self):
        self.state = {"steps": 0}
        self.action_history = []
        return SimpleNamespace(
            done=False,
            render=RenderBundle(mode=RenderMode.TEXT, text=f"seed={self.seed}|step=0", pixels=None),
            reward=0.0,
            info={},
            step_index=0,
            episode_index=0,
        )

    def step_many(self, actions):
        outputs = []
        for action in actions:
            self.action_history.append(int(action))
            self.state["steps"] += 1
            reward = 1.0 if int(action) == ACTION_NAME_TO_INDEX["do"] else 0.0
            outputs.append(
                SimpleNamespace(
                    done=self.state["steps"] >= 2,
                    reward=reward,
                    render=RenderBundle(
                        mode=RenderMode.TEXT,
                        text=f"seed={self.seed}|step={self.state['steps']}",
                        pixels=None,
                    ),
                )
            )
        return outputs


def _fake_chat_completion(*, messages, **kwargs):  # type: ignore[no-untyped-def]
    del kwargs
    user_prompt = str(messages[-1]["content"])
    if "Recent rollout evidence:" in user_prompt:
        actions = ["move_right", "move_up", "do", "move_left"]
    else:
        actions = ["move_right", "move_up", "move_left", "move_down"]
    return {
        "choices": [
            {
                "message": {
                    "content": "",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "craftax_interact",
                                "arguments": {"actions_list": actions},
                            }
                        }
                    ],
                }
            }
        ]
    }


def _run_variant(*, seeds: list[int], enabled: bool) -> dict[str, object]:
    rewards: list[float] = []
    details: list[dict[str, object]] = []
    current_runner: dict[str, FakeRunner | None] = {"runner": None}

    def fake_make_runner(*, kind: str, seed: int, render_mode: RenderMode):  # type: ignore[no-untyped-def]
        del kind, render_mode
        runner = FakeRunner(seed=seed)
        current_runner["runner"] = runner
        return runner

    rollout_patch = patch("nanohorizon.craftax_core.rollout.make_runner", fake_make_runner)
    chat_patch = patch("nanohorizon.craftax_core.rollout._chat_completion", _fake_chat_completion)
    achievement_patch = patch(
        "nanohorizon.craftax_core.rollout.achievement_names_from_state",
        lambda state: ["collect_wood"]
        if current_runner["runner"] is not None
        and ACTION_NAME_TO_INDEX["do"] in current_runner["runner"].action_history
        else [],
    )
    evidence_patch = (
        patch("nanohorizon.craftax_core.rollout._rollout_evidence_prompt", lambda **_: "")
        if not enabled
        else None
    )

    patches = [rollout_patch, chat_patch, achievement_patch]
    if evidence_patch is not None:
        patches.append(evidence_patch)

    with patches[0], patches[1], patches[2]:
        if evidence_patch is not None:
            with evidence_patch:
                for seed in seeds:
                    result = run_rollout(
                        inference_url="http://example.test/v1/chat/completions",
                        model="demo",
                        api_key="",
                        seed=seed,
                        max_steps=2,
                        trace_correlation_id=f"trace-{seed}",
                        system_prompt="system",
                        target_action_batch_size=4,
                        min_action_batch_size=4,
                        request_logprobs=False,
                    )
                    reward = float(result["reward_info"]["outcome_objectives"]["reward"])
                    rewards.append(reward)
                    details.append(
                        {
                            "seed": seed,
                            "reward": reward,
                            "actions": result["metadata"]["action_history"],
                            "llm_call_count": result["metadata"]["llm_call_count"],
                        }
                    )
        else:
            for seed in seeds:
                result = run_rollout(
                    inference_url="http://example.test/v1/chat/completions",
                    model="demo",
                    api_key="",
                    seed=seed,
                    max_steps=2,
                    trace_correlation_id=f"trace-{seed}",
                    system_prompt="system",
                    target_action_batch_size=4,
                    min_action_batch_size=4,
                    request_logprobs=False,
                )
                reward = float(result["reward_info"]["outcome_objectives"]["reward"])
                rewards.append(reward)
                details.append(
                    {
                        "seed": seed,
                        "reward": reward,
                        "actions": result["metadata"]["action_history"],
                        "llm_call_count": result["metadata"]["llm_call_count"],
                    }
                )

    return {
        "mean_outcome_reward": mean(rewards) if rewards else 0.0,
        "details": details,
    }


def main() -> None:
    seeds = [10000, 10001, 10002, 10003]
    baseline = _run_variant(seeds=seeds, enabled=False)
    candidate = _run_variant(seeds=seeds, enabled=True)
    payload = {
        "seeds": seeds,
        "baseline": baseline,
        "candidate": candidate,
        "delta": float(candidate["mean_outcome_reward"]) - float(baseline["mean_outcome_reward"]),
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / "baseline_vs_candidate.json"
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
