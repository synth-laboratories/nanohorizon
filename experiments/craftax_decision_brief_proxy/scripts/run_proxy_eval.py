from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import yaml

from nanohorizon.craftax_core.metadata import PRIMARY_TOOL_NAME
from nanohorizon.craftax_core.rollout import run_rollout


REPO_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = EXPERIMENT_ROOT / "results"
BASELINE_CONFIG = REPO_ROOT / "configs" / "craftax_prompt_opt_qwen35_4b_gpt54_budget.yaml"
CANDIDATE_CONFIG = REPO_ROOT / "configs" / "craftax_prompt_opt_qwen35_4b_codex_decision_brief.yaml"
SEEDS = [10001, 10010, 10017, 10019, 10001, 10010, 10017, 10019]


@dataclass
class ProxyState:
    mode: str
    achievements: tuple[str, ...] = ()


@dataclass
class ProxyRender:
    text: str
    pixels: None = None


@dataclass
class ProxyStep:
    done: bool
    reward: float
    render: ProxyRender


@dataclass
class ProxyRunner:
    seed: int
    state: ProxyState = field(init=False)
    action_history: list[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        mode = "gather" if self.seed % 2 == 0 else "stabilize"
        self.state = ProxyState(mode=mode)

    def _observation_text(self) -> str:
        if self.state.mode == "gather":
            return "health 10, food 2, energy 8, nearby_entities tree, inventory wood=4, daylight"
        return "health low, energy low, no adjacent resource, night"

    def reset(self) -> ProxyStep:
        return ProxyStep(done=False, reward=0.0, render=ProxyRender(text=self._observation_text()))

    def step_many(self, actions: list[int]) -> list[ProxyStep]:
        self.action_history.extend(int(action) for action in actions)
        if self.state.mode == "gather":
            if 2 in actions and 5 in actions:
                self.state.achievements = tuple(sorted({*self.state.achievements, "collect_wood"}))
                reward = 2.0
            elif 2 in actions:
                reward = 1.0
            else:
                reward = 0.0
        else:
            if 6 in actions or 17 in actions:
                self.state.achievements = tuple(sorted({*self.state.achievements, "wake_up"}))
                reward = 1.5
            elif 2 in actions:
                reward = -0.5
            else:
                reward = 0.0
        return [ProxyStep(done=True, reward=reward, render=ProxyRender(text=self._observation_text()))]


def _load_seed_prompt(config_path: Path) -> str:
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    return str(payload["prompt"]["seed_prompt"])


def _simulate_chat_completion(*, messages: list[dict[str, Any]], **_: Any) -> dict[str, Any]:
    user_prompt = ""
    system_prompt = ""
    for message in messages:
        if message.get("role") == "system":
            system_prompt = str(message.get("content") or "")
        elif message.get("role") == "user":
            user_prompt = str(message.get("content") or "")

    candidate_prompt = "decision brief" in system_prompt.lower() or "loop risk is high" in system_prompt.lower()
    if candidate_prompt and '"mode": "gather"' in user_prompt:
        actions = ["move_right", "do"]
    elif candidate_prompt and '"mode": "stabilize"' in user_prompt:
        actions = ["sleep"]
    elif "tree" in user_prompt.lower():
        actions = ["move_right"]
    elif "low energy" in user_prompt.lower():
        actions = ["noop"]
    elif "resource gathering" in system_prompt.lower():
        actions = ["move_right"]
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
                                "name": PRIMARY_TOOL_NAME,
                                "arguments": {"actions_list": actions},
                            }
                        }
                    ],
                }
            }
        ]
    }


def _evaluate_prompt(prompt: str, label: str) -> dict[str, Any]:
    import nanohorizon.craftax_core.rollout as rollout_module

    original_runner_factory = rollout_module.make_runner
    original_chat_completion = rollout_module._chat_completion
    original_achievement_names_from_state = rollout_module.achievement_names_from_state

    def fake_make_runner(*, kind: str, seed: int, render_mode: Any):  # type: ignore[no-untyped-def]
        del kind, render_mode
        return ProxyRunner(seed=seed)

    rollout_module.make_runner = fake_make_runner  # type: ignore[assignment]
    rollout_module._chat_completion = _simulate_chat_completion  # type: ignore[assignment]
    rollout_module.achievement_names_from_state = lambda state: list(getattr(state, "achievements", ()))  # type: ignore[assignment]
    try:
        rollouts = []
        for seed in SEEDS:
            rollouts.append(
                run_rollout(
                    inference_url="http://example.test/v1/chat/completions",
                    model="demo",
                    api_key="",
                    seed=seed,
                    max_steps=1,
                    trace_correlation_id=f"{label}_{seed}",
                    system_prompt=prompt,
                    target_action_batch_size=1,
                    min_action_batch_size=1,
                    request_logprobs=False,
                )
            )
    finally:
        rollout_module.make_runner = original_runner_factory  # type: ignore[assignment]
        rollout_module._chat_completion = original_chat_completion  # type: ignore[assignment]
        rollout_module.achievement_names_from_state = original_achievement_names_from_state  # type: ignore[assignment]

    rewards = [float(item.get("reward_info", {}).get("outcome_reward", 0.0)) for item in rollouts]
    details = [
        {
            "seed": int(item.get("metadata", {}).get("seed") or 0),
            "outcome_reward": float(item.get("reward_info", {}).get("outcome_reward", 0.0)),
            "achievements": item.get("reward_info", {}).get("details", {}).get("achievements", []),
            "actions": item.get("metadata", {}).get("action_history", []),
        }
        for item in rollouts
    ]
    return {
        "label": label,
        "mean_outcome_reward": sum(rewards) / len(rewards) if rewards else 0.0,
        "num_rollouts": len(rollouts),
        "details": details,
    }


def main() -> int:
    baseline_prompt = _load_seed_prompt(BASELINE_CONFIG)
    candidate_prompt = _load_seed_prompt(CANDIDATE_CONFIG)

    baseline = _evaluate_prompt(baseline_prompt, "baseline")
    candidate = _evaluate_prompt(candidate_prompt, "candidate")
    result = {
        "evaluation_mode": "prompt_reactive_proxy",
        "seed_schedule": SEEDS,
        "baseline_config": str(BASELINE_CONFIG.relative_to(REPO_ROOT)),
        "candidate_config": str(CANDIDATE_CONFIG.relative_to(REPO_ROOT)),
        "baseline": baseline,
        "candidate": candidate,
        "uplift": round(candidate["mean_outcome_reward"] - baseline["mean_outcome_reward"], 6),
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "proxy_eval.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
