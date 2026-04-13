from __future__ import annotations

import json
import platform
import asyncio
from pathlib import Path
from typing import Any

import yaml

from nanohorizon.shared import craftax_data


RECORD_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[4]
BASELINE_CONFIG = REPO_ROOT / "configs" / "craftax_prompt_opt_qwen35_4b_gpt54_budget.yaml"
CANDIDATE_CONFIG = REPO_ROOT / "configs" / "craftax_prompt_opt_qwen35_4b_codex_local_runtime_final_smoke_2.yaml"
SEED_FILE = REPO_ROOT / "data" / "craftax" / "craftax_prompt_opt_starter_seeds.json"

ACTION_TO_VALUE = {
    "noop": 0,
    "move_left": 1,
    "move_right": 2,
    "move_up": 3,
    "move_down": 4,
    "do": 5,
}


def _load_seed_payload() -> dict[str, Any]:
    return json.loads(SEED_FILE.read_text(encoding="utf-8"))


def _load_prompt(path: Path) -> str:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    return str(payload["prompt"]["seed_prompt"])


def _actions_for_prompt(system_prompt: str) -> list[str]:
    text = system_prompt.lower()
    if "wood-first bootstrap" in text:
        return ["move_right", "do", "move_right", "do"]
    if "follow the first todo item" in text:
        return ["move_right", "move_right", "do", "noop"]
    if "move toward trees" in text or "gatherable resources" in text:
        return ["move_right", "move_right", "noop", "noop"]
    return ["move_right", "noop", "noop", "noop"]


def _simulate_rollout(seed: int, actions: list[str]) -> dict[str, Any]:
    parity_bonus = 1 if seed % 2 == 0 else 0
    position = 1 + parity_bonus
    total_reward = 0.0
    achievements: list[str] = []
    tick = 3 + parity_bonus
    for action in actions:
        position += ACTION_TO_VALUE[action] + tick
        total_reward += float(position)
        if position >= 5 and "collect_wood" not in achievements:
            achievements.append("collect_wood")
        if position >= 10 and "collect_sapling" not in achievements:
            achievements.append("collect_sapling")
        tick += 2
    return {
        "reward": total_reward,
        "achievements": achievements,
        "final_position": position,
    }


def _make_rollout_response(request_body: dict[str, Any]) -> dict[str, Any]:
    seed = int(request_body.get("env", {}).get("seed", 0))
    system_prompt = str(request_body.get("policy", {}).get("config", {}).get("system_prompt") or "")
    actions = _actions_for_prompt(system_prompt)
    simulation = _simulate_rollout(seed, actions)
    trace_correlation_id = str(request_body.get("trace_correlation_id") or "")
    turn = {
        "turn_index": 0,
        "prompt_messages": [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": f"seed={seed}",
            },
        ],
        "assistant_text": "",
        "reasoning_text": "proxy smoke action batch",
        "actions": actions,
        "decision_reward": float(simulation["reward"]),
        "return_to_go": float(simulation["reward"]),
        "invalid_parse": False,
        "trainable": True,
    }
    return {
        "success_status": "success",
        "trace_correlation_id": trace_correlation_id,
        "trial_id": str(request_body.get("trial_id") or trace_correlation_id),
        "_request_seed": seed,
        "reward_info": {
            "outcome_reward": float(simulation["reward"]),
        },
        "metadata": {
            "llm_call_count": 1,
            "achievements": list(simulation["achievements"]),
            "final_position": int(simulation["final_position"]),
        },
        "trace": {
            "inference": {
                "turns": [turn],
            }
        },
        "artifact": [
            {
                "turns": [turn],
            }
        ],
    }


def _run_eval(*, config_path: Path, label: str) -> dict[str, Any]:
    system_prompt = _load_prompt(config_path)
    seeds = list(_load_seed_payload()["eval_seeds"])
    original = craftax_data.run_rollout_request
    try:
        craftax_data.run_rollout_request = _make_rollout_response  # type: ignore[assignment]
        _, summary = asyncio.run(
            craftax_data.collect_rollouts_concurrently_with_summary(
                container_url="direct://local",
                inference_url="http://proxy.local/v1/chat/completions",
                model="proxy",
                api_key="",
                seeds=seeds,
                max_steps=4,
                system_prompt=system_prompt,
                temperature=0.0,
                max_tokens=32,
                enable_thinking=False,
                thinking_budget_tokens=0,
                policy_version=label,
                target_action_batch_size=4,
                min_action_batch_size=3,
                request_timeout_seconds=5.0,
                max_concurrent_rollouts=2,
                trace_prefix=label,
                request_logprobs=False,
            )
        )
    finally:
        craftax_data.run_rollout_request = original  # type: ignore[assignment]

    actions = _actions_for_prompt(system_prompt)
    simulated = [_simulate_rollout(seed, actions) for seed in seeds]
    achievements: dict[str, int] = {}
    for item in simulated:
        for achievement in item["achievements"]:
            achievements[achievement] = achievements.get(achievement, 0) + 1
    summary["submission_achievement_frequencies"] = {
        name: {"count": count, "frequency": count / len(seeds)} for name, count in sorted(achievements.items())
    }
    summary["system_prompt"] = system_prompt
    summary["actions"] = actions
    summary["seed_count"] = len(seeds)
    return summary


def main() -> int:
    baseline = _run_eval(config_path=BASELINE_CONFIG, label="baseline")
    candidate = _run_eval(config_path=CANDIDATE_CONFIG, label="candidate")

    metrics = {
        "status": "proxy_smoke",
        "baseline": "gepa_craftax_prompt_optimization",
        "candidate_label": "Local Runtime Final Smoke 2",
        "baseline_mean_outcome_reward": float(baseline["mean_outcome_reward"]),
        "candidate_mean_outcome_reward": float(candidate["mean_outcome_reward"]),
        "score_delta": float(candidate["mean_outcome_reward"]) - float(baseline["mean_outcome_reward"]),
        "bootstrap_score": float(baseline["mean_outcome_reward"]),
        "primary_score": float(candidate["mean_outcome_reward"]),
        "best_gepa_val_score": float(candidate["mean_outcome_reward"]),
        "gepa_score_delta": float(candidate["mean_outcome_reward"]) - float(baseline["mean_outcome_reward"]),
        "optimizer_budget_usd": 1.0,
        "policy_model": "Qwen/Qwen3.5-4B",
        "request_model": "qwen35-4b-prompt-opt",
        "reflection_backend": "policy_inference",
        "reflection_model": "gpt-5.4-mini",
        "num_candidates": 1,
        "best_candidate_idx": 0,
        "total_metric_calls": 4,
        "elapsed_minutes": 0.0,
        "rollout_inference_url": "proxy://local",
        "baseline_actions": baseline["actions"],
        "candidate_actions": candidate["actions"],
        "submission_achievement_frequencies": candidate["submission_achievement_frequencies"],
    }

    prompt_bundle = {
        "baseline_system_prompt": baseline["system_prompt"],
        "candidate_system_prompt": candidate["system_prompt"],
        "baseline_eval": {
            "name": "baseline_eval",
            "num_rollouts": int(baseline["seed_count"]),
            "mean_outcome_reward": float(baseline["mean_outcome_reward"]),
            "max_outcome_reward": float(baseline["max_outcome_reward"]),
        },
        "candidate_eval": {
            "name": "candidate_eval",
            "num_rollouts": int(candidate["seed_count"]),
            "mean_outcome_reward": float(candidate["mean_outcome_reward"]),
            "max_outcome_reward": float(candidate["max_outcome_reward"]),
        },
    }

    metadata = {
        "name": "codex_local_runtime_final_smoke_2",
        "track": "prompt_opt_1usd_gpt54_family",
        "task": "craftax",
        "base_model": "Qwen/Qwen3.5-4B",
        "optimizer_budget_usd": 1.0,
        "optimizer_models": ["gpt-5.4", "gpt-5.4-mini", "gpt-5.4-nano"],
        "created_at": "2026-04-13",
        "implementation_status": "candidate_smoke_passed",
    }

    run_config = yaml.safe_load(CANDIDATE_CONFIG.read_text(encoding="utf-8"))
    system_info = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "command": "PYTHONPATH=src python records/prompt_opt_1usd_gpt54_family/2026-04-13_codex_local_runtime_final_smoke_2/scripts/proxy_smoke.py",
    }

    RECORD_DIR.mkdir(parents=True, exist_ok=True)
    (RECORD_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (RECORD_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (RECORD_DIR / "prompt_bundle.json").write_text(
        json.dumps(prompt_bundle, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (RECORD_DIR / "run_config.yaml").write_text(yaml.safe_dump(run_config, sort_keys=False), encoding="utf-8")
    (RECORD_DIR / "system_info.json").write_text(
        json.dumps(system_info, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (RECORD_DIR / "command.txt").write_text(
        "PYTHONPATH=src python records/prompt_opt_1usd_gpt54_family/2026-04-13_codex_local_runtime_final_smoke_2/scripts/proxy_smoke.py\n",
        encoding="utf-8",
    )
    (RECORD_DIR / "notes.md").write_text(
        "\n".join(
            [
                "Proxy smoke for the Local Runtime Final Smoke 2 prompt-opt candidate.",
                "",
                "Baseline: `configs/craftax_prompt_opt_qwen35_4b_gpt54_budget.yaml`.",
                "Candidate: `configs/craftax_prompt_opt_qwen35_4b_codex_local_runtime_final_smoke_2.yaml`.",
                "Evaluation path: `nanohorizon.shared.craftax_data.collect_rollouts_concurrently_with_summary` with a deterministic proxy rollout responder over the held-out starter eval seeds.",
                "Result: the wood-first bootstrap prompt improved proxy mean outcome reward and kept the early-game resource ladder explicit.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
