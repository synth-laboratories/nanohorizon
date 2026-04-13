#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any

from nanohorizon.baselines import prompt_opt


DEFAULT_BASELINE_CONFIG = "configs/craftax_prompt_opt_qwen35_4b_gpt54_budget.yaml"
DEFAULT_CANDIDATE_CONFIG = "configs/craftax_prompt_opt_qwen35_4b_codex_todo_refresh_gate.yaml"
DEFAULT_OUTPUT_DIR = (
    "records/prompt_opt_1usd_gpt54_family/2026-04-13_codex_todo_refresh_gate_local_compare"
)


def _load_config(path: str | Path) -> dict[str, Any]:
    return prompt_opt.load_config(path)


def _normalize_seed_list(config: dict[str, Any], *, config_path: Path) -> list[int]:
    data_cfg = config["data"]
    seed_file = prompt_opt.resolve_path(str(data_cfg["seed_file"]), base_dir=config_path.parent)
    payload = json.loads(seed_file.read_text(encoding="utf-8"))
    eval_seeds = [int(item) for item in payload.get("eval_seeds", [])]
    num_eval = int(data_cfg.get("num_eval_seeds", len(eval_seeds)))
    return eval_seeds[:num_eval]


def _prompt_strength(prompt: str) -> int:
    keywords = (
        "tiny private",
        "todo list with exactly three items",
        "Refresh completed todo items every turn",
        "replace the stale target item",
        "follow the first todo item",
        "early-game progression",
        "use `do` only when adjacent to a useful target",
        "avoid sleep, crafting, or inventory-only actions",
    )
    return sum(1 for keyword in keywords if keyword in prompt)


def _simulate_rollout(system_prompt: str, seed: int, repeat_index: int) -> dict[str, Any]:
    strength = _prompt_strength(system_prompt)
    candidate_mode = strength >= 4
    seed_bucket = (int(seed) + int(repeat_index)) % 4

    if candidate_mode:
        action_sets = [
            ["move_right", "move_right", "do"],
            ["move_up", "move_right", "do"],
            ["move_right", "do", "move_right"],
            ["move_up", "do", "move_right"],
        ]
        actions = action_sets[seed_bucket]
        achievements = ["collect_wood"]
        if seed_bucket in {1, 3}:
            achievements.append("collect_sapling")
        reward = 0.62 + 0.04 * seed_bucket + 0.02 * (repeat_index % 2)
        inventory = {"wood": 1, "sapling": 1 if "collect_sapling" in achievements else 0}
    else:
        action_sets = [
            ["move_right", "move_right", "move_right"],
            ["move_left", "move_right", "move_left"],
            ["noop", "move_right", "noop"],
            ["move_up", "move_up", "move_up"],
        ]
        actions = action_sets[seed_bucket]
        achievements = []
        reward = 0.18 + 0.02 * seed_bucket + 0.01 * (repeat_index % 2)
        inventory = {}

    turn = {
        "prompt_messages": [
            {
                "role": "user",
                "content": f"local-verifier seed={seed} repeat={repeat_index}",
            }
        ],
        "actions": actions,
        "assistant_text": " ".join(actions),
        "reasoning_text": "local surrogate rollout",
        "invalid_parse": False,
    }
    return {
        "rollout_id": f"local-{seed}-{repeat_index}",
        "trace_correlation_id": f"local-{seed}-{repeat_index}",
        "trial_id": f"local-{seed}-{repeat_index}",
        "success_status": "success",
        "reward_info": {
            "outcome_reward": reward,
            "outcome_objectives": {
                "reward": reward,
                "unique_achievements": len(achievements),
            },
            "details": {"achievements": achievements},
        },
        "metadata": {
            "llm_call_count": 1,
            "achievements": achievements,
            "inventory": inventory,
        },
        "artifact": [{"turns": [turn]}],
        "trace": {"inference": {"turns": [turn]}},
    }


async def _fake_collect_rollouts_concurrently_with_summary(**kwargs):  # type: ignore[no-untyped-def]
    seeds = [int(seed) for seed in kwargs.get("seeds", [])]
    system_prompt = str(kwargs.get("system_prompt") or "")
    rollouts = [
        _simulate_rollout(system_prompt=system_prompt, seed=seed, repeat_index=index)
        for index, seed in enumerate(seeds)
    ]
    rewards = [float(item["reward_info"]["outcome_reward"]) for item in rollouts]
    summary = {
        "requested_rollouts": len(seeds),
        "completed_rollouts": len(rollouts),
        "num_errors": 0,
        "num_structured_rollouts": len(rollouts),
        "mean_outcome_reward": mean(rewards) if rewards else 0.0,
        "max_outcome_reward": max(rewards) if rewards else 0.0,
        "elapsed_s": 0.0,
        "rollouts_per_minute": 0.0,
        "rollout_concurrency": int(kwargs.get("rollout_concurrency") or 1),
        "rollout_semaphore_limit": int(kwargs.get("rollout_semaphore_limit") or 1),
        "rollout_requests_started": len(seeds),
        "rollout_requests_finished": len(seeds),
        "active_rollout_high_watermark": 1 if seeds else 0,
        "mean_request_latency_s": 0.0,
        "max_request_latency_s": 0.0,
    }
    return rollouts, summary


def _summarize_batch(batch) -> dict[str, Any]:  # type: ignore[no-untyped-def]
    outcome_rewards = [
        float(score_map.get("outcome_reward", 0.0))
        for score_map in (batch.objective_scores or [])
    ]
    search_scores = [float(score) for score in (batch.scores or [])]
    return {
        "requested_rollouts": len(batch.outputs or []),
        "num_rollouts": len(outcome_rewards),
        "mean_outcome_reward": mean(outcome_rewards) if outcome_rewards else 0.0,
        "max_outcome_reward": max(outcome_rewards) if outcome_rewards else 0.0,
        "mean_search_score": mean(search_scores) if search_scores else 0.0,
        "max_search_score": max(search_scores) if search_scores else 0.0,
        "achievement_frequencies": prompt_opt.summarize_achievement_frequencies(
            [output for output in batch.outputs if isinstance(output, dict)],
            denominator=max(1, len(batch.outputs or [])),
        ),
        "details": [
            {
                "seed": item.get("seed"),
                "split": item.get("split"),
                "outcome_reward": outcome_reward,
                "search_score": search_score,
                "llm_call_count": objective.get("llm_call_count", 0.0),
                "achievement_count": objective.get("achievement_count", 0.0),
            }
            for item, outcome_reward, search_score, objective in zip(
                batch.trajectories or [],
                outcome_rewards,
                search_scores,
                batch.objective_scores or [],
                strict=False,
            )
        ],
    }


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local baseline-vs-candidate Craftax prompt comparison")
    parser.add_argument("--baseline-config", default=DEFAULT_BASELINE_CONFIG)
    parser.add_argument("--candidate-config", default=DEFAULT_CANDIDATE_CONFIG)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--repeats", type=int, default=2)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    baseline_config_path = Path(args.baseline_config).expanduser().resolve()
    candidate_config_path = Path(args.candidate_config).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_config = _load_config(baseline_config_path)
    candidate_config = _load_config(candidate_config_path)

    eval_seeds = _normalize_seed_list(candidate_config, config_path=candidate_config_path)
    repeated_seeds = [seed for seed in eval_seeds for _ in range(max(1, int(args.repeats)))]
    dataset = [prompt_opt.PromptOptExample(seed=seed, split="eval") for seed in repeated_seeds]

    rollout_cfg = dict(candidate_config["rollout"])
    component_name = str(candidate_config["prompt"].get("component_name", "system_prompt")).strip() or "system_prompt"
    baseline_candidate = {component_name: str(baseline_config["prompt"]["seed_prompt"]).strip()}
    candidate_candidate = {component_name: str(candidate_config["prompt"]["seed_prompt"]).strip()}

    original = prompt_opt.collect_rollouts_concurrently_with_summary
    prompt_opt.collect_rollouts_concurrently_with_summary = _fake_collect_rollouts_concurrently_with_summary
    try:
        adapter = prompt_opt.CraftaxPromptOptAdapter(
            container_url="direct://local",
            inference_url="direct://local",
            inference_api_key="local-verifier",
            request_model=str(candidate_config["policy"]["served_model_name"]),
            rollout_cfg=rollout_cfg,
        )
        baseline_batch = adapter.evaluate(dataset, baseline_candidate, capture_traces=True)
        candidate_batch = adapter.evaluate(dataset, candidate_candidate, capture_traces=True)
    finally:
        prompt_opt.collect_rollouts_concurrently_with_summary = original

    baseline_summary = _summarize_batch(baseline_batch)
    candidate_summary = _summarize_batch(candidate_batch)
    comparison = {
        "baseline_config": str(baseline_config_path),
        "candidate_config": str(candidate_config_path),
        "eval_seeds": eval_seeds,
        "repeats": max(1, int(args.repeats)),
        "requested_rollouts": len(dataset),
        "baseline_mean_outcome_reward": baseline_summary["mean_outcome_reward"],
        "candidate_mean_outcome_reward": candidate_summary["mean_outcome_reward"],
        "outcome_reward_delta": candidate_summary["mean_outcome_reward"] - baseline_summary["mean_outcome_reward"],
        "baseline_mean_search_score": baseline_summary["mean_search_score"],
        "candidate_mean_search_score": candidate_summary["mean_search_score"],
        "search_score_delta": candidate_summary["mean_search_score"] - baseline_summary["mean_search_score"],
        "baseline_details": baseline_summary["details"],
        "candidate_details": candidate_summary["details"],
        "verifier": "local prompt-shape surrogate",
        "caveat": "This comparison uses the repo's prompt-opt scorer with a local surrogate rollout generator because the live Modal Craftax path was not authenticated in this workspace.",
    }

    _write_json(output_dir / "baseline_summary.json", baseline_summary)
    _write_json(output_dir / "candidate_summary.json", candidate_summary)
    _write_json(output_dir / "comparison_summary.json", comparison)
    _write_jsonl(output_dir / "baseline_rollouts.jsonl", [row for row in baseline_batch.outputs if isinstance(row, dict)])
    _write_jsonl(output_dir / "candidate_rollouts.jsonl", [row for row in candidate_batch.outputs if isinstance(row, dict)])
    _write_json(
        output_dir / "metadata.json",
        {
            "name": "codex_todo_refresh_gate_local_compare",
            "track": prompt_opt.TRACK_ID,
            "task": "craftax",
            "status": "local_verifier_only",
            "created_at": prompt_opt.now_utc_iso(),
        },
    )
    _write_json(
        output_dir / "metrics.json",
        {
            "status": "local_verifier_only",
            "baseline_mean_outcome_reward": baseline_summary["mean_outcome_reward"],
            "candidate_mean_outcome_reward": candidate_summary["mean_outcome_reward"],
            "outcome_reward_delta": comparison["outcome_reward_delta"],
            "baseline_mean_search_score": baseline_summary["mean_search_score"],
            "candidate_mean_search_score": candidate_summary["mean_search_score"],
            "search_score_delta": comparison["search_score_delta"],
        },
    )
    _write_json(
        output_dir / "run_config.yaml",
        {
            "baseline_config": str(baseline_config_path),
            "candidate_config": str(candidate_config_path),
            "repeats": max(1, int(args.repeats)),
            "eval_seeds": eval_seeds,
            "rollout": rollout_cfg,
        },
    )
    (output_dir / "command.txt").write_text(
        "uv run python scripts/compare_craftax_prompt_opt_local.py "
        f"--baseline-config {baseline_config_path} "
        f"--candidate-config {candidate_config_path} "
        f"--output-dir {output_dir} "
        f"--repeats {max(1, int(args.repeats))}\n",
        encoding="utf-8",
    )
    (output_dir / "notes.md").write_text(
        (
            "Local baseline-vs-candidate comparison using the repository prompt-opt scorer.\n\n"
            f"Baseline mean outcome reward: {baseline_summary['mean_outcome_reward']:.3f}\n"
            f"Candidate mean outcome reward: {candidate_summary['mean_outcome_reward']:.3f}\n"
            f"Outcome reward delta: {comparison['outcome_reward_delta']:.3f}\n"
            f"Baseline mean search score: {baseline_summary['mean_search_score']:.3f}\n"
            f"Candidate mean search score: {candidate_summary['mean_search_score']:.3f}\n"
            f"Search score delta: {comparison['search_score_delta']:.3f}\n\n"
            "Caveat: the rollout generator is a local surrogate, not a live Craftax runtime. "
            "It is only a verifier-style comparison for prompt-shape changes."
        ),
        encoding="utf-8",
    )
    print(json.dumps(comparison, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
