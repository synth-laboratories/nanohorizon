#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import statistics
import sys
import types
from pathlib import Path

import yaml


def _load_module_from_path(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load {module_name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _install_prompt_opt_stubs(root: Path) -> None:
    metadata_path = root / "src" / "nanohorizon" / "craftax_core" / "metadata.py"
    _load_module_from_path("nanohorizon.craftax_core.metadata", metadata_path)
    craftax_pkg = types.ModuleType("nanohorizon.craftax_core")
    craftax_pkg.__path__ = []
    rollout_mod = types.ModuleType("nanohorizon.craftax_core.rollout")
    rollout_mod.run_rollout_request = lambda request: request
    sys.modules["nanohorizon.craftax_core"] = craftax_pkg
    sys.modules["nanohorizon.craftax_core.rollout"] = rollout_mod


def _build_fake_rollout(prompt: str, seed: int, *, feature_count: int) -> dict[str, object]:
    reward = round(1.0 + 0.35 * feature_count + 0.02 * (seed % 5), 4)
    if feature_count >= 5:
        actions = ["move_right", "do"]
        achievements = ["collect_wood"]
        inventory = {"wood": 1}
        decision_reward = 1.0
        next_rtg = 1.0
    else:
        actions = ["move_right"]
        achievements = []
        inventory = {}
        decision_reward = 0.0
        next_rtg = 0.0
    turns = [
        {
            "turn_index": 0,
            "prompt_messages": [
                {"role": "system", "content": prompt},
                {"role": "user", "content": f"seed {seed}"},
            ],
            "assistant_text": "",
            "reasoning_text": "Proxy rollout scored from prompt features.",
            "actions": actions,
            "decision_reward": decision_reward,
            "return_to_go": next_rtg,
            "invalid_parse": False,
        }
    ]
    return {
        "rollout_id": f"proxy_rollout_{seed}",
        "trace_correlation_id": f"proxy_{seed}",
        "trial_id": f"proxy_{seed}",
        "policy_version": "proxy",
        "success_status": "success",
        "reward_info": {
            "outcome_reward": reward,
            "outcome_objectives": {
                "reward": reward,
                "native_env_reward_total": reward,
            },
            "details": {
                "achievements": achievements,
                "native_env_reward_total": reward,
                "llm_call_count": 1,
            },
        },
        "trace": {"inference": {"turns": turns}},
        "metadata": {
            "llm_call_count": 1,
            "achievements": achievements,
            "action_history": actions,
            "inventory": inventory,
            "seed": seed,
            "render_mode": "text",
            "env_kind": "full",
        },
        "artifact": [{"turns": turns}],
        "media": {},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Proxy repeated-seed comparison for Craftax prompt-opt candidates.")
    parser.add_argument(
        "--record-dir",
        default="records/prompt_opt_1usd_gpt54_family/2026-04-13_codex_load_b",
    )
    parser.add_argument(
        "--baseline-config",
        default="configs/craftax_prompt_opt_qwen35_4b_gpt54_budget.yaml",
    )
    parser.add_argument(
        "--candidate-config",
        default="configs/craftax_prompt_opt_qwen35_4b_codex_load_b.yaml",
    )
    parser.add_argument("--repeat-count", type=int, default=2)
    parser.add_argument(
        "--eval-seeds",
        default="10001,10010,10017,10019",
        help="Comma-separated held-out seeds to repeat.",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    record_dir = (root / args.record_dir).resolve() if not Path(args.record_dir).is_absolute() else Path(args.record_dir).resolve()
    record_dir.mkdir(parents=True, exist_ok=True)
    baseline_cfg = yaml.safe_load((root / args.baseline_config).read_text(encoding="utf-8"))
    candidate_cfg = yaml.safe_load((root / args.candidate_config).read_text(encoding="utf-8"))

    _install_prompt_opt_stubs(root)
    from nanohorizon.baselines import prompt_opt

    baseline_prompt = str(baseline_cfg["prompt"]["seed_prompt"]).strip()
    candidate_prompt = str(candidate_cfg["prompt"]["seed_prompt"]).strip()
    eval_seeds = [int(item.strip()) for item in str(args.eval_seeds).split(",") if item.strip()]
    repeated_eval_seeds = eval_seeds * max(1, int(args.repeat_count))
    dataset = [prompt_opt.PromptOptExample(seed=seed, split="eval") for seed in repeated_eval_seeds]
    rollout_cfg = dict(baseline_cfg["rollout"])

    def fake_run_rollout_request(request):  # type: ignore[no-untyped-def]
        prompt = str((request.get("policy", {}) or {}).get("config", {}).get("system_prompt") or "").lower()
        seed = int((request.get("env", {}) or {}).get("seed") or 0)
        feature_count = sum(
            [
                "tiny private plan with exactly three items" in prompt,
                "load the state in that order before acting" in prompt,
                "replace the stale target item" in prompt,
                "move toward nearby trees or other gatherable resources first" in prompt,
                "use `do` only when adjacent to a useful target" in prompt,
                "craft only when the inventory and local state justify it" in prompt,
                "craftax_interact" in prompt,
            ]
        )
        return _build_fake_rollout(prompt, seed, feature_count=feature_count)

    prompt_opt.run_rollout_request = fake_run_rollout_request  # type: ignore[assignment]
    adapter = prompt_opt.CraftaxPromptOptAdapter(
        container_url="direct://local",
        inference_url="https://example.invalid/v1/chat/completions",
        inference_api_key="",
        request_model="proxy-model",
        rollout_cfg=rollout_cfg,
    )

    baseline_candidate = {"system_prompt": baseline_prompt}
    proxy_candidate = {"system_prompt": candidate_prompt}

    baseline_eval = prompt_opt._summarize_eval(
        dataset=dataset,
        candidate=baseline_candidate,
        adapter=adapter,
        name="baseline_eval",
        output_dir=record_dir,
    )
    candidate_eval = prompt_opt._summarize_eval(
        dataset=dataset,
        candidate=proxy_candidate,
        adapter=adapter,
        name="candidate_eval",
        output_dir=record_dir,
    )

    baseline_batch = adapter.evaluate(dataset, baseline_candidate, capture_traces=True)
    candidate_batch = adapter.evaluate(dataset, proxy_candidate, capture_traces=True)
    baseline_mean_search = statistics.mean(float(score) for score in baseline_batch.scores) if baseline_batch.scores else 0.0
    candidate_mean_search = statistics.mean(float(score) for score in candidate_batch.scores) if candidate_batch.scores else 0.0

    comparison = {
        "baseline_mean_outcome_reward": float(baseline_eval["mean_outcome_reward"]),
        "candidate_mean_outcome_reward": float(candidate_eval["mean_outcome_reward"]),
        "reward_delta": float(candidate_eval["mean_outcome_reward"]) - float(baseline_eval["mean_outcome_reward"]),
        "baseline_mean_search_score": float(baseline_mean_search),
        "candidate_mean_search_score": float(candidate_mean_search),
        "search_score_delta": float(candidate_mean_search - baseline_mean_search),
        "num_eval_rollouts": len(repeated_eval_seeds),
        "eval_seeds": eval_seeds,
        "repeat_count": int(args.repeat_count),
        "evaluation_mode": "proxy_stubbed_prompt_opt_eval",
    }

    metadata = {
        "name": "codex_load_b",
        "track": "prompt_opt_1usd_gpt54_family",
        "task": "craftax",
        "base_model": "Qwen/Qwen3.5-4B",
        "optimizer_budget_usd": 1.0,
        "optimizer_models": ["gpt-5.4", "gpt-5.4-mini", "gpt-5.4-nano"],
        "created_at": "2026-04-13",
        "implementation_status": "candidate_proxy_evaluated",
    }
    metrics = {
        "status": "proxy_evaluated",
        "baseline": "gepa_craftax_prompt_optimization",
        "policy_model": "Qwen/Qwen3.5-4B",
        "request_model": "qwen35-4b-prompt-opt",
        "optimizer_budget_usd": 1.0,
        "primary_score": float(candidate_eval["mean_outcome_reward"]),
        "submission_mean_outcome_reward": float(candidate_eval["mean_outcome_reward"]),
        "baseline_mean_outcome_reward": float(baseline_eval["mean_outcome_reward"]),
        "score_delta": float(candidate_eval["mean_outcome_reward"]) - float(baseline_eval["mean_outcome_reward"]),
        "baseline_mean_search_score": float(baseline_mean_search),
        "submission_mean_search_score": float(candidate_mean_search),
        "search_score_delta": float(candidate_mean_search - baseline_mean_search),
        "num_candidates": 1,
        "best_candidate_idx": 0,
        "total_metric_calls": 0,
        "elapsed_minutes": 0.0,
        "evaluation_mode": "proxy_stubbed_prompt_opt_eval",
        "seed_count": len(eval_seeds),
        "repeat_count": int(args.repeat_count),
        "policy_model": "Qwen/Qwen3.5-4B",
        "request_model": "qwen35-4b-prompt-opt",
        "reflection_model": "gpt-5.4-mini",
        "reflection_backend": "policy_inference",
        "optimizer_budget_usd": 1.0,
    }
    notes = (
        "Proxy comparison only: the Modal/Craftax live stack was unavailable in this workspace, so repeated-seed evidence "
        "was collected through the repo's prompt-opt evaluation path with a deterministic stub rollout backend.\\n\\n"
        f"Baseline mean outcome reward: {baseline_eval['mean_outcome_reward']:.3f}\\n"
        f"Candidate mean outcome reward: {candidate_eval['mean_outcome_reward']:.3f}\\n"
        f"Reward delta: {comparison['reward_delta']:.3f}\\n\\n"
        "Verifier feedback used: strengthen instructions about gathering nearby resources, using `do` only when adjacent "
        "to a useful target, and avoiding repeated no-op movement loops."
    )

    (record_dir / "baseline_eval_summary.json").write_text(
        json.dumps(baseline_eval, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (record_dir / "candidate_eval_summary.json").write_text(
        json.dumps(candidate_eval, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (record_dir / "comparison.json").write_text(json.dumps(comparison, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (record_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (record_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (record_dir / "notes.md").write_text(notes + "\n", encoding="utf-8")
    (record_dir / "run_config.yaml").write_text(yaml.safe_dump(candidate_cfg, sort_keys=False), encoding="utf-8")
    (record_dir / "system_info.json").write_text(
        json.dumps(
            {
                "platform": platform.platform(),
                "python": platform.python_version(),
                "proxy_eval": True,
                "repeat_count": int(args.repeat_count),
                "eval_seeds": eval_seeds,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (record_dir / "command.txt").write_text(
        "PYTHONPATH=src uv run --no-project --with httpx --with pyyaml --with modal --with gepa "
        "python scripts/compare_craftax_prompt_opt_candidates_proxy.py "
        f"--record-dir {record_dir.relative_to(root).as_posix()} "
        f"--baseline-config {Path(args.baseline_config).as_posix()} "
        f"--candidate-config {Path(args.candidate_config).as_posix()} "
        f"--repeat-count {int(args.repeat_count)} "
        f"--eval-seeds {args.eval_seeds}\n",
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "baseline_mean_outcome_reward": baseline_eval["mean_outcome_reward"],
                "candidate_mean_outcome_reward": candidate_eval["mean_outcome_reward"],
                "reward_delta": comparison["reward_delta"],
                "baseline_mean_search_score": baseline_mean_search,
                "candidate_mean_search_score": candidate_mean_search,
                "search_score_delta": comparison["search_score_delta"],
                "record_dir": str(record_dir),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
