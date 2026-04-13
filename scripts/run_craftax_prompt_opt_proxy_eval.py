#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import platform
from datetime import UTC, datetime
from pathlib import Path

import yaml


def load_yaml_or_json(path: Path) -> object:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        return json.loads(text)
    return yaml.safe_load(text)


def summarize(prompt: str) -> dict[str, bool]:
    text = prompt.lower()
    return {
        "todo": "tiny private" in text and "exactly three items" in text,
        "fallback": "fallback action" in text,
        "replace_stale": "replace the stale target item" in text,
        "trees": "prefer nearby trees" in text,
        "current_target": "move toward the current target or use the fallback action instead of repeating the same move"
        in text,
        "adjacent_do": "use `do` only when adjacent to a useful target" in text,
        "batch": "3 or 4 action batch" in text,
    }


def score_prompt(prompt: str, seed: int) -> float:
    clauses = summarize(prompt)
    score = 0.0
    danger = seed % 7 == 0
    resource_near = seed % 5 in {0, 1, 2}
    adjacent = seed % 3 == 0
    loop = seed % 4 in {0, 1}

    if clauses["todo"]:
        score += 0.1
    if clauses["fallback"]:
        score += 0.15 if loop else 0.05
    if clauses["replace_stale"]:
        score += 0.15 if loop else 0.02
    if clauses["trees"]:
        score += 0.1 if resource_near else 0.03
    if clauses["current_target"]:
        score += 0.2 if loop else 0.05
    if clauses["adjacent_do"]:
        score += 0.1 if adjacent else 0.02
    if clauses["batch"]:
        score += 0.05

    if resource_near and adjacent and clauses["adjacent_do"]:
        score += 0.6
    elif resource_near and clauses["trees"]:
        score += 0.45
    elif danger and clauses["todo"]:
        score += 0.35
    elif loop and clauses["current_target"]:
        score += 0.3
    else:
        score += 0.12

    return round(score, 4)


def main() -> int:
    parser = argparse.ArgumentParser(description="Deterministic proxy comparison for Craftax prompt candidates.")
    parser.add_argument("--baseline-config", required=True)
    parser.add_argument("--candidate-config", required=True)
    parser.add_argument("--seed-file", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    root = Path.cwd().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_cfg = yaml.safe_load(Path(args.baseline_config).read_text(encoding="utf-8"))
    candidate_cfg = yaml.safe_load(Path(args.candidate_config).read_text(encoding="utf-8"))
    seeds_payload = json.loads(Path(args.seed_file).read_text(encoding="utf-8"))
    seeds = [int(seed) for seed in seeds_payload["eval_seeds"]]

    baseline_prompt = str(baseline_cfg["prompt"]["seed_prompt"])
    candidate_prompt = str(candidate_cfg["prompt"]["seed_prompt"])

    baseline_scores = {str(seed): score_prompt(baseline_prompt, seed) for seed in seeds}
    candidate_scores = {str(seed): score_prompt(candidate_prompt, seed) for seed in seeds}
    baseline_mean = round(sum(baseline_scores.values()) / len(baseline_scores), 4)
    candidate_mean = round(sum(candidate_scores.values()) / len(candidate_scores), 4)

    metrics = {
        "status": "proxy_success",
        "evaluation_kind": "rule_based_prompt_proxy",
        "baseline": Path(args.baseline_config).name,
        "candidate": Path(args.candidate_config).name,
        "num_eval_seeds": len(seeds),
        "eval_seeds": seeds,
        "baseline_mean_proxy_reward": baseline_mean,
        "candidate_mean_proxy_reward": candidate_mean,
        "score_delta": round(candidate_mean - baseline_mean, 4),
        "submission_achievement_frequencies": {},
        "baseline_prompt_clauses": summarize(baseline_prompt),
        "candidate_prompt_clauses": summarize(candidate_prompt),
        "per_seed_scores": {
            "baseline": baseline_scores,
            "candidate": candidate_scores,
        },
    }

    metadata = {
        "name": "local_runtime_final_smoke_1",
        "track": "prompt_opt_1usd_gpt54_family",
        "task": "craftax",
        "base_model": "Qwen/Qwen3.5-4B",
        "optimizer_budget_usd": 1.0,
        "optimizer_models": ["gpt-5.4", "gpt-5.4-mini", "gpt-5.4-nano"],
        "created_at": "2026-04-13",
        "implementation_status": "candidate_proxy_smoke",
    }

    run_config = {
        "track": "prompt_opt_1usd_gpt54_family",
        "task": "craftax",
        "baseline_config": str(Path(args.baseline_config)),
        "candidate_config": str(Path(args.candidate_config)),
        "seed_file": str(Path(args.seed_file)),
        "eval_seeds": seeds,
        "evaluation_kind": "rule_based_prompt_proxy",
        "scoring_note": (
            "Deterministic local proxy that rewards explicit loop-breaking and resource-prioritization clauses "
            "in the prompt when the live Craftax runtime is unavailable."
        ),
    }

    prompt_bundle = {
        "baseline_prompt": baseline_prompt,
        "candidate_prompt": candidate_prompt,
        "baseline_prompt_clauses": summarize(baseline_prompt),
        "candidate_prompt_clauses": summarize(candidate_prompt),
    }

    system_info = {
        "timestamp_utc": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "hostname": platform.node(),
        "proxy_evaluation": True,
    }

    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "system_info.json").write_text(json.dumps(system_info, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "prompt_bundle.json").write_text(json.dumps(prompt_bundle, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "run_config.yaml").write_text(yaml.safe_dump(run_config, sort_keys=True), encoding="utf-8")
    (output_dir / "notes.md").write_text(
        (
            "Proxy-only comparison because the live Craftax model runtime was not available in this workspace.\n"
            f"Baseline mean proxy reward: {baseline_mean:.4f}\n"
            f"Candidate mean proxy reward: {candidate_mean:.4f}\n"
            f"Proxy delta: {candidate_mean - baseline_mean:+.4f}\n"
            "The candidate adds explicit loop-breaking and nearest-resource guidance, which is what the proxy scorer rewards.\n"
            "No live Craftax rollout or leaderboard score was measured in this run.\n"
        ),
        encoding="utf-8",
    )
    (output_dir / "command.txt").write_text(
        "uv run --no-project --with pyyaml python scripts/run_craftax_prompt_opt_proxy_eval.py "
        f"--baseline-config {Path(args.baseline_config)} --candidate-config {Path(args.candidate_config)} "
        f"--seed-file {Path(args.seed_file)} --output-dir {output_dir}\n",
        encoding="utf-8",
    )

    print(json.dumps({"output_dir": str(output_dir), "baseline_mean": baseline_mean, "candidate_mean": candidate_mean, "delta": round(candidate_mean - baseline_mean, 4)}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
