#!/usr/bin/env python3
"""Score the Go-Explore prompt optimization results."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

TASK_ID = "nanohorizon_go_explore_prompt_opt"
ROOT = Path(__file__).resolve().parent.parent

ARTIFACTS = {
    "eval_report": "eval_report.md",
    "go_explore_result": "artifacts/go_explore_result.json",
    "experiment_summary": "artifacts/experiment_summary.json",
    "reportbench_output": "artifacts/reportbench_output.json",
}


def _resolve_output_root(raw: str | None) -> Path:
    if not raw:
        raise RuntimeError("--output-root is required")
    return Path(raw).expanduser().resolve()


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected JSON object in {path}")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _criterion(*, criterion_id: str, passed: bool, weight: float) -> dict[str, Any]:
    return {"id": criterion_id, "score": 1.0 if passed else 0.0, "weight": weight}


def cmd_score(output_root: Path, verifier_mode: str) -> int:
    for rel in ("task.toml",):
        source = ROOT / rel
        if source.exists():
            dest = output_root / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, dest)

    report_path = output_root / ARTIFACTS["eval_report"]
    result_path = output_root / ARTIFACTS["go_explore_result"]
    summary_path = output_root / ARTIFACTS["experiment_summary"]
    output_path = output_root / ARTIFACTS["reportbench_output"]

    if not report_path.exists():
        raise RuntimeError(f"missing report artifact: {report_path}")
    if not result_path.exists():
        raise RuntimeError(f"missing go_explore_result artifact: {result_path}")
    if not summary_path.exists():
        raise RuntimeError(f"missing experiment_summary artifact: {summary_path}")

    result = _load_json(result_path)
    summary = _load_json(summary_path)

    baseline_reward = summary.get("baseline_reward")
    best_reward = summary.get("best_reward")
    uplift = summary.get("uplift")
    training_rollouts = summary.get("training_rollouts_used")
    report_text = report_path.read_text(encoding="utf-8")

    has_report = bool(report_text.strip())
    has_results = baseline_reward is not None and best_reward is not None
    has_uplift = isinstance(uplift, (int, float)) and uplift > 0
    under_budget = isinstance(training_rollouts, (int, float)) and training_rollouts <= 500
    achievement_evidence = (
        summary.get("baseline_mean_achievements") is not None
        and summary.get("best_mean_achievements") is not None
    )
    uplift_per_minute = summary.get("uplift_per_minute")
    has_uplift_rate = isinstance(uplift_per_minute, (int, float))
    has_proof = "seed" in report_text.lower() and ("|" in report_text or "seed" in report_text)

    criteria = [
        _criterion(criterion_id="report_present", passed=has_report, weight=0.15),
        _criterion(criterion_id="results_present", passed=has_results, weight=0.15),
        _criterion(criterion_id="positive_uplift", passed=has_uplift, weight=0.30),
        _criterion(criterion_id="under_budget", passed=under_budget, weight=0.15),
        _criterion(criterion_id="achievement_evidence", passed=achievement_evidence, weight=0.10),
        _criterion(criterion_id="uplift_per_minute_present", passed=has_uplift_rate, weight=0.05),
        _criterion(criterion_id="proof_in_report", passed=has_proof, weight=0.10),
    ]
    verifier_score = round(sum(c["score"] * c["weight"] for c in criteria), 6)

    payload = {
        "task_id": TASK_ID,
        "state": "succeeded" if verifier_score >= 0.999999 else "failed",
        "artifacts": {k: v for k, v in ARTIFACTS.items() if k != "reportbench_output"},
        "reward": {
            "value": best_reward,
            "primary_metric": "heldout_mean_reward",
            "source_artifact": ARTIFACTS["experiment_summary"],
            "info": {
                "baseline_reward": baseline_reward,
                "best_reward": best_reward,
                "uplift": uplift,
                "training_rollouts_used": training_rollouts,
                "baseline_mean_achievements": summary.get("baseline_mean_achievements"),
                "best_mean_achievements": summary.get("best_mean_achievements"),
                "achievement_uplift": summary.get("achievement_uplift"),
                "uplift_per_minute": uplift_per_minute,
            },
        },
        "runtime": {
            "elapsed_seconds": summary.get("elapsed_seconds"),
            "model": summary.get("model"),
        },
        "verifier": {
            "score": verifier_score,
            "summary": "Go-Explore prompt optimization scored from submitted artifacts.",
            "criteria": criteria,
            "latest_state": "done" if verifier_score >= 0.999999 else "failed",
            "verifier_mode": verifier_mode,
        },
    }
    _write_json(output_path, payload)
    print(json.dumps({
        "score": verifier_score,
        "baseline_reward": baseline_reward,
        "best_reward": best_reward,
        "uplift": uplift,
        "training_rollouts": training_rollouts,
        "state": payload["state"],
    }, indent=2, sort_keys=True))
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("score",))
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--verifier-mode", default="precheck")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_root = _resolve_output_root(args.output_root)
    if args.command == "score":
        return cmd_score(output_root, verifier_mode=args.verifier_mode)
    raise RuntimeError(f"unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
