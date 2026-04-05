#!/usr/bin/env python3
"""Score the NanoHorizon unique-achievement GEPA bundle."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

TASK_ID = "nanohorizon_unique_achievement_gepa_oe"
ROOT = Path(__file__).resolve().parent.parent

ARTIFACTS = {
    "eval_report": "eval_report.md",
    "baseline": "artifacts/unique_achievement_baseline.json",
    "result": "artifacts/unique_achievement_gepa_result.json",
    "summary": "artifacts/unique_achievement_summary.json",
    "reportbench_output": "artifacts/reportbench_output.json",
}

PROMPT_OPT_TRACK_ID = "prompt_opt_local_gemini25_flash_lite"
REQUIRED_BASELINE_SCRIPT = "run_craftax_prompt_opt_gemini25_flash_lite_local.sh"
REQUIRED_BASELINE_CONFIG = "craftax_prompt_opt_gemini25_flash_lite_local_eval20.yaml"
INVALID_BASELINE_PATH_MARKERS = ("offline_", "rlvr_")
INVALID_SALVAGE_TEXT_MARKERS = (
    "manual probe",
    "manual_probe",
    "prior-run surrogate",
    "copied result",
    "borrowed",
    "/synth/state/.out/smr/projects/",
    "checked-in reference",
    "checked_in_reference",
    "no fresh live rerun was possible",
)
MIN_MEANINGFUL_TRAINING_ROLLOUTS = 100


def _contains_invalid_baseline_marker(value: Any) -> bool:
    text = str(value or "")
    return any(marker in text for marker in INVALID_BASELINE_PATH_MARKERS) or "reference_baseline" in text


def _contains_invalid_salvage_marker(value: Any) -> bool:
    text = str(value or "").lower()
    return any(marker in text for marker in INVALID_SALVAGE_TEXT_MARKERS)


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
    source = ROOT / "task.toml"
    if source.exists():
        dest = output_root / "task.toml"
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dest)

    report_path = output_root / ARTIFACTS["eval_report"]
    baseline_path = output_root / ARTIFACTS["baseline"]
    result_path = output_root / ARTIFACTS["result"]
    summary_path = output_root / ARTIFACTS["summary"]
    output_path = output_root / ARTIFACTS["reportbench_output"]

    for required in (report_path, baseline_path, result_path, summary_path):
        if not required.exists():
            raise RuntimeError(f"missing required artifact: {required}")

    summary = _load_json(summary_path)
    baseline = _load_json(baseline_path)
    result = _load_json(result_path)
    report_text = report_path.read_text(encoding="utf-8")
    baseline_text = json.dumps(baseline, sort_keys=True)
    result_text = json.dumps(result, sort_keys=True)
    summary_text = json.dumps(summary, sort_keys=True)

    baseline_unique = summary.get("baseline_unique_achievement_score")
    optimized_unique = summary.get("optimized_unique_achievement_score")
    uplift = summary.get("unique_achievement_uplift")
    rollouts = summary.get("training_rollouts_used")
    termination_reason = (
        summary.get("termination_reason")
        or result.get("termination_reason")
        or baseline.get("termination_reason")
    )
    baseline_track_id = (
        baseline.get("baseline_track_id")
        or summary.get("baseline_track_id")
        or result.get("baseline_track_id")
    )
    baseline_record_path = (
        baseline.get("baseline_record_path")
        or result.get("baseline_record_path")
        or summary.get("baseline_record_path")
    )
    baseline_script_path = (
        baseline.get("baseline_script_path")
        or result.get("baseline_script_path")
        or summary.get("baseline_script_path")
    )
    baseline_config_path = (
        baseline.get("baseline_config_path")
        or result.get("baseline_config_path")
        or summary.get("baseline_config_path")
    )
    baseline_is_prompt_opt = (
        baseline_track_id == PROMPT_OPT_TRACK_ID
        and not _contains_invalid_baseline_marker(baseline_record_path)
        and not _contains_invalid_baseline_marker(baseline_script_path)
        and not _contains_invalid_baseline_marker(baseline_config_path)
        and REQUIRED_BASELINE_SCRIPT in str(baseline_script_path or "")
        and REQUIRED_BASELINE_CONFIG in str(baseline_config_path or "")
        and str(baseline.get("baseline_source_kind") or "").strip() not in {"checked_in_reference_bundle", ""}
        and str(result.get("result_source_kind") or "").strip() not in {"checked_in_reference_gepa_result"}
    )
    no_salvage_artifacts = not any(
        _contains_invalid_salvage_marker(value)
        for value in (
            report_text,
            baseline_text,
            result_text,
            summary_text,
        )
    )
    meaningful_budget_use = isinstance(rollouts, (int, float)) and rollouts <= 500 and (
        rollouts >= MIN_MEANINGFUL_TRAINING_ROLLOUTS or bool(str(termination_reason or "").strip())
    )

    criteria = [
        _criterion(
            criterion_id="baseline_present",
            passed=baseline_unique is not None,
            weight=0.15,
        ),
        _criterion(
            criterion_id="baseline_is_prompt_opt",
            passed=baseline_is_prompt_opt,
            weight=0.20,
        ),
        _criterion(
            criterion_id="optimized_present",
            passed=optimized_unique is not None,
            weight=0.15,
        ),
        _criterion(
            criterion_id="positive_uplift",
            passed=isinstance(uplift, (int, float)) and uplift > 0,
            weight=0.20,
        ),
        _criterion(
            criterion_id="meaningful_budget_accounting",
            passed=meaningful_budget_use,
            weight=0.15,
        ),
        _criterion(
            criterion_id="fresh_run_no_salvage",
            passed=no_salvage_artifacts,
            weight=0.10,
        ),
        _criterion(
            criterion_id="report_mentions_unique_achievements",
            passed="unique" in report_text.lower() and "achievement" in report_text.lower(),
            weight=0.10,
        ),
        _criterion(
            criterion_id="report_mentions_throughput",
            passed="500" in report_text and ("minute" in report_text.lower() or "throughput" in report_text.lower()),
            weight=0.10,
        ),
    ]
    verifier_score = round(sum(c["score"] * c["weight"] for c in criteria), 6)

    payload = {
        "task_id": TASK_ID,
        "state": "succeeded" if verifier_score >= 0.999999 else "failed",
        "artifacts": {k: v for k, v in ARTIFACTS.items() if k != "reportbench_output"},
        "reward": {
            "value": optimized_unique,
            "primary_metric": "heldout_unique_achievement_score",
            "source_artifact": ARTIFACTS["summary"],
            "info": {
                "baseline_unique_achievement_score": baseline_unique,
                "optimized_unique_achievement_score": optimized_unique,
                "unique_achievement_uplift": uplift,
                "training_rollouts_used": rollouts,
                "termination_reason": termination_reason,
                "elapsed_seconds": summary.get("elapsed_seconds"),
                "baseline_track_id": baseline_track_id,
                "baseline_record_path": baseline_record_path,
            },
        },
        "verifier": {
            "score": verifier_score,
            "summary": "Unique-achievement GEPA bundle scored from submitted artifacts.",
            "criteria": criteria,
            "latest_state": "done" if verifier_score >= 0.999999 else "failed",
            "verifier_mode": verifier_mode,
        },
    }
    _write_json(output_path, payload)
    print(
        json.dumps(
            {
                "score": verifier_score,
                "baseline_unique_achievement_score": baseline_unique,
                "optimized_unique_achievement_score": optimized_unique,
                "unique_achievement_uplift": uplift,
                "state": payload["state"],
            },
            indent=2,
            sort_keys=True,
        )
    )
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
