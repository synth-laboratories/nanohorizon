#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

TASK_ID = "nanohorizon_craftax_hello_world"
TASK_TITLE = "NanoHorizon Craftax Hello World"
ROOT = Path(__file__).resolve().parent.parent
RUNS_ROOT = ROOT / "runs" / TASK_ID
WORKER_SCRIPT = Path(__file__).resolve().parent / "nanohorizon_craftax_hello_world_worker.py"
RESOURCE_SCRIPT = Path(__file__).resolve().parent / "craftax_runtime_resource.py"

ARTIFACTS = {
    "eval_summary": "artifacts/eval_summary.json",
    "rollouts": "artifacts/rollouts.jsonl",
    "result_manifest": "artifacts/result_manifest.json",
    "container_proof": "artifacts/container_proof.json",
    "verifier_review": "artifacts/verifier_review.json",
    "reportbench_output": "artifacts/reportbench_output.json",
    "reproduction": "reports/reproduction.md",
}


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _looks_like_live_smr_workspace(path: Path) -> bool:
    return (path / "starting-data").exists()


def _default_output_root() -> Path:
    cwd = Path.cwd().resolve()
    if _looks_like_live_smr_workspace(cwd):
        return cwd
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return RUNS_ROOT / stamp


def _resolve_output_root(raw: str | None) -> Path:
    if not raw:
        return _default_output_root()
    candidate = Path(raw).expanduser().resolve()
    cwd = Path.cwd().resolve()
    if _looks_like_live_smr_workspace(cwd) and not str(candidate).startswith(str(cwd)):
        return cwd
    return candidate


def _artifact_path(output_root: Path, key: str) -> Path:
    return output_root / ARTIFACTS[key]


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected JSON object in {path}")
    return payload


def _resolve_nanohorizon_root() -> Path:
    explicit = str(os.getenv("NANOHORIZON_REPO_ROOT") or "").strip()
    cwd = Path.cwd().resolve()
    candidates = [
        Path(explicit).expanduser() if explicit else None,
        ROOT,
        cwd,
        cwd / "project",
        Path("/Users/joshpurtell/Documents/GitHub/nanohorizon"),
        Path.home() / "Documents" / "GitHub" / "nanohorizon",
    ]
    for candidate in candidates:
        if candidate is None:
            continue
        pyproject = candidate / "pyproject.toml"
        if pyproject.exists() and (candidate / "src" / "nanohorizon").exists():
            return candidate.resolve()
    raise RuntimeError("Unable to find the local nanohorizon checkout. Set NANOHORIZON_REPO_ROOT.")


def _resolve_nanohorizon_python(nanohorizon_root: Path) -> str:
    explicit = str(os.getenv("NANOHORIZON_PYTHON") or "").strip()
    candidates = [
        Path(explicit).expanduser() if explicit else None,
        nanohorizon_root / ".venv" / "bin" / "python",
        Path(sys.executable),
    ]
    for candidate in candidates:
        if candidate is None or not candidate.exists():
            continue
        return str(candidate.resolve())
    return sys.executable


def _run_baseline(output_root: Path) -> subprocess.CompletedProcess[str]:
    nanohorizon_root = _resolve_nanohorizon_root()
    nanohorizon_python = _resolve_nanohorizon_python(nanohorizon_root)
    summary_path = _artifact_path(output_root, "eval_summary")
    rollouts_path = _artifact_path(output_root, "rollouts")
    env = os.environ.copy()
    pythonpath_parts = [
        str(nanohorizon_root),
        str(nanohorizon_root / "src"),
    ]
    existing_pythonpath = str(env.get("PYTHONPATH") or "").strip()
    if existing_pythonpath:
        pythonpath_parts.append(existing_pythonpath)
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)
    env["NANOHORIZON_REPO_ROOT"] = str(nanohorizon_root)
    proof_process = subprocess.run(
        [
            nanohorizon_python,
            str(RESOURCE_SCRIPT),
            "local-proof",
            "--output",
            str(_artifact_path(output_root, "container_proof")),
        ],
        cwd=str(output_root),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if proof_process.returncode != 0:
        _write_json(
            _artifact_path(output_root, "container_proof"),
            {
                "recorded_at": _utc_now(),
                "resource_id": "craftax_rollout_http",
                "resource_ownership": "actor_managed",
                "shared_context_key": "runtime_resources.craftax",
                "ready": False,
                "proof_exit_code": int(proof_process.returncode),
                "stdout_preview": proof_process.stdout[:2000],
                "stderr_preview": proof_process.stderr[:2000],
                "note": "Actor-managed Craftax resource proof failed before the worker eval.",
            },
        )
        return proof_process
    cmd = [
        nanohorizon_python,
        str(WORKER_SCRIPT),
        "--summary-output",
        str(summary_path),
        "--rollouts-output",
        str(rollouts_path),
    ]
    return subprocess.run(
        cmd,
        cwd=str(output_root),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def _requested_rollouts(summary: dict[str, Any]) -> int:
    return int(summary.get("requested_rollouts", 0) or 0)


def _observed_rollouts(summary: dict[str, Any]) -> int:
    explicit = summary.get("num_rollouts")
    if explicit is not None:
        return int(explicit or 0)
    rollout_summary = summary.get("rollout_summary")
    if isinstance(rollout_summary, dict):
        if rollout_summary.get("num_structured_rollouts") is not None:
            return int(rollout_summary.get("num_structured_rollouts") or 0)
        if rollout_summary.get("completed_rollouts") is not None:
            return int(rollout_summary.get("completed_rollouts") or 0)
    return 0


def _num_errors(summary: dict[str, Any]) -> int:
    explicit = summary.get("num_errors")
    if explicit is not None:
        return int(explicit or 0)
    rollout_summary = summary.get("rollout_summary")
    if isinstance(rollout_summary, dict):
        return int(rollout_summary.get("num_errors") or 0)
    return 0


def _concrete_failure_reasons(summary: dict[str, Any], manifest: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    if manifest.get("status") != "succeeded":
        reasons.append("runner did not succeed")
    requested_rollouts = _requested_rollouts(summary)
    observed_rollouts = _observed_rollouts(summary)
    num_errors = _num_errors(summary)
    if summary.get("mean_outcome_reward") is None:
        reasons.append("mean_outcome_reward missing from summary")
    if requested_rollouts < 1:
        reasons.append(f"requested_rollouts must be positive, got {requested_rollouts}")
    requested_concurrency = int(summary.get("requested_rollout_concurrency", 0) or 0)
    if requested_concurrency < 1:
        reasons.append(f"requested_rollout_concurrency must be positive, got {requested_concurrency}")
    if observed_rollouts != requested_rollouts:
        reasons.append(
            f"structured rollout count mismatch: observed={observed_rollouts} requested={requested_rollouts}"
        )
    if num_errors > 0:
        reasons.append(f"rollout execution reported {num_errors} error(s)")
    return reasons


def _concrete_run_succeeded(summary: dict[str, Any], manifest: dict[str, Any]) -> bool:
    return not _concrete_failure_reasons(summary, manifest)


def _write_result_manifest(output_root: Path, *, process: subprocess.CompletedProcess[str]) -> dict[str, Any]:
    summary = _load_json(_artifact_path(output_root, "eval_summary"))
    provisional_status = "succeeded" if process.returncode == 0 else "failed"
    provisional_manifest = {"status": provisional_status}
    failure_reasons = _concrete_failure_reasons(summary, provisional_manifest)
    final_status = "failed" if failure_reasons else provisional_status
    manifest = {
        "task_id": TASK_ID,
        "completed_at": _utc_now(),
        "status": final_status,
        "runner_exit_code": int(process.returncode),
        "model": summary.get("model"),
        "mean_outcome_reward": summary.get("mean_outcome_reward"),
        "requested_rollouts": summary.get("requested_rollouts"),
        "requested_total_llm_calls": summary.get("requested_total_llm_calls"),
        "requested_max_steps_per_rollout": summary.get("requested_max_steps_per_rollout"),
        "requested_rollout_concurrency": summary.get("requested_rollout_concurrency"),
        "observed_rollouts": _observed_rollouts(summary),
        "num_errors": _num_errors(summary),
        "failure_reasons": failure_reasons,
        "rollouts_path": ARTIFACTS["rollouts"],
        "summary_path": ARTIFACTS["eval_summary"],
        "stdout_preview": process.stdout[:2000],
        "stderr_preview": process.stderr[:2000],
    }
    _write_json(_artifact_path(output_root, "result_manifest"), manifest)
    return manifest


def _write_reproduction_report(output_root: Path) -> None:
    summary = _load_json(_artifact_path(output_root, "eval_summary"))
    container_proof = (
        _load_json(_artifact_path(output_root, "container_proof"))
        if _artifact_path(output_root, "container_proof").exists()
        else {}
    )
    report = (
        "# NanoHorizon Craftax Hello World\n\n"
        f"- Completed at: `{_utc_now()}`\n"
        f"- Model: `{summary.get('model')}`\n"
        f"- Task: `{summary.get('task')}`\n"
        f"- Requested trajectories: `{summary.get('requested_rollouts')}`\n"
        f"- Requested total LLM calls: `{summary.get('requested_total_llm_calls')}`\n"
        f"- Requested LLM calls per rollout cap: `{summary.get('requested_llm_calls_per_rollout')}`\n"
        f"- Requested rollout concurrency: `{summary.get('requested_rollout_concurrency')}`\n"
        f"- Mean reward: `{summary.get('mean_outcome_reward')}`\n"
        f"- Max reward: `{summary.get('max_outcome_reward')}`\n"
        f"- Mean LLM calls per rollout: `{summary.get('mean_llm_calls_per_rollout')}`\n"
        f"- Errors: `{summary.get('num_errors')}`\n"
        f"- Craftax resource mode: `{container_proof.get('mode') or container_proof.get('resource_mode')}`\n"
        f"- Craftax resource proof: `{ARTIFACTS['container_proof']}`\n"
    )
    output_path = _artifact_path(output_root, "reproduction")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")


def _compute_verifier_review(output_root: Path, verifier_mode: str) -> dict[str, Any]:
    summary = _load_json(_artifact_path(output_root, "eval_summary"))
    manifest = _load_json(_artifact_path(output_root, "result_manifest"))
    report_exists = _artifact_path(output_root, "reproduction").exists()
    rollouts_exists = _artifact_path(output_root, "rollouts").exists()
    container_proof_exists = _artifact_path(output_root, "container_proof").exists()
    requested_rollouts = _requested_rollouts(summary)
    observed_rollouts = _observed_rollouts(summary)
    num_errors = _num_errors(summary)
    failure_reasons = _concrete_failure_reasons(summary, manifest)
    score = 1.0
    notes: list[str] = []
    if failure_reasons:
        score = 0.0
        notes.extend(failure_reasons)
    if not report_exists:
        score = min(score, 0.2)
        notes.append("report missing")
    if not rollouts_exists:
        score = min(score, 0.2)
        notes.append("rollout evidence missing")
    if not container_proof_exists:
        score = min(score, 0.8)
        notes.append("container resource proof missing")
    if verifier_mode != "precheck":
        notes.append("public_repo_runner_uses_deterministic_review_only")
    return {
        "score": round(float(score), 6),
        "summary": (
            "NanoHorizon Craftax hello-world bundle satisfied the concrete rollout contract."
            if score == 1.0
            else "NanoHorizon Craftax hello-world bundle did not satisfy the concrete rollout contract."
        ),
        "criteria": [
            {
                "id": "artifact_completeness",
                "score": 1.0 if report_exists and rollouts_exists and container_proof_exists else 0.0,
                "weight": 0.30,
            },
            {
                "id": "reward_grounding",
                "score": 1.0 if summary.get("mean_outcome_reward") is not None and not failure_reasons else 0.0,
                "weight": 0.30,
            },
            {
                "id": "rollout_evidence",
                "score": 1.0 if rollouts_exists and observed_rollouts == requested_rollouts and num_errors == 0 else 0.0,
                "weight": 0.20,
            },
            {"id": "report_grounding", "score": 1.0 if report_exists else 0.0, "weight": 0.20},
        ],
        "notes": notes or ["bundle is grounded in the concrete rollout summary and rollout records"],
        "verifier_mode": verifier_mode,
    }


def _benchmark_verdict(verifier_score: Any, concrete_success: bool) -> dict[str, Any]:
    try:
        score = float(verifier_score)
    except (TypeError, ValueError):
        score = 0.0
    passed = concrete_success and score >= 0.999999
    return {
        "authority": "nanohorizon_public_runner",
        "task_id": TASK_ID,
        "passed": passed,
        "primary_score": score,
        "pass_threshold": 0.999999,
        "score_source": "artifacts/reportbench_output.json:verifier.score",
        "gates": [
            {
                "id": "verifier_score_present",
                "passed": passed,
                "reason": None if passed else "verifier_score_missing_or_below_threshold",
            }
        ],
    }


def _build_reportbench_output(output_root: Path, verifier_mode: str) -> dict[str, Any]:
    summary = _load_json(_artifact_path(output_root, "eval_summary"))
    manifest = _load_json(_artifact_path(output_root, "result_manifest"))
    verifier = _load_json(_artifact_path(output_root, "verifier_review"))
    container_proof = (
        _load_json(_artifact_path(output_root, "container_proof"))
        if _artifact_path(output_root, "container_proof").exists()
        else {}
    )
    concrete_success = _concrete_run_succeeded(summary, manifest)
    verifier_score = verifier.get("score")
    verdict = _benchmark_verdict(verifier_score, concrete_success)
    return {
        "task_id": TASK_ID,
        "state": "passed" if verdict["passed"] else "failed",
        "primary_metric": "mean_outcome_reward",
        "primary_score": verifier_score,
        "benchmark_verdict": verdict,
        "reward": {
            "primary_metric": "mean_outcome_reward",
            "source_artifact": ARTIFACTS["eval_summary"],
            "value": summary.get("mean_outcome_reward"),
        },
        "runtime": {
            "elapsed_seconds": (summary.get("rollout_summary") or {}).get("elapsed_s")
            if isinstance(summary.get("rollout_summary"), dict)
            else None,
            "model": summary.get("model"),
        },
        "verifier": {
            "score": verifier_score,
            "summary": verifier.get("summary"),
            "criteria": verifier.get("criteria"),
            "latest_state": "done" if verdict["passed"] else "failed",
            "verifier_mode": verifier_mode,
        },
        "verifier_score": verifier_score,
        "model": summary.get("model"),
        "task": summary.get("task"),
        "requested_rollouts": summary.get("requested_rollouts"),
        "requested_total_llm_calls": summary.get("requested_total_llm_calls"),
        "requested_max_steps_per_rollout": summary.get("requested_max_steps_per_rollout"),
        "requested_rollout_concurrency": summary.get("requested_rollout_concurrency"),
        "mean_llm_calls_per_rollout": summary.get("mean_llm_calls_per_rollout"),
        "report_path": ARTIFACTS["reproduction"],
        "rollouts_path": ARTIFACTS["rollouts"],
        "result_manifest_path": ARTIFACTS["result_manifest"],
        "container_resource": container_proof,
        "artifacts": {
            "eval_summary": ARTIFACTS["eval_summary"],
            "rollouts": ARTIFACTS["rollouts"],
            "result_manifest": ARTIFACTS["result_manifest"],
            "container_proof": ARTIFACTS["container_proof"],
            "verifier_review": ARTIFACTS["verifier_review"],
            "report": ARTIFACTS["reproduction"],
        },
        "task_title": TASK_TITLE,
        "verifier_mode": verifier_mode,
        "completed_at": _utc_now(),
    }


def run(args: argparse.Namespace) -> int:
    output_root = _resolve_output_root(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    process = _run_baseline(output_root)
    if process.returncode != 0:
        summary_path = _artifact_path(output_root, "eval_summary")
        requested_rollouts = int(os.getenv("NANOHORIZON_ROLLOUTS") or "1")
        if not summary_path.exists():
            _write_json(
                summary_path,
                {
                    "benchmark": TASK_ID,
                    "task": "craftax",
                    "model": str(os.getenv("NANOHORIZON_MODEL") or "x-ai/grok-4.1-fast"),
                    "requested_rollouts": requested_rollouts,
                    "requested_total_llm_calls": requested_rollouts,
                    "requested_max_steps_per_rollout": 1,
                    "requested_llm_calls_per_rollout": 1,
                    "requested_rollout_concurrency": int(os.getenv("NANOHORIZON_ROLLOUT_CONCURRENCY") or "1"),
                    "mean_outcome_reward": 0.0,
                    "max_outcome_reward": 0.0,
                    "mean_llm_calls_per_rollout": 0.0,
                    "num_rollouts": 0,
                    "num_errors": requested_rollouts,
                    "runner_failure": {
                        "exit_code": int(process.returncode),
                        "stdout": process.stdout[:2000],
                        "stderr": process.stderr[:2000],
                    },
                },
            )
        rollouts_path = _artifact_path(output_root, "rollouts")
        if not rollouts_path.exists():
            rollouts_path.parent.mkdir(parents=True, exist_ok=True)
            rollouts_path.write_text("", encoding="utf-8")
    _write_result_manifest(output_root, process=process)
    _write_reproduction_report(output_root)
    return int(process.returncode)


def score(args: argparse.Namespace) -> int:
    output_root = _resolve_output_root(args.output_root)
    verifier = _compute_verifier_review(output_root, args.verifier_mode)
    _write_json(_artifact_path(output_root, "verifier_review"), verifier)
    output = _build_reportbench_output(output_root, args.verifier_mode)
    _write_json(_artifact_path(output_root, "reportbench_output"), output)
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the NanoHorizon Craftax hello-world task.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--output-root", default="")
    run_parser.set_defaults(func=run)

    score_parser = subparsers.add_parser("score")
    score_parser.add_argument("--output-root", default="")
    score_parser.add_argument("--verifier-mode", default="precheck")
    score_parser.set_defaults(func=score)

    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())

