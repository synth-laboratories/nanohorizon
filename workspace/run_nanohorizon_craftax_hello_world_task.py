#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import json
import os
import socket
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib import request

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
    "craftax_scorecard": "artifacts/craftax_scorecard.json",
    "craftax_rollout_media_json": "artifacts/craftax_rollout_media.json",
    "craftax_experiment_result": "artifacts/craftax_experiment_result.json",
    "reproduction": "reports/reproduction.md",
}

LOCAL_HTTP_TIMEOUT_SECONDS = 180.0


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


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text:
            continue
        payload = json.loads(text)
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


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
        Path(sys.executable),
        nanohorizon_root / ".venv" / "bin" / "python",
    ]
    for candidate in candidates:
        if candidate is None or not candidate.exists():
            continue
        return str(candidate.absolute())
    return sys.executable


def _apply_craftax_runtime_env(env: dict[str, str], nanohorizon_root: Path) -> None:
    pythonpath_parts = [
        str(nanohorizon_root),
        str(nanohorizon_root / "src"),
    ]
    existing_pythonpath = str(env.get("PYTHONPATH") or "").strip()
    if existing_pythonpath:
        pythonpath_parts.append(existing_pythonpath)
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)
    env["NANOHORIZON_REPO_ROOT"] = str(nanohorizon_root)
    env.setdefault("JAX_PLATFORMS", "cpu")
    env.setdefault("JAX_PLATFORM_NAME", "cpu")
    env.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    env.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.25")


def _allocate_local_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _probe_http_health(url: str, timeout_seconds: float = 2.0) -> dict[str, Any]:
    with request.urlopen(f"{url.rstrip('/')}/health", timeout=timeout_seconds) as response:
        body = response.read().decode("utf-8", errors="replace")
        return {
            "status_code": int(response.status),
            "ok": 200 <= int(response.status) < 300,
            "body_preview": body[:1000],
        }


def _start_local_http_resource(
    output_root: Path,
    *,
    nanohorizon_python: str,
    env: dict[str, str],
) -> tuple[subprocess.Popen[str], str]:
    port = int(os.getenv("NANOHORIZON_CRAFTAX_BIND_PORT") or _allocate_local_port())
    service_env = env.copy()
    service_env["NANOHORIZON_CRAFTAX_BIND_HOST"] = "127.0.0.1"
    service_env["NANOHORIZON_CRAFTAX_BIND_PORT"] = str(port)
    service_env.setdefault("NANOHORIZON_CRAFTAX_UVICORN_WORKERS", "1")
    service_url = f"http://127.0.0.1:{port}"
    stdout_path = output_root / "artifacts" / "craftax_http_stdout.log"
    stderr_path = output_root / "artifacts" / "craftax_http_stderr.log"
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stdout_handle = stdout_path.open("w", encoding="utf-8")
    stderr_handle = stderr_path.open("w", encoding="utf-8")
    process = subprocess.Popen(
        [nanohorizon_python, "-m", "nanohorizon.craftax_core.http_shim"],
        cwd=str(output_root),
        env=service_env,
        stdout=stdout_handle,
        stderr=stderr_handle,
        text=True,
    )
    deadline = time.monotonic() + LOCAL_HTTP_TIMEOUT_SECONDS
    last_error = ""
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                "Craftax local HTTP resource exited before /health became ready. "
                f"exit_code={process.returncode} stdout={stdout_path} stderr={stderr_path}"
            )
        try:
            health = _probe_http_health(service_url)
            if health.get("ok"):
                return process, service_url
        except Exception as exc:  # noqa: BLE001
            last_error = repr(exc)
        time.sleep(1.0)
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)
    raise RuntimeError(
        "Craftax local HTTP resource did not become healthy. "
        f"last_error={last_error} stdout={stdout_path} stderr={stderr_path}"
    )


def _stop_local_http_resource(process: subprocess.Popen[str] | None) -> None:
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)


def _run_baseline(output_root: Path) -> subprocess.CompletedProcess[str]:
    nanohorizon_root = _resolve_nanohorizon_root()
    nanohorizon_python = _resolve_nanohorizon_python(nanohorizon_root)
    summary_path = _artifact_path(output_root, "eval_summary")
    rollouts_path = _artifact_path(output_root, "rollouts")
    env = os.environ.copy()
    _apply_craftax_runtime_env(env, nanohorizon_root)
    local_http_process: subprocess.Popen[str] | None = None
    explicit_url = str(env.get("NANOHORIZON_CRAFTAX_CONTAINER_URL") or "").strip()
    direct_requested = str(env.get("NANOHORIZON_CRAFTAX_RESOURCE_MODE") or "").strip() == "direct"
    try:
        if not explicit_url and not direct_requested:
            local_http_process, service_url = _start_local_http_resource(
                output_root,
                nanohorizon_python=nanohorizon_python,
                env=env,
            )
            env["NANOHORIZON_CRAFTAX_CONTAINER_URL"] = service_url
            env.pop("NANOHORIZON_ALLOW_DIRECT_LOCAL", None)
        elif direct_requested:
            env["NANOHORIZON_ALLOW_DIRECT_LOCAL"] = "1"
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
        if local_http_process is not None:
            proof = _load_json(_artifact_path(output_root, "container_proof"))
            proof.update(
                {
                    "mode": "local_http",
                    "managed_by": "workspace/run_nanohorizon_craftax_hello_world_task.py",
                    "resource_url": env["NANOHORIZON_CRAFTAX_CONTAINER_URL"],
                    "stdout_log": "artifacts/craftax_http_stdout.log",
                    "stderr_log": "artifacts/craftax_http_stderr.log",
                }
            )
            _write_json(_artifact_path(output_root, "container_proof"), proof)
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
    finally:
        _stop_local_http_resource(local_http_process)


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


def _rollout_reward(rollout: dict[str, Any]) -> float:
    reward_info = rollout.get("reward_info")
    if isinstance(reward_info, dict):
        objectives = reward_info.get("outcome_objectives")
        if isinstance(objectives, dict):
            for key in ("unique_achievements", "reward", "native_env_reward_total"):
                try:
                    value = objectives.get(key)
                    if value is not None:
                        return float(value)
                except (TypeError, ValueError):
                    pass
        try:
            return float(reward_info.get("outcome_reward", 0.0))
        except (TypeError, ValueError):
            return 0.0
    return 0.0


def _rollout_achievements(rollout: dict[str, Any]) -> list[str]:
    metadata = rollout.get("metadata")
    if isinstance(metadata, dict) and isinstance(metadata.get("achievements"), list):
        return [str(item).strip() for item in metadata["achievements"] if str(item).strip()]
    reward_info = rollout.get("reward_info")
    if isinstance(reward_info, dict):
        details = reward_info.get("details")
        if isinstance(details, dict) and isinstance(details.get("achievements"), list):
            return [str(item).strip() for item in details["achievements"] if str(item).strip()]
    return []


def _rollout_seed(rollout: dict[str, Any]) -> int | None:
    metadata = rollout.get("metadata")
    if isinstance(metadata, dict):
        try:
            value = metadata.get("seed")
            if value is not None:
                return int(value)
        except (TypeError, ValueError):
            pass
    try:
        value = rollout.get("_request_seed") or rollout.get("seed")
        if value is not None:
            return int(value)
    except (TypeError, ValueError):
        pass
    return None


def _relative_path(path: Path, output_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(output_root.resolve()))
    except ValueError:
        return str(path)


def _gif_media_ref(output_root: Path, rollout: dict[str, Any]) -> dict[str, Any] | None:
    media = rollout.get("media")
    if not isinstance(media, dict):
        return None
    raw_data_url = str(media.get("data_url") or media.get("gif_url") or "").strip()
    if raw_data_url.startswith("data:image/gif;base64,"):
        return {
            "kind": "gif",
            "content_type": "image/gif",
            "path": str(media.get("gif_path") or ""),
            "url": raw_data_url,
            "data_url": raw_data_url,
            "file_size_bytes": int(media.get("file_size_bytes") or 0),
        }
    raw_gif_path = str(media.get("gif_path") or "").strip()
    if not raw_gif_path:
        return None
    gif_path = Path(raw_gif_path)
    if not gif_path.is_absolute():
        gif_path = output_root / gif_path
    if not gif_path.exists():
        return None
    data_url = "data:image/gif;base64," + base64.b64encode(gif_path.read_bytes()).decode("ascii")
    return {
        "kind": "gif",
        "content_type": "image/gif",
        "path": _relative_path(gif_path, output_root),
        "url": data_url,
        "data_url": data_url,
        "file_size_bytes": gif_path.stat().st_size,
    }


def _build_rollout_detail(output_root: Path, rollout: dict[str, Any], index: int) -> dict[str, Any]:
    rollout_id = str(
        rollout.get("rollout_id")
        or rollout.get("trial_id")
        or rollout.get("trace_correlation_id")
        or f"rollout_{index:02d}"
    )
    media_ref = _gif_media_ref(output_root, rollout)
    media_refs = [media_ref] if media_ref is not None else []
    detail = {
        "rollout_id": rollout_id,
        "label": f"Rollout {index + 1}",
        "success_status": str(rollout.get("success_status") or ("error" if rollout.get("error") else "unknown")),
        "outcome_reward": _rollout_reward(rollout),
        "reward": _rollout_reward(rollout),
        "seed": _rollout_seed(rollout),
        "achievements": _rollout_achievements(rollout),
        "media_refs": media_refs,
    }
    if media_ref is not None:
        detail["gif_url"] = media_ref["data_url"]
    if rollout.get("error"):
        detail["error"] = str(rollout.get("error") or "")
    return detail


def _write_open_research_outputs(output_root: Path) -> None:
    summary = _load_json(_artifact_path(output_root, "eval_summary"))
    manifest = _load_json(_artifact_path(output_root, "result_manifest"))
    rollouts = _load_jsonl(_artifact_path(output_root, "rollouts"))
    rollout_details = [
        _build_rollout_detail(output_root, rollout, index)
        for index, rollout in enumerate(rollouts)
        if isinstance(rollout, dict)
    ]
    requested_rollouts = int(summary.get("requested_rollouts") or len(rollout_details) or 0)
    successful_rollouts = [
        item
        for item in rollout_details
        if str(item.get("success_status") or "").lower() == "success" and not item.get("error")
    ]
    media_rollouts = [item for item in rollout_details if item.get("media_refs")]
    rewards = [float(item.get("outcome_reward") or 0.0) for item in successful_rollouts]
    score = float(summary.get("mean_outcome_reward") or (sum(rewards) / len(rewards) if rewards else 0.0))
    media_manifest = {
        "schema_version": "open_research.rollout_media.v1",
        "application_id": "craftax",
        "track": "craftax",
        "generated_at": _utc_now(),
        "requested_rollouts": requested_rollouts,
        "rollout_count": len(rollout_details),
        "media_rollout_count": len(media_rollouts),
        "rollouts": rollout_details,
    }
    scorecard = {
        "schema_version": "open_research.scorecard.v1",
        "application_id": "craftax",
        "track": "craftax",
        "primary_metric": "mean_outcome_reward",
        "primary_score": score,
        "requested_rollouts": requested_rollouts,
        "observed_rollouts": len(rollout_details),
        "successful_rollouts": len(successful_rollouts),
        "media_rollouts": len(media_rollouts),
        "achievement_count": len({name for item in rollout_details for name in item.get("achievements", [])}),
        "status": manifest.get("status"),
        "model": summary.get("model"),
    }
    experiment_result = {
        "schema_version": "open_research.experiment_result.v1",
        "application_id": "craftax",
        "track": "craftax",
        "benchmark": TASK_ID,
        "status": manifest.get("status"),
        "primary_metric": "mean_outcome_reward",
        "score": score,
        "aggregate_reward": score,
        "requested_rollouts": requested_rollouts,
        "observed_rollouts": len(rollout_details),
        "successful_rollouts": len(successful_rollouts),
        "media_rollouts": len(media_rollouts),
        "achievements": sorted({name for item in rollout_details for name in item.get("achievements", [])}),
        "rollout_details": rollout_details,
        "media": media_manifest,
        "source_artifacts": {
            "scorecard": ARTIFACTS["craftax_scorecard"],
            "rollout_media": ARTIFACTS["craftax_rollout_media_json"],
            "eval_summary": ARTIFACTS["eval_summary"],
            "rollouts": ARTIFACTS["rollouts"],
            "result_manifest": ARTIFACTS["result_manifest"],
        },
        "model": summary.get("model"),
        "completed_at": manifest.get("completed_at") or _utc_now(),
    }
    _write_json(_artifact_path(output_root, "craftax_rollout_media_json"), media_manifest)
    _write_json(_artifact_path(output_root, "craftax_scorecard"), scorecard)
    _write_json(_artifact_path(output_root, "craftax_experiment_result"), experiment_result)


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
            "craftax_scorecard": ARTIFACTS["craftax_scorecard"],
            "craftax_rollout_media": ARTIFACTS["craftax_rollout_media_json"],
            "craftax_experiment_result": ARTIFACTS["craftax_experiment_result"],
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
        requested_rollouts = int(os.getenv("NANOHORIZON_ROLLOUTS") or "10")
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
                    "requested_rollout_concurrency": int(os.getenv("NANOHORIZON_ROLLOUT_CONCURRENCY") or "10"),
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
    _write_open_research_outputs(output_root)
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
