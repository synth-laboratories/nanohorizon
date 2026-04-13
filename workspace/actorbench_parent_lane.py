from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected JSON object in {path}")
    return payload


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def nested_get(payload: Any, dotted_key: str) -> Any:
    current = payload
    for part in str(dotted_key or "").split("."):
        if not part:
            continue
        if not isinstance(current, dict):
            return None
        current = current.get(part)
    return current


def resolve_output_root(raw: str | None, runs_root: Path) -> Path:
    if raw:
        return Path(raw).expanduser().resolve()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return runs_root / stamp


def artifact_path(output_root: Path, relative_path: str) -> Path:
    return output_root / relative_path


def resolve_parent_runner(config: dict[str, Any], wrapper_path: Path) -> Path:
    bundled = wrapper_path.with_name(str(config["bundled_parent_runner"]))
    if bundled.exists():
        return bundled
    evals_root = wrapper_path.resolve().parents[4]
    candidate = evals_root / "reportbench" / "lanes" / str(config["parent_task_id"]) / "workspace" / str(config["host_parent_runner"])
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"unable to resolve parent runner for {config['task_id']}")


def run_parent_action(config: dict[str, Any], wrapper_path: Path, action: str, output_root: Path, verifier_mode: str | None = None) -> int:
    parent_runner = resolve_parent_runner(config, wrapper_path)
    command = [sys.executable, str(parent_runner), action, "--output-root", str(output_root)]
    env = os.environ.copy()
    env.setdefault("NANOHORIZON_REPO_ROOT", str(wrapper_path.resolve().parents[1]))
    if verifier_mode:
        command.extend(["--verifier-mode", verifier_mode])
    completed = subprocess.run(command, cwd=str(output_root), env=env, check=False)
    return int(completed.returncode)


def _evaluate_check(output_root: Path, check: dict[str, Any]) -> tuple[float, str]:
    kind = str(check.get("kind") or "").strip()
    label = str(check.get("id") or kind)
    if kind == "all_files_exist":
        paths = [artifact_path(output_root, str(item)) for item in check.get("paths", [])]
        ok = all(path.exists() for path in paths)
        return (1.0 if ok else 0.0, f"{label}: {'ok' if ok else 'missing files'}")
    if kind == "json_equals":
        payload = load_json(artifact_path(output_root, str(check["path"])))
        actual = nested_get(payload, str(check["key"]))
        expected = check.get("value")
        ok = actual == expected
        return (1.0 if ok else 0.0, f"{label}: expected {expected!r}, got {actual!r}")
    if kind == "json_equals_map":
        payload = load_json(artifact_path(output_root, str(check["path"])))
        items = dict(check.get("items") or {})
        ok = True
        mismatches: list[str] = []
        for key, expected in items.items():
            actual = nested_get(payload, str(key))
            if actual != expected:
                ok = False
                mismatches.append(f"{key}={actual!r} expected {expected!r}")
        return (1.0 if ok else 0.0, f"{label}: {'ok' if ok else '; '.join(mismatches)}")
    if kind == "json_present":
        payload = load_json(artifact_path(output_root, str(check["path"])))
        actual = nested_get(payload, str(check["key"]))
        ok = actual is not None
        return (1.0 if ok else 0.0, f"{label}: {'ok' if ok else 'missing value'}")
    if kind == "json_present_any":
        payload = load_json(artifact_path(output_root, str(check["path"])))
        keys = [str(item) for item in check.get("keys", [])]
        ok = any(nested_get(payload, key) is not None for key in keys)
        return (1.0 if ok else 0.0, f"{label}: {'ok' if ok else 'no candidate values present'}")
    if kind == "json_truthy":
        payload = load_json(artifact_path(output_root, str(check["path"])))
        actual = nested_get(payload, str(check["key"]))
        ok = bool(actual)
        return (1.0 if ok else 0.0, f"{label}: {'ok' if ok else 'value is empty'}")
    if kind == "text_contains_all":
        text = artifact_path(output_root, str(check["path"])).read_text(encoding="utf-8")
        missing = [item for item in check.get("substrings", []) if str(item) not in text]
        ok = not missing
        return (1.0 if ok else 0.0, f"{label}: {'ok' if ok else f'missing {missing!r}'}")
    if kind == "text_contains_any":
        text = artifact_path(output_root, str(check["path"])).read_text(encoding="utf-8")
        candidates = [str(item) for item in check.get("substrings", [])]
        ok = any(item in text for item in candidates)
        return (1.0 if ok else 0.0, f"{label}: {'ok' if ok else 'no candidate substring matched'}")
    raise RuntimeError(f"unsupported actorbench check kind: {kind}")


def build_actorbench_verifier(config: dict[str, Any], output_root: Path) -> dict[str, Any]:
    criteria_payload: list[dict[str, Any]] = []
    notes: list[str] = []
    weighted_total = 0.0
    weight_sum = 0.0
    for check in config.get("criteria", []):
        score, note = _evaluate_check(output_root, dict(check))
        weight = float(check.get("weight") or 0.0)
        criteria_payload.append(
            {
                "id": str(check.get("id") or check.get("kind") or "criterion"),
                "score": round(score, 6),
                "weight": weight,
                "rationale": note,
            }
        )
        weighted_total += score * weight
        weight_sum += weight
        if score < 1.0:
            notes.append(note)
    actor_score = round(weighted_total / weight_sum, 6) if weight_sum else 0.0
    summary = str(config.get("summary") or f"ActorBench score for {config['task_id']}.")
    if not notes:
        notes.append("bundle satisfied the actor-focused checks")
    return {
        "task_id": config["task_id"],
        "parent_task_id": config["parent_task_id"],
        "target_actor_type": config.get("target_actor_type", "worker"),
        "score": actor_score,
        "summary": summary,
        "criteria": criteria_payload,
        "notes": notes,
        "completed_at": utc_now(),
    }


def build_actorbench_output(config: dict[str, Any], output_root: Path, verifier: dict[str, Any]) -> dict[str, Any]:
    parent_output_path = artifact_path(output_root, str(config["parent_output_path"]))
    parent_output = load_json(parent_output_path) if parent_output_path.exists() else {}
    primary_metric_artifact = artifact_path(output_root, str(config.get("primary_metric_artifact") or config["parent_output_path"]))
    primary_metric_value = None
    if primary_metric_artifact.exists() and config.get("primary_metric_key"):
        payload = load_json(primary_metric_artifact)
        primary_metric_value = nested_get(payload, str(config["primary_metric_key"]))
    if primary_metric_value is None:
        primary_metric_value = parent_output.get("primary_score")
    state = parent_output.get("state") or parent_output.get("status") or ("succeeded" if verifier.get("score") == 1.0 else "failed")
    return {
        "task_id": config["task_id"],
        "benchmark": "actorbench",
        "state": state,
        "target_actor_type": config.get("target_actor_type", "worker"),
        "parent_task_id": config["parent_task_id"],
        "actor_score": verifier.get("score"),
        "primary_metric": str(config.get("primary_metric_name") or "actor_score"),
        "primary_metric_value": primary_metric_value,
        "focus": config.get("focus"),
        "parent_output_path": str(config["parent_output_path"]),
        "parent_primary_score": parent_output.get("primary_score"),
        "parent_verifier_score": parent_output.get("verifier_score"),
        "report_path": str(config.get("report_path") or "reports/reproduction.md"),
        "completed_at": utc_now(),
    }


def run_lane_cli(config: dict[str, Any], argv: list[str] | None = None) -> int:
    wrapper_path = Path(config["wrapper_path"]).resolve()
    runs_root = wrapper_path.parents[1] / "runs" / str(config["task_id"])

    parser = argparse.ArgumentParser(description=str(config["task_id"]))
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--output-root", default=None)

    score_parser = subparsers.add_parser("score")
    score_parser.add_argument("--output-root", default=None)
    score_parser.add_argument("--verifier-mode", default="precheck")

    args = parser.parse_args(argv)
    output_root = resolve_output_root(getattr(args, "output_root", None), runs_root)
    output_root.mkdir(parents=True, exist_ok=True)

    if args.command == "run":
        return run_parent_action(config, wrapper_path, "run", output_root)

    parent_rc = run_parent_action(
        config,
        wrapper_path,
        "score",
        output_root,
        verifier_mode=str(getattr(args, "verifier_mode", "precheck")),
    )
    verifier = build_actorbench_verifier(config, output_root)
    write_json(artifact_path(output_root, "artifacts/actorbench_verifier_review.json"), verifier)
    output = build_actorbench_output(config, output_root, verifier)
    write_json(artifact_path(output_root, "artifacts/actorbench_output.json"), output)
    print(json.dumps(output, indent=2, sort_keys=True))
    return int(parent_rc)
