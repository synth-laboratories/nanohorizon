from __future__ import annotations

import json
import os
import platform
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml


def now_utc_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    text = config_path.read_text(encoding="utf-8")
    payload = json.loads(text) if config_path.suffix.lower() == ".json" else yaml.safe_load(text)
    if not isinstance(payload, dict):
        raise ValueError(f"config must decode to an object: {config_path}")
    return payload


def resolve_path(path: str | Path, *, base_dir: str | Path | None = None) -> Path:
    target = Path(path).expanduser()
    if target.is_absolute():
        return target.resolve()
    anchor = Path(base_dir).expanduser().resolve() if base_dir is not None else Path.cwd().resolve()
    return (anchor / target).resolve()


def ensure_dir(path: str | Path) -> Path:
    target = Path(path).expanduser().resolve()
    target.mkdir(parents=True, exist_ok=True)
    return target


def write_json(path: str | Path, payload: Any) -> None:
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: str | Path, text: str) -> None:
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw in Path(path).expanduser().resolve().read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"jsonl row must be an object: {path}")
        rows.append(payload)
    return rows


def git_commit_or_unknown(repo_root: str | Path) -> str:
    try:
        completed = subprocess.run(
            ["git", "-C", str(Path(repo_root).expanduser().resolve()), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()
    except Exception:
        return "unknown"


def system_info() -> dict[str, Any]:
    return {
        "timestamp_utc": now_utc_iso(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "hostname": platform.node(),
        "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES", ""),
        "runpod_pod_id": os.getenv("RUNPOD_POD_ID", ""),
    }


class Timer:
    def __init__(self) -> None:
        self.start = time.time()
        self.started_at = now_utc_iso()

    @property
    def elapsed_seconds(self) -> float:
        return time.time() - self.start

    @property
    def elapsed_minutes(self) -> float:
        return self.elapsed_seconds / 60.0

    @property
    def ended_at(self) -> str:
        return now_utc_iso()
