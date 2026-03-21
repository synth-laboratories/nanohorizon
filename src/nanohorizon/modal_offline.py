from __future__ import annotations

# ruff: noqa: E402
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

import modal

REMOTE_SRC = Path("/root/nanohorizon/src")
if REMOTE_SRC.exists():
    sys.path.insert(0, str(REMOTE_SRC))

from nanohorizon.common import ensure_dir, now_utc_iso
from nanohorizon.modal_common import (
    ARTIFACT_DIR,
    GPU_OFFLINE,
    OFFLINE_VENV_ROOT,
    REMOTE_ROOT,
    offline_image,
    volume_mounts,
)

APP_NAME = os.getenv("NANOHORIZON_MODAL_OFFLINE_APP_NAME", "nanohorizon-crafter-offline")

app = modal.App(APP_NAME)
image = offline_image()
MIN_CONTAINERS = int(os.getenv("NANOHORIZON_MODAL_OFFLINE_MIN_CONTAINERS", "1"))
SCALEDOWN_WINDOW = int(os.getenv("NANOHORIZON_MODAL_OFFLINE_SCALEDOWN_WINDOW", str(30 * 60)))


def _default_output_dir() -> str:
    stamp = now_utc_iso().replace(":", "").replace("+00:00", "Z")
    return f"{ARTIFACT_DIR}/offline/{stamp}"


@app.function(
    image=image,
    gpu=GPU_OFFLINE,
    timeout=60 * 60,
    min_containers=MIN_CONTAINERS,
    scaledown_window=SCALEDOWN_WINDOW,
    volumes=volume_mounts(),
)
def run(
    *,
    config: str = "configs/crafter_offline_reference.yaml",
    output_dir: str = "",
    teacher_model: str = "Qwen/Qwen3.5-9B",
    teacher_api_key: str = "",
    teacher_enforce_eager: int = 0,
    teacher_startup_attempts: int = 240,
    teacher_startup_sleep_seconds: int = 2,
    min_teacher_reward: float | None = None,
    max_teacher_rows: int | None = None,
    filter_collect_wood: int | None = None,
    crafter_container_url: str = "",
    crafter_container_worker_token: str = "",
    teacher_inference_url: str = "",
) -> dict[str, object]:
    os.chdir(REMOTE_ROOT)
    destination = ensure_dir(output_dir or _default_output_dir())
    env = os.environ.copy()
    code_version = env.get("NANOHORIZON_CODE_VERSION", "unknown")
    env.update(
        {
            "PYTHONPATH": f"{REMOTE_ROOT}/src",
            "NANOHORIZON_OFFLINE_OUTPUT_ROOT": str(destination),
            "NANOHORIZON_AUTO_INSTALL": "0",
            "NANOHORIZON_START_LOCAL_TEACHER": "0" if teacher_inference_url else "1",
            "NANOHORIZON_TEACHER_MODEL": teacher_model,
            "NANOHORIZON_TEACHER_ENFORCE_EAGER": str(teacher_enforce_eager),
            "NANOHORIZON_TEACHER_STARTUP_ATTEMPTS": str(teacher_startup_attempts),
            "NANOHORIZON_TEACHER_STARTUP_SLEEP_SECONDS": str(teacher_startup_sleep_seconds),
            "NANOHORIZON_VENV_ROOT": OFFLINE_VENV_ROOT,
            "NANOHORIZON_MODAL_OFFLINE_APP_NAME": APP_NAME,
            "NANOHORIZON_CODE_VERSION": code_version,
        }
    )
    for name in ("NANOHORIZON_TEACHER_MAX_MODEL_LEN", "NANOHORIZON_TEACHER_REASONING_PARSER", "NANOHORIZON_TEACHER_TOOL_CALL_PARSER"):
        value = os.environ.get(name)
        if value is not None:
            env[name] = value
    if crafter_container_url:
        env["NANOHORIZON_CRAFTER_CONTAINER_URL"] = crafter_container_url
    if crafter_container_worker_token:
        env["NANOHORIZON_CRAFTER_CONTAINER_WORKER_TOKEN"] = crafter_container_worker_token
    if teacher_inference_url:
        env["NANOHORIZON_TEACHER_INFERENCE_URL"] = teacher_inference_url
    if teacher_api_key:
        env["NANOHORIZON_TEACHER_API_KEY"] = teacher_api_key
    if min_teacher_reward is not None:
        env["NANOHORIZON_MIN_TEACHER_REWARD"] = str(min_teacher_reward)
    if max_teacher_rows is not None:
        env["NANOHORIZON_MAX_TEACHER_ROWS"] = str(max_teacher_rows)
    if filter_collect_wood is not None:
        env["NANOHORIZON_FILTER_COLLECT_WOOD"] = str(filter_collect_wood)
    cmd = [
        "bash",
        f"{REMOTE_ROOT}/scripts/internal/run_crafter_offline_modal_worker.sh",
    ]
    env["NANOHORIZON_OFFLINE_CONFIG"] = f"{REMOTE_ROOT}/{config}"
    print("Remote offline command:", " ".join(shlex.quote(part) for part in cmd), flush=True)
    subprocess.check_call(cmd, env=env)
    metrics_path = destination / "metrics.json"
    payload = {
        "output_dir": str(destination),
        "metrics": json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {},
    }
    return payload


@app.local_entrypoint()
def main(
    config: str = "configs/crafter_offline_reference.yaml",
    output_dir: str = "",
    teacher_model: str = "Qwen/Qwen3.5-9B",
    teacher_api_key: str = "",
    teacher_enforce_eager: int = 0,
    teacher_startup_attempts: int = 240,
    teacher_startup_sleep_seconds: int = 2,
    min_teacher_reward: float | None = None,
    max_teacher_rows: int | None = None,
    filter_collect_wood: int | None = None,
    crafter_container_url: str = "",
    crafter_container_worker_token: str = "",
    teacher_inference_url: str = "",
) -> None:
    result = run.remote(
        config=config,
        output_dir=output_dir,
        teacher_model=teacher_model,
        teacher_api_key=teacher_api_key,
        teacher_enforce_eager=teacher_enforce_eager,
        teacher_startup_attempts=teacher_startup_attempts,
        teacher_startup_sleep_seconds=teacher_startup_sleep_seconds,
        min_teacher_reward=min_teacher_reward,
        max_teacher_rows=max_teacher_rows,
        filter_collect_wood=filter_collect_wood,
        crafter_container_url=crafter_container_url,
        crafter_container_worker_token=crafter_container_worker_token,
        teacher_inference_url=teacher_inference_url,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
