from __future__ import annotations

# ruff: noqa: E402
import json
import os
import shlex
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import modal

REMOTE_SRC = Path("/root/nanohorizon/src")
if REMOTE_SRC.exists():
    sys.path.insert(0, str(REMOTE_SRC))

from nanohorizon.shared.common import ensure_dir, now_utc_iso
from nanohorizon.shared.modal_common import (
    ARTIFACT_DIR,
    GPU_OFFLINE,
    OFFLINE_VENV_ROOT,
    REMOTE_ROOT,
    offline_worker_image,
    volume_mounts,
)

CRAFTAX_PORT = 8903

APP_NAME = os.getenv("NANOHORIZON_MODAL_OFFLINE_APP_NAME", "nanohorizon-craftax-offline")

app = modal.App(APP_NAME)
image = offline_worker_image()
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
    config: str = "configs/craftax_offline_reference.yaml",
    output_dir: str = "",
    teacher_model: str = "Qwen/Qwen3.5-9B",
    teacher_api_key: str = "",
    teacher_enforce_eager: int = 1,
    teacher_max_model_len: int | None = None,
    teacher_max_num_seqs: int | None = None,
    teacher_gpu_memory_utilization: float | None = None,
    teacher_startup_attempts: int = 240,
    teacher_startup_sleep_seconds: int = 2,
    min_teacher_reward: float | None = None,
    max_teacher_rows: int | None = None,
    filter_collect_wood: int | None = None,
    allowed_teacher_achievements: str = "",
    craftax_container_url: str = "",
    craftax_container_worker_token: str = "",
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
    if teacher_max_model_len is not None:
        env["NANOHORIZON_TEACHER_MAX_MODEL_LEN"] = str(teacher_max_model_len)
    if teacher_max_num_seqs is not None:
        env["NANOHORIZON_TEACHER_MAX_NUM_SEQS"] = str(teacher_max_num_seqs)
    if teacher_gpu_memory_utilization is not None:
        env["NANOHORIZON_TEACHER_GPU_MEMORY_UTILIZATION"] = str(teacher_gpu_memory_utilization)
    for name in (
        "NANOHORIZON_TEACHER_MAX_MODEL_LEN",
        "NANOHORIZON_TEACHER_MAX_NUM_SEQS",
        "NANOHORIZON_TEACHER_GPU_MEMORY_UTILIZATION",
        "NANOHORIZON_TEACHER_REASONING_PARSER",
        "NANOHORIZON_TEACHER_TOOL_CALL_PARSER",
    ):
        value = os.environ.get(name)
        if value is not None:
            env[name] = value
    # Start the Craftax shim locally if no external URL was provided.
    craftax_proc = None
    if not craftax_container_url:
        craftax_log = destination / "craftax_container.log"
        craftax_log.parent.mkdir(parents=True, exist_ok=True)
        craftax_env = {
            **os.environ,
            "NANOHORIZON_CRAFTAX_BIND_HOST": "0.0.0.0",
            "NANOHORIZON_CRAFTAX_BIND_PORT": str(CRAFTAX_PORT),
            "NANOHORIZON_CRAFTAX_BIND_HOST": "0.0.0.0",
            "NANOHORIZON_CRAFTAX_BIND_PORT": str(CRAFTAX_PORT),
            "CUDA_VISIBLE_DEVICES": "",
            "JAX_PLATFORMS": "cpu",
            "JAX_PLATFORM_NAME": "cpu",
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
            "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.0",
        }
        craftax_fh = open(str(craftax_log), "w")
        craftax_proc = subprocess.Popen(
            [sys.executable, "-m", "nanohorizon.craftax_core.http_shim"],
            env=craftax_env,
            stdout=craftax_fh,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        craftax_url = f"http://127.0.0.1:{CRAFTAX_PORT}"
        print(f"Starting Craftax shim on {craftax_url}", flush=True)
        # Wait for health
        deadline = time.time() + 60.0
        while time.time() < deadline:
            try:
                with urllib.request.urlopen(f"{craftax_url}/health", timeout=3) as resp:
                    if 200 <= resp.status < 300:
                        print(f"Craftax runtime healthy at {craftax_url}", flush=True)
                        break
            except Exception:
                pass
            time.sleep(1.0)
        else:
            print("WARNING: Craftax runtime did not become healthy within 60s", flush=True)
        env["NANOHORIZON_CRAFTAX_CONTAINER_URL"] = craftax_url
        env["NANOHORIZON_CRAFTAX_CONTAINER_URL"] = craftax_url
    elif craftax_container_url:
        env["NANOHORIZON_CRAFTAX_CONTAINER_URL"] = craftax_container_url
        env["NANOHORIZON_CRAFTAX_CONTAINER_URL"] = craftax_container_url
    if craftax_container_worker_token:
        env["NANOHORIZON_CRAFTAX_CONTAINER_WORKER_TOKEN"] = craftax_container_worker_token
        env["NANOHORIZON_CRAFTAX_CONTAINER_WORKER_TOKEN"] = craftax_container_worker_token
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
    if allowed_teacher_achievements.strip():
        env["NANOHORIZON_ALLOWED_TEACHER_ACHIEVEMENTS"] = allowed_teacher_achievements.strip()
    cmd = [
        "bash",
        f"{REMOTE_ROOT}/scripts/internal/run_craftax_offline_modal_worker.sh",
    ]
    env["NANOHORIZON_OFFLINE_CONFIG"] = f"{REMOTE_ROOT}/{config}"
    print("Remote offline command:", " ".join(shlex.quote(part) for part in cmd), flush=True)
    try:
        subprocess.check_call(cmd, env=env)
    finally:
        if craftax_proc is not None:
            try:
                os.killpg(os.getpgid(craftax_proc.pid), signal.SIGTERM)
            except (ProcessLookupError, OSError):
                pass
    metrics_path = destination / "metrics.json"
    payload = {
        "output_dir": str(destination),
        "metrics": json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {},
    }
    return payload


@app.local_entrypoint()
def main(
    config: str = "configs/craftax_offline_reference.yaml",
    output_dir: str = "",
    teacher_model: str = "Qwen/Qwen3.5-9B",
    teacher_api_key: str = "",
    teacher_enforce_eager: int = 1,
    teacher_max_model_len: int | None = None,
    teacher_max_num_seqs: int | None = None,
    teacher_gpu_memory_utilization: float | None = None,
    teacher_startup_attempts: int = 240,
    teacher_startup_sleep_seconds: int = 2,
    min_teacher_reward: float | None = None,
    max_teacher_rows: int | None = None,
    filter_collect_wood: int | None = None,
    allowed_teacher_achievements: str = "",
    craftax_container_url: str = "",
    craftax_container_worker_token: str = "",
    teacher_inference_url: str = "",
) -> None:
    result = run.remote(
        config=config,
        output_dir=output_dir,
        teacher_model=teacher_model,
        teacher_api_key=teacher_api_key,
        teacher_enforce_eager=teacher_enforce_eager,
        teacher_max_model_len=teacher_max_model_len,
        teacher_max_num_seqs=teacher_max_num_seqs,
        teacher_gpu_memory_utilization=teacher_gpu_memory_utilization,
        teacher_startup_attempts=teacher_startup_attempts,
        teacher_startup_sleep_seconds=teacher_startup_sleep_seconds,
        min_teacher_reward=min_teacher_reward,
        max_teacher_rows=max_teacher_rows,
        filter_collect_wood=filter_collect_wood,
        allowed_teacher_achievements=allowed_teacher_achievements,
        craftax_container_url=craftax_container_url,
        craftax_container_worker_token=craftax_container_worker_token,
        teacher_inference_url=teacher_inference_url,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
