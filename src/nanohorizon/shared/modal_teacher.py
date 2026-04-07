# ruff: noqa: E402
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, cast

import httpx
import modal

REMOTE_SRC = Path("/root/nanohorizon/src")
if REMOTE_SRC.exists():
    sys.path.insert(0, str(REMOTE_SRC))

from nanohorizon.custom_vllm.runtime import enable_thinking_budget_support
from nanohorizon.shared.modal_common import GPU_TEACHER, rlvr_vllm_image, volume_mounts

APP_NAME = os.environ.get("NANOHORIZON_MODAL_TEACHER_APP_NAME", "nanohorizon-craftax-teacher").strip() or "nanohorizon-craftax-teacher"
VLLM_PORT = 8000
DEFAULT_MODEL = os.environ.get("NANOHORIZON_TEACHER_MODEL", "Qwen/Qwen3.5-9B").strip() or "Qwen/Qwen3.5-9B"
DEFAULT_SERVED_MODEL_NAME = os.environ.get("NANOHORIZON_TEACHER_SERVED_MODEL_NAME", "").strip()
DEFAULT_API_KEY = os.environ.get("NANOHORIZON_TEACHER_API_KEY", "dummy-local-key").strip() or "dummy-local-key"
DEFAULT_MAX_MODEL_LEN = int(os.environ.get("NANOHORIZON_TEACHER_MAX_MODEL_LEN", "4096").strip() or "4096")
DEFAULT_LORA_NAME = os.environ.get("NANOHORIZON_TEACHER_LORA_NAME", "").strip()
DEFAULT_LORA_PATH = os.environ.get("NANOHORIZON_TEACHER_LORA_PATH", "").strip()
DEFAULT_MAX_LORA_RANK = int(os.environ.get("NANOHORIZON_TEACHER_MAX_LORA_RANK", "16").strip() or "16")

app = modal.App(APP_NAME)
image = rlvr_vllm_image()


@app.cls(
    image=image,
    gpu=GPU_TEACHER,
    timeout=60 * 60 * 24,
    max_containers=1,
    scaledown_window=60 * 10,
    buffer_containers=int(os.environ.get("NANOHORIZON_TEACHER_BUFFER_CONTAINERS", "0")),
    volumes=volume_mounts(),
)
class TeacherServer:
    model: str = modal.parameter(default=DEFAULT_MODEL)
    served_model_name: str = modal.parameter(default=DEFAULT_SERVED_MODEL_NAME)
    api_key: str = modal.parameter(default=DEFAULT_API_KEY)
    max_model_len: int = modal.parameter(default=DEFAULT_MAX_MODEL_LEN)
    lora_name: str = modal.parameter(default=DEFAULT_LORA_NAME)
    lora_path: str = modal.parameter(default=DEFAULT_LORA_PATH)
    max_lora_rank: int = modal.parameter(default=DEFAULT_MAX_LORA_RANK)

    @modal.web_server(port=VLLM_PORT, startup_timeout=60 * 20)
    def serve(self) -> None:
        vllm_bin = "vllm"
        model = self.model.strip() or DEFAULT_MODEL
        served_model_name = self.served_model_name.strip() or model
        api_key = self.api_key.strip() or DEFAULT_API_KEY
        max_model_len = max(1024, int(self.max_model_len))
        lora_name = self.lora_name.strip()
        lora_path = self.lora_path.strip()
        max_lora_rank = max(1, int(self.max_lora_rank))
        runtime_env = {**os.environ, "PYTHONUNBUFFERED": "1"}
        existing_pythonpath = str(runtime_env.get("PYTHONPATH") or "").strip()
        runtime_env["PYTHONPATH"] = (
            f"{REMOTE_SRC}:{existing_pythonpath}" if existing_pythonpath else str(REMOTE_SRC)
        )
        cmd = [
            vllm_bin,
            "serve",
            model,
            "--served-model-name",
            served_model_name,
            "--host",
            "0.0.0.0",
            "--port",
            str(VLLM_PORT),
            "--max-model-len",
            str(max_model_len),
            "--max-num-seqs",
            "64",
            "--gpu-memory-utilization",
            "0.92",
            "--uvicorn-log-level",
            "info",
            "--enable-prefix-caching",
            "--reasoning-parser",
            "qwen3",
            "--enforce-eager",
            "--enable-auto-tool-choice",
            "--tool-call-parser",
            "qwen3_coder",
            "--api-key",
            api_key,
        ]
        if lora_path:
            cmd += [
                "--enable-lora",
                "--max-lora-rank",
                str(max_lora_rank),
                "--lora-modules",
                f"{(lora_name or 'policy-lora')}={lora_path}",
            ]
        cmd, env = enable_thinking_budget_support(
            cmd=cmd,
            env=runtime_env,
            model_ref=model,
        )
        print("Launching teacher vLLM:", " ".join(shlex.quote(x) for x in cmd), flush=True)
        process = subprocess.Popen(cmd, env=env)
        deadline = time.time() + (60 * 20)
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else None
        models_url = f"http://127.0.0.1:{VLLM_PORT}/v1/models"
        while time.time() < deadline:
            if process.poll() is not None:
                raise RuntimeError(f"teacher vLLM exited before readiness with code {process.returncode}")
            try:
                with httpx.Client(timeout=5.0) as client:
                    response = client.get(models_url, headers=headers)
                if response.status_code == 200:
                    return
            except Exception:
                pass
            time.sleep(1.0)
        raise RuntimeError("timed out waiting for teacher vLLM readiness")


@app.local_entrypoint()
def run_teacher_server(
    keepalive_s: int = 3600,
    model: str = DEFAULT_MODEL,
    served_model_name: str = DEFAULT_SERVED_MODEL_NAME,
    api_key: str = DEFAULT_API_KEY,
    max_model_len: int = DEFAULT_MAX_MODEL_LEN,
    lora_name: str = DEFAULT_LORA_NAME,
    lora_path: str = DEFAULT_LORA_PATH,
    max_lora_rank: int = DEFAULT_MAX_LORA_RANK,
) -> None:
    server = cast(Any, TeacherServer)(
        model=model,
        served_model_name=served_model_name,
        api_key=api_key,
        max_model_len=max_model_len,
        lora_name=lora_name,
        lora_path=lora_path,
        max_lora_rank=max_lora_rank,
    )
    print(f"vLLM endpoint: {server.serve.get_web_url()}", flush=True)
    time.sleep(max(1, keepalive_s))
