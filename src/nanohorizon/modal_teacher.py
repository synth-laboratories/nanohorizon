# ruff: noqa: E402
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, cast

import modal

REMOTE_SRC = Path("/root/nanohorizon/src")
if REMOTE_SRC.exists():
    sys.path.insert(0, str(REMOTE_SRC))

from nanohorizon.custom_vllm.runtime import enable_thinking_budget_support
from nanohorizon.modal_common import GPU_TEACHER, OFFLINE_VENV_ROOT, offline_image, volume_mounts

APP_NAME = os.environ.get("NANOHORIZON_MODAL_TEACHER_APP_NAME", "nanohorizon-crafter-teacher").strip() or "nanohorizon-crafter-teacher"
VLLM_PORT = 8000
DEFAULT_MODEL = os.environ.get("NANOHORIZON_TEACHER_MODEL", "Qwen/Qwen3.5-9B").strip() or "Qwen/Qwen3.5-9B"
DEFAULT_SERVED_MODEL_NAME = os.environ.get("NANOHORIZON_TEACHER_SERVED_MODEL_NAME", "").strip()
DEFAULT_API_KEY = os.environ.get("NANOHORIZON_TEACHER_API_KEY", "dummy-local-key").strip() or "dummy-local-key"
DEFAULT_MAX_MODEL_LEN = int(os.environ.get("NANOHORIZON_TEACHER_MAX_MODEL_LEN", "4096").strip() or "4096")
DEFAULT_LORA_NAME = os.environ.get("NANOHORIZON_TEACHER_LORA_NAME", "").strip()
DEFAULT_LORA_PATH = os.environ.get("NANOHORIZON_TEACHER_LORA_PATH", "").strip()
DEFAULT_MAX_LORA_RANK = int(os.environ.get("NANOHORIZON_TEACHER_MAX_LORA_RANK", "16").strip() or "16")

app = modal.App(APP_NAME)
image = offline_image()


@app.cls(
    image=image,
    gpu=GPU_TEACHER,
    timeout=60 * 60 * 24,
    scaledown_window=60 * 10,
    volumes=volume_mounts(),
)
@modal.concurrent(max_inputs=32)
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
        vllm_bin = f"{OFFLINE_VENV_ROOT}/teacher/bin/vllm"
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
        subprocess.Popen(cmd, env=env)


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
