# ruff: noqa: E402

import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path, PurePosixPath
from typing import Any, cast

import httpx
import modal

REMOTE_SRC = Path("/root/nanohorizon/src")
if REMOTE_SRC.exists():
    sys.path.insert(0, str(REMOTE_SRC))

from nanohorizon.common import ensure_dir, load_config, now_utc_iso
from nanohorizon.custom_vllm.runtime import enable_thinking_budget_support
from nanohorizon.modal_common import (
    GPU_RLVR,
    OFFLINE_VENV_ROOT,
    PROJECT_ROOT,
    RECORDS_DIR,
    REMOTE_ROOT,
    offline_image,
    volume_mounts,
)
from nanohorizon.rlvr_training import run_training

APP_NAME = "nanohorizon-crafter-rlvr"
CRAFTER_PORT = 8903
VLLM_PORT = 8000
PROXY_PORT = 8100
DEFAULT_REQUEST_TIMEOUT_S = 60 * 20
CRAFTER_CORE_ROOT = Path(
    os.getenv("NANOHORIZON_CRAFTER_CORE_ROOT") or str(PROJECT_ROOT.parent / "crafter-rs")
).expanduser()
REMOTE_CRAFTER_CORE = PurePosixPath("/root/crafter-rs")
REMOTE_CRAFTER_BIN = PurePosixPath(f"{REMOTE_ROOT}/containers/crafter_rs/target/release/crafter-rs-container")
DEFAULT_INFERENCE_API_KEY = (
    os.getenv("NANOHORIZON_RLVR_INFERENCE_API_KEY", "nanohorizon-rlvr-key").strip()
    or "nanohorizon-rlvr-key"
)

app = modal.App(APP_NAME)


def _default_output_dir() -> str:
    stamp = now_utc_iso().replace(":", "").replace("+00:00", "Z")
    return f"{RECORDS_DIR}/rlvr_20min_2xa100_40gb/{stamp}_reference_baseline"


def _rlvr_runtime_image() -> modal.Image:
    if not CRAFTER_CORE_ROOT.exists():
        raise RuntimeError(f"missing crafter-rs checkout: {CRAFTER_CORE_ROOT}")
    return (
        offline_image()
        .pip_install("fastapi>=0.115.0", "uvicorn>=0.32.0")
        .apt_install("cargo", "rustc")
        .add_local_dir(CRAFTER_CORE_ROOT.as_posix(), remote_path=str(REMOTE_CRAFTER_CORE))
        .run_commands(
            f"cd {REMOTE_ROOT}/containers/crafter_rs && cargo build --release --bin crafter-rs-container"
        )
    )


image = _rlvr_runtime_image()


def _pythonpath_with_repo() -> str:
    repo_src = f"{REMOTE_ROOT}/src"
    existing = str(os.environ.get("PYTHONPATH") or "").strip()
    if not existing:
        return repo_src
    parts = [repo_src, *[item for item in existing.split(os.pathsep) if item]]
    deduped: list[str] = []
    for part in parts:
        if part not in deduped:
            deduped.append(part)
    return os.pathsep.join(deduped)


def _wait_for_health(url: str, *, require_upstream_ready: bool = False) -> dict[str, Any]:
    deadline = time.time() + DEFAULT_REQUEST_TIMEOUT_S
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            with httpx.Client(timeout=10.0) as client:
                response = client.get(url)
            if response.status_code == 200:
                payload = response.json()
                if not require_upstream_ready or bool(payload.get("upstream_ready", False)):
                    return payload
                last_error = RuntimeError(f"upstream not ready: {payload}")
            else:
                last_error = RuntimeError(f"health returned HTTP {response.status_code}")
        except Exception as exc:
            last_error = exc
        time.sleep(1.0)
    raise RuntimeError(f"timed out waiting for {url}: {last_error!r}")


@app.cls(
    image=image,
    timeout=60 * 60,
    scaledown_window=60 * 10,
    volumes=volume_mounts(),
)
class CrafterService:
    @modal.web_server(port=CRAFTER_PORT, startup_timeout=60 * 10)
    def serve(self) -> None:
        runtime_env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONPATH": _pythonpath_with_repo()}
        cmd = [str(REMOTE_CRAFTER_BIN)]
        print("Launching Crafter service:", " ".join(shlex.quote(part) for part in cmd), flush=True)
        subprocess.Popen(cmd, env=runtime_env)


@app.cls(
    image=image,
    gpu=GPU_RLVR,
    timeout=60 * 60 * 24,
    scaledown_window=60 * 10,
    volumes=volume_mounts(),
)
@modal.concurrent(max_inputs=32)
class RLVRInferenceServer:
    model: str = modal.parameter(default="Qwen/Qwen3.5-4B")
    served_model_name: str = modal.parameter(default="qwen35-4b-rlvr")
    api_key: str = modal.parameter(default=DEFAULT_INFERENCE_API_KEY)
    max_model_len: int = modal.parameter(default=4096)
    max_lora_rank: int = modal.parameter(default=64)

    @modal.web_server(port=PROXY_PORT, startup_timeout=60 * 20)
    def serve(self) -> None:
        vllm_bin = f"{OFFLINE_VENV_ROOT}/teacher/bin/vllm"
        model = self.model.strip() or "Qwen/Qwen3.5-4B"
        served_model_name = self.served_model_name.strip() or "qwen35-4b-rlvr"
        api_key = self.api_key.strip() or DEFAULT_INFERENCE_API_KEY
        max_model_len = max(1024, int(self.max_model_len))
        max_lora_rank = max(1, int(self.max_lora_rank))
        runtime_env = {**os.environ, "PYTHONUNBUFFERED": "1"}
        runtime_env["PYTHONPATH"] = _pythonpath_with_repo()
        runtime_env["NANOHORIZON_RLVR_PROXY_UPSTREAM_BASE_URL"] = f"http://127.0.0.1:{VLLM_PORT}/v1"
        runtime_env["NANOHORIZON_RLVR_PROXY_API_KEY"] = api_key
        runtime_env["NANOHORIZON_RLVR_PROXY_SERVED_MODEL_NAME"] = served_model_name
        runtime_env["NANOHORIZON_RLVR_PROXY_DEFAULT_REQUEST_MODEL"] = served_model_name
        runtime_env["NANOHORIZON_RLVR_PROXY_STATE_PATH"] = "/tmp/nanohorizon_rlvr_proxy_state.json"

        vllm_cmd = [
            vllm_bin,
            "serve",
            model,
            "--served-model-name",
            served_model_name,
            "--host",
            "127.0.0.1",
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
            "--enable-auto-tool-choice",
            "--tool-call-parser",
            "qwen3_coder",
            "--enable-lora",
            "--max-lora-rank",
            str(max_lora_rank),
            "--api-key",
            api_key,
            "--enforce-eager",
        ]
        vllm_cmd, runtime_env = enable_thinking_budget_support(
            cmd=vllm_cmd,
            env=runtime_env,
            model_ref=model,
        )
        proxy_cmd = [
            sys.executable,
            "-m",
            "uvicorn",
            "nanohorizon.rlvr_inference_proxy:app",
            "--host",
            "0.0.0.0",
            "--port",
            str(PROXY_PORT),
        ]
        print("Launching RLVR vLLM:", " ".join(shlex.quote(part) for part in vllm_cmd), flush=True)
        subprocess.Popen(vllm_cmd, env=runtime_env)
        print("Launching RLVR proxy:", " ".join(shlex.quote(part) for part in proxy_cmd), flush=True)
        subprocess.Popen(proxy_cmd, env=runtime_env)


@app.function(
    image=image,
    gpu=GPU_RLVR,
    timeout=60 * 60,
    volumes=volume_mounts(),
)
def run(
    *,
    config: str = "configs/crafter_rlvr_qwen35_4b_2xa100_20min.yaml",
    output_dir: str = "",
    container_url: str,
    inference_url: str,
    inference_admin_url: str,
    inference_api_key: str,
    request_model: str,
) -> dict[str, object]:
    os.chdir(REMOTE_ROOT)
    destination = ensure_dir(output_dir or _default_output_dir())
    payload = run_training(
        config_path=f"{REMOTE_ROOT}/{config}",
        output_dir=str(destination),
        container_url=container_url,
        inference_url=inference_url,
        inference_admin_url=inference_admin_url,
        inference_api_key=inference_api_key,
        request_model=request_model,
    )
    return payload


@app.local_entrypoint()
def main(
    config: str = "configs/crafter_rlvr_qwen35_4b_2xa100_20min.yaml",
    output_dir: str = "",
) -> None:
    config_path = PROJECT_ROOT / config
    config_payload = load_config(config_path)
    model_name = str(config_payload.get("model", {}).get("model") or "Qwen/Qwen3.5-4B")
    served_model_name = str(
        config_payload.get("model", {}).get("served_model_name") or "qwen35-4b-rlvr"
    ).strip() or "qwen35-4b-rlvr"
    max_model_len = int(
        config_payload.get("inference", {}).get("max_model_len")
        or config_payload.get("training", {}).get("max_length")
        or 4096
    )
    inference_api_key = str(
        os.getenv("NANOHORIZON_RLVR_INFERENCE_API_KEY")
        or config_payload.get("inference", {}).get("api_key")
        or DEFAULT_INFERENCE_API_KEY
    ).strip() or DEFAULT_INFERENCE_API_KEY

    crafter_service = cast(Any, CrafterService)()
    inference_service = cast(Any, RLVRInferenceServer)(
        model=model_name,
        served_model_name=served_model_name,
        api_key=inference_api_key,
        max_model_len=max_model_len,
        max_lora_rank=int(config_payload.get("training", {}).get("lora_rank", 16)),
    )
    container_url = crafter_service.serve.get_web_url().rstrip("/")
    inference_base_url = inference_service.serve.get_web_url().rstrip("/")

    crafter_health = _wait_for_health(f"{container_url}/health")
    inference_health = _wait_for_health(f"{inference_base_url}/health", require_upstream_ready=True)
    result = run.remote(
        config=config,
        output_dir=output_dir,
        container_url=container_url,
        inference_url=f"{inference_base_url}/v1/chat/completions",
        inference_admin_url=f"{inference_base_url}/admin",
        inference_api_key=inference_api_key,
        request_model=served_model_name,
    )
    print(
        json.dumps(
            {
                "container_url": container_url,
                "inference_url": f"{inference_base_url}/v1/chat/completions",
                "container_health": crafter_health,
                "inference_health": inference_health,
                "result": result,
            },
            indent=2,
            sort_keys=True,
        )
    )
