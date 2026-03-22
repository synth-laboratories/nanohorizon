# ruff: noqa: E402

import asyncio
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

from nanohorizon.common import ensure_dir, load_config, now_utc_iso, write_json
from nanohorizon.crafter_data import collect_rollouts_concurrently_with_summary
from nanohorizon.custom_vllm.runtime import (
    build_thinking_budget_request_overrides,
    enable_thinking_budget_support,
)
from nanohorizon.modal_common import (
    COMMON_PACKAGES,
    GPU_PROMPT_OPT,
    OFFLINE_VENV_ROOT,
    PROJECT_ROOT,
    RECORDS_DIR,
    REMOTE_ROOT,
    VLLM_BASE_IMAGE,
    VLLM_IMAGE_PYTHON_VERSION,
    _cuda_base_image,
    prompt_image,
    volume_mounts,
)

APP_NAME = "nanohorizon-crafter-prompt-opt"
CRAFTER_PORT = 8903
VLLM_PORT = 8000
DEFAULT_REQUEST_TIMEOUT_S = 60 * 20
DEFAULT_MODEL = "Qwen/Qwen3.5-4B"
DEFAULT_SERVED_MODEL_NAME = "qwen35-4b-prompt-opt"
DEFAULT_API_KEY = "nanohorizon-prompt-opt-key"
DEFAULT_MAX_MODEL_LEN = 8192
CRAFTER_CORE_ROOT = Path(
    os.getenv("NANOHORIZON_CRAFTER_CORE_ROOT") or str(PROJECT_ROOT.parent / "crafter-rs")
).expanduser()
REMOTE_CRAFTER_CORE = PurePosixPath("/root/crafter-rs")
REMOTE_CRAFTER_BIN = PurePosixPath(
    f"{REMOTE_ROOT}/containers/crafter_rs/target/release/crafter-rs-container"
)

app = modal.App(APP_NAME)


def _default_output_dir() -> str:
    stamp = now_utc_iso().replace(":", "").replace("+00:00", "Z")
    return f"{RECORDS_DIR}/prompt_opt_1usd_gpt54_family/{stamp}_reference_baseline"


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


def _normalize_inference_url(raw_url: str) -> str:
    url = str(raw_url or "").strip()
    if not url:
        return ""
    if url.endswith("/chat/completions"):
        return url
    if url.endswith("/v1"):
        return f"{url}/chat/completions"
    if url.endswith("/v1/"):
        return f"{url}chat/completions"
    return url.rstrip("/") + "/v1/chat/completions"


def _resolve_local_config_path(config: str) -> Path:
    raw = Path(config).expanduser()
    if raw.is_absolute():
        return raw.resolve()
    return (PROJECT_ROOT / raw).resolve()


def _resolve_remote_config_path(config: str) -> Path:
    raw = Path(config)
    if raw.is_absolute():
        return raw.resolve()
    return (Path(REMOTE_ROOT) / raw).resolve()


def _config_arg_for_remote(config_path: Path) -> str:
    try:
        return config_path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError as exc:
        raise RuntimeError(
            f"prompt-opt config must live under the repo root: {config_path}"
        ) from exc


def _prompt_crafter_image() -> modal.Image:
    if not CRAFTER_CORE_ROOT.exists():
        raise RuntimeError(f"missing crafter-rs checkout: {CRAFTER_CORE_ROOT}")
    return (
        _cuda_base_image()
        .pip_install(*COMMON_PACKAGES)
        .run_commands(
            "curl https://sh.rustup.rs -sSf | sh -s -- -y --profile minimal --default-toolchain stable",
            "/root/.cargo/bin/cargo --version",
        )
        .add_local_dir(
            (PROJECT_ROOT / "src").as_posix(), remote_path=f"{REMOTE_ROOT}/src", copy=True
        )
        .add_local_dir(
            (PROJECT_ROOT / "scripts").as_posix(), remote_path=f"{REMOTE_ROOT}/scripts", copy=True
        )
        .add_local_dir(
            (PROJECT_ROOT / "configs").as_posix(), remote_path=f"{REMOTE_ROOT}/configs", copy=True
        )
        .add_local_dir(
            (PROJECT_ROOT / "data").as_posix(), remote_path=f"{REMOTE_ROOT}/data", copy=True
        )
        .add_local_dir(
            (PROJECT_ROOT / "containers" / "crafter_rs" / "src").as_posix(),
            remote_path=f"{REMOTE_ROOT}/containers/crafter_rs/src",
            copy=True,
        )
        .add_local_file(
            (PROJECT_ROOT / "containers" / "crafter_rs" / "Cargo.toml").as_posix(),
            remote_path=f"{REMOTE_ROOT}/containers/crafter_rs/Cargo.toml",
            copy=True,
        )
        .add_local_file(
            (PROJECT_ROOT / "containers" / "crafter_rs" / "Cargo.lock").as_posix(),
            remote_path=f"{REMOTE_ROOT}/containers/crafter_rs/Cargo.lock",
            copy=True,
        )
        .add_local_dir(
            CRAFTER_CORE_ROOT.as_posix(), remote_path=str(REMOTE_CRAFTER_CORE), copy=True
        )
        .run_commands(
            f"cd {REMOTE_ROOT}/containers/crafter_rs && "
            "/root/.cargo/bin/cargo build --release --bin crafter-rs-container"
        )
    )


def _prompt_vllm_image() -> modal.Image:
    teacher_venv = f"{OFFLINE_VENV_ROOT}/teacher"
    return (
        modal.Image.from_registry(VLLM_BASE_IMAGE, add_python=VLLM_IMAGE_PYTHON_VERSION)
        .entrypoint([])
        .apt_install("curl")
        .pip_install(*COMMON_PACKAGES)
        .run_commands(
            f"python -m venv {teacher_venv}",
            f"{teacher_venv}/bin/python -m pip install --upgrade pip",
            f"{teacher_venv}/bin/python -m pip install "
            "\"httpx>=0.28.1\" \"pyyaml>=6.0.2\" \"vllm>=0.10.0\"",
        )
        .add_local_dir(
            (PROJECT_ROOT / "src").as_posix(), remote_path=f"{REMOTE_ROOT}/src", copy=True
        )
        .add_local_dir(
            (PROJECT_ROOT / "scripts").as_posix(), remote_path=f"{REMOTE_ROOT}/scripts", copy=True
        )
        .add_local_dir(
            (PROJECT_ROOT / "configs").as_posix(), remote_path=f"{REMOTE_ROOT}/configs", copy=True
        )
        .add_local_dir(
            (PROJECT_ROOT / "data").as_posix(), remote_path=f"{REMOTE_ROOT}/data", copy=True
        )
        .add_local_file(
            (PROJECT_ROOT / "pyproject.toml").as_posix(),
            remote_path=f"{REMOTE_ROOT}/pyproject.toml",
            copy=True,
        )
    )


if REMOTE_SRC.exists():
    prompt_runner_image = modal.Image.debian_slim(python_version="3.11")
    crafter_image = modal.Image.debian_slim(python_version="3.11")
    inference_image = modal.Image.debian_slim(python_version="3.11")
else:
    prompt_runner_image = prompt_image()
    crafter_image = _prompt_crafter_image()
    inference_image = _prompt_vllm_image()
openai_secret = modal.Secret.from_dict(
    {
        key: value
        for key, value in {
            "OPENAI_API_KEY": str(os.getenv("OPENAI_API_KEY") or "").strip(),
            "OPENAI_BASE_URL": str(os.getenv("OPENAI_BASE_URL") or "").strip(),
        }.items()
        if value
    }
)


def _wait_for_health(url: str) -> dict[str, Any]:
    deadline = time.time() + DEFAULT_REQUEST_TIMEOUT_S
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            with httpx.Client(timeout=10.0, follow_redirects=True) as client:
                response = client.get(url)
            response.raise_for_status()
            body = response.json() if response.content else {}
            if isinstance(body, dict):
                return body
            return {"status": "ok", "body": body}
        except Exception as exc:
            last_error = exc
            time.sleep(1.0)
    raise RuntimeError(f"timed out waiting for {url}: {last_error!r}")


def _wait_for_task_info(base_url: str) -> dict[str, Any]:
    deadline = time.time() + DEFAULT_REQUEST_TIMEOUT_S
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            with httpx.Client(timeout=10.0, follow_redirects=True) as client:
                response = client.get(f"{base_url.rstrip('/')}/task_info")
            response.raise_for_status()
            payload = response.json()
            if isinstance(payload, dict):
                return payload
        except Exception as exc:
            last_error = exc
        time.sleep(1.0)
    raise RuntimeError(f"timed out waiting for {base_url}/task_info: {last_error!r}")


def _probe_inference_chat(*, inference_base_url: str, api_key: str, model: str) -> dict[str, Any]:
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Reply with OK."}],
        "max_tokens": 1,
        "temperature": 0.0,
        **build_thinking_budget_request_overrides(enable_thinking=False, thinking_budget=0),
    }
    with httpx.Client(timeout=600.0, follow_redirects=True) as client:
        response = client.post(
            f"{inference_base_url.rstrip('/')}/v1/chat/completions",
            headers=headers,
            json=payload,
        )
    response.raise_for_status()
    body = response.json()
    return {
        "status": "ok",
        "model": body.get("model"),
        "choices": len(body.get("choices") or []),
    }


def _probe_container_roundtrip(
    *,
    container_url: str,
    inference_url: str,
    api_key: str,
    request_model: str,
    seed_prompt: str,
) -> dict[str, Any]:
    rollouts, summary = asyncio.run(
        collect_rollouts_concurrently_with_summary(
            container_url=container_url,
            inference_url=inference_url,
            model=request_model,
            api_key=api_key,
            seeds=[0],
            max_steps=2,
            system_prompt=seed_prompt,
            temperature=0.0,
            max_tokens=512,
            enable_thinking=True,
            thinking_budget_tokens=128,
            policy_version="prompt-opt-preflight",
            target_action_batch_size=2,
            min_action_batch_size=1,
            request_timeout_seconds=600.0,
            max_concurrent_rollouts=1,
            trace_prefix="prompt_opt_preflight",
            rollout_concurrency=1,
            rollout_semaphore_limit=1,
        )
    )
    rollout = rollouts[0] if rollouts else {}
    return {
        "summary": summary,
        "success_status": rollout.get("success_status") if isinstance(rollout, dict) else None,
        "reward": rollout.get("reward_info", {}).get("outcome_reward")
        if isinstance(rollout, dict) and isinstance(rollout.get("reward_info"), dict)
        else None,
    }


@app.cls(
    image=crafter_image,
    timeout=60 * 60 * 4,
    min_containers=1,
    max_containers=1,
    scaledown_window=60 * 10,
    volumes=volume_mounts(),
)
@modal.concurrent(max_inputs=32)
class CrafterService:
    @modal.web_server(port=CRAFTER_PORT, startup_timeout=60 * 10)
    def serve(self) -> None:
        env = {
            **os.environ,
            "PYTHONPATH": _pythonpath_with_repo(),
            "NANOHORIZON_CRAFTER_BIND_HOST": "0.0.0.0",
            "NANOHORIZON_CRAFTER_BIND_PORT": str(CRAFTER_PORT),
        }
        cmd = [str(REMOTE_CRAFTER_BIN)]
        print("Launching Crafter service:", " ".join(shlex.quote(x) for x in cmd), flush=True)
        subprocess.Popen(cmd, env=env)


@app.cls(
    image=inference_image,
    gpu=GPU_PROMPT_OPT,
    timeout=60 * 60 * 4,
    scaledown_window=60 * 10,
    volumes=volume_mounts(),
)
@modal.concurrent(max_inputs=16)
class PromptOptInferenceServer:
    model: str = modal.parameter(default=DEFAULT_MODEL)
    served_model_name: str = modal.parameter(default=DEFAULT_SERVED_MODEL_NAME)
    api_key: str = modal.parameter(default=DEFAULT_API_KEY)
    max_model_len: int = modal.parameter(default=DEFAULT_MAX_MODEL_LEN)

    @modal.web_server(port=VLLM_PORT, startup_timeout=60 * 25)
    def serve(self) -> None:
        vllm_bin = f"{OFFLINE_VENV_ROOT}/teacher/bin/vllm"
        runtime_env = {
            **os.environ,
            "PYTHONUNBUFFERED": "1",
            "PYTHONPATH": _pythonpath_with_repo(),
        }
        cmd = [
            vllm_bin,
            "serve",
            self.model.strip() or DEFAULT_MODEL,
            "--served-model-name",
            self.served_model_name.strip() or DEFAULT_SERVED_MODEL_NAME,
            "--host",
            "0.0.0.0",
            "--port",
            str(VLLM_PORT),
            "--max-model-len",
            str(max(1024, int(self.max_model_len))),
            "--max-num-seqs",
            "32",
            "--gpu-memory-utilization",
            "0.9",
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
            self.api_key.strip() or DEFAULT_API_KEY,
        ]
        cmd, env = enable_thinking_budget_support(
            cmd=cmd,
            env=runtime_env,
            model_ref=self.model.strip() or DEFAULT_MODEL,
        )
        print("Launching prompt-opt vLLM:", " ".join(shlex.quote(x) for x in cmd), flush=True)
        subprocess.Popen(cmd, env=env)


@app.function(
    image=prompt_runner_image,
    timeout=60 * 60 * 3,
    volumes=volume_mounts(),
    secrets=[openai_secret],
)
def run(
    *,
    config: str,
    output_dir: str,
    container_url: str,
    inference_url: str,
    inference_api_key: str,
    request_model: str,
    bootstrap_info: dict[str, Any],
) -> dict[str, Any]:
    from nanohorizon.prompt_opt_training import run_training

    os.chdir(REMOTE_ROOT)
    destination = ensure_dir(output_dir or _default_output_dir())
    write_json(destination / "bootstrap_info.json", bootstrap_info)
    return run_training(
        config_path=_resolve_remote_config_path(config),
        output_dir=destination,
        container_url=container_url,
        inference_url=inference_url,
        inference_api_key=inference_api_key,
        request_model=request_model,
    )


@app.local_entrypoint()
def main(
    config: str = "configs/crafter_prompt_opt_qwen35_4b_gpt54_budget.yaml",
    output_dir: str = "",
) -> None:
    config_path = _resolve_local_config_path(config)
    config_payload = load_config(config_path)
    remote_config_arg = _config_arg_for_remote(config_path)
    resolved_output_dir = output_dir or _default_output_dir()
    model = str(config_payload["policy"]["model"]).strip() or DEFAULT_MODEL
    inference_api_key = (
        str(config_payload["policy"].get("inference_api_key", "")).strip() or DEFAULT_API_KEY
    )
    served_model_name = (
        str(config_payload["policy"].get("served_model_name", "")).strip()
        or DEFAULT_SERVED_MODEL_NAME
    )
    max_model_len = int(config_payload["policy"].get("max_model_len", DEFAULT_MAX_MODEL_LEN))
    crafter = cast(Any, CrafterService)()
    container_url = crafter.serve.get_web_url()
    inference = cast(Any, PromptOptInferenceServer)(
        model=model,
        served_model_name=served_model_name,
        api_key=inference_api_key,
        max_model_len=max_model_len,
    )
    inference_url = inference.serve.get_web_url()
    rollout_inference_url = _normalize_inference_url(inference_url)
    print(
        json.dumps(
            {
                "stage": "prompt_opt_bootstrap_urls",
                "container_url": container_url,
                "inference_url": inference_url,
                "rollout_inference_url": rollout_inference_url,
                "request_model": served_model_name,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    seed_prompt = str(config_payload["prompt"]["seed_prompt"]).strip()
    bootstrap_info = {
        "container_url": container_url,
        "inference_url": inference_url,
        "rollout_inference_url": rollout_inference_url,
        "request_model": served_model_name,
        "preflights": {
            "crafter_health": (print('{"stage":"preflight_crafter_health"}', flush=True) or _wait_for_health(f"{container_url.rstrip('/')}/health")),
            "crafter_task_info": (print('{"stage":"preflight_crafter_task_info"}', flush=True) or _wait_for_task_info(container_url)),
            "inference_health": (print('{"stage":"preflight_inference_health"}', flush=True) or _wait_for_health(f"{inference_url.rstrip('/')}/health")),
            "inference_chat": (print('{"stage":"preflight_inference_chat"}', flush=True) or _probe_inference_chat(
                inference_base_url=inference_url,
                api_key=inference_api_key,
                model=served_model_name,
            )),
            "crafter_roundtrip": (print('{"stage":"preflight_crafter_roundtrip"}', flush=True) or _probe_container_roundtrip(
                container_url=container_url,
                inference_url=rollout_inference_url,
                api_key=inference_api_key,
                request_model=served_model_name,
                seed_prompt=seed_prompt,
            )),
        },
    }
    result = run.remote(
        config=remote_config_arg,
        output_dir=resolved_output_dir,
        container_url=container_url,
        inference_url=inference_url,
        inference_api_key=inference_api_key,
        request_model=served_model_name,
        bootstrap_info=bootstrap_info,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
