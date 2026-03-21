# ruff: noqa: E402

import asyncio
import json
import os
import shlex
import shutil
import socket
import subprocess
import sys
import time
import uuid
from pathlib import Path, PurePosixPath
from typing import Any

import httpx
import modal
from modal import experimental as modal_exp

REMOTE_SRC = Path("/root/nanohorizon/src")
if REMOTE_SRC.exists():
    sys.path.insert(0, str(REMOTE_SRC))

from nanohorizon.common import ensure_dir, load_config, now_utc_iso, write_json
from nanohorizon.crafter_data import collect_rollouts_concurrently_with_summary
from nanohorizon.custom_vllm.runtime import (
    build_thinking_budget_request_overrides,
    enable_thinking_budget_support,
)
from nanohorizon.lora_bundle import extract_lora_bundle
from nanohorizon.modal_common import (
    GPU_RLVR,
    OFFLINE_VENV_ROOT,
    PROJECT_ROOT,
    RECORDS_DIR,
    RECORDS_VOLUME,
    REMOTE_ROOT,
    TRAIN_PACKAGES,
    TRITON_CACHE_DIR,
    VLLM_COMPILE_CACHE_DIR,
    _cuda_base_image,
    volume_mounts,
)
from nanohorizon.rlvr_training import DEFAULT_SYSTEM_PROMPT, run_training

APP_NAME = "nanohorizon-crafter-rlvr"
CRAFTER_PORT = 8903
VLLM_PORT = 8000
CLUSTER_SIZE = 2
DEFAULT_REQUEST_TIMEOUT_S = 60 * 20
DEFAULT_CLUSTER_SIGNAL_TIMEOUT_S = 60 * 25
DEFAULT_INFERENCE_MODEL = "Qwen/Qwen3.5-4B"
DEFAULT_SERVED_MODEL_NAME = "qwen35-4b-rlvr"
DEFAULT_MAX_MODEL_LEN = 8192
DEFAULT_MAX_LORA_RANK = 16
RUNTIME_LORA_DIR = Path("/tmp/nanohorizon-rlvr-loras")
RUNTIME_VLLM_BIN = Path(f"{OFFLINE_VENV_ROOT}/teacher/bin/vllm")
CRAFTER_CORE_ROOT = Path(
    os.getenv("NANOHORIZON_CRAFTER_CORE_ROOT") or str(PROJECT_ROOT.parent / "crafter-rs")
).expanduser()
REMOTE_CRAFTER_CORE = PurePosixPath("/root/crafter-rs")
REMOTE_CRAFTER_BIN = PurePosixPath(
    f"{REMOTE_ROOT}/containers/crafter_rs/target/release/crafter-rs-container"
)
DEFAULT_INFERENCE_API_KEY = (
    os.getenv("NANOHORIZON_RLVR_INFERENCE_API_KEY", "nanohorizon-rlvr-key").strip()
    or "nanohorizon-rlvr-key"
)

app = modal.App(APP_NAME)


def _default_output_dir() -> str:
    stamp = now_utc_iso().replace(":", "").replace("+00:00", "Z")
    return f"{RECORDS_DIR}/rlvr_20min_2xa100_40gb/{stamp}_reference_baseline"


def _default_local_preflight_failure_dir() -> Path:
    stamp = now_utc_iso().replace(":", "").replace("+00:00", "Z")
    return PROJECT_ROOT / "artifacts" / "rlvr_preflight_failures" / stamp


def _rlvr_runtime_image() -> modal.Image:
    if not CRAFTER_CORE_ROOT.exists():
        raise RuntimeError(f"missing crafter-rs checkout: {CRAFTER_CORE_ROOT}")
    teacher_venv = f"{OFFLINE_VENV_ROOT}/teacher"
    return (
        _cuda_base_image()
        .pip_install(*TRAIN_PACKAGES, "fastapi>=0.115.0", "uvicorn>=0.32.0")
        .run_commands(
            f"python -m venv {teacher_venv}",
            f"{teacher_venv}/bin/python -m pip install --upgrade pip",
            f"{teacher_venv}/bin/python -m pip install "
            "\"httpx>=0.28.1\" \"pyyaml>=6.0.2\" \"vllm>=0.10.0\"",
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
        .add_local_file(
            (PROJECT_ROOT / "containers" / "crafter_rs" / "README.md").as_posix(),
            remote_path=f"{REMOTE_ROOT}/containers/crafter_rs/README.md",
            copy=True,
        )
        .add_local_file(
            (PROJECT_ROOT / "pyproject.toml").as_posix(),
            remote_path=f"{REMOTE_ROOT}/pyproject.toml",
            copy=True,
        )
        .add_local_file(
            (PROJECT_ROOT / "README.md").as_posix(),
            remote_path=f"{REMOTE_ROOT}/README.md",
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


if REMOTE_SRC.exists():
    runtime_image = modal.Image.debian_slim(python_version="3.11")
else:
    runtime_image = _rlvr_runtime_image()


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


def _volume_commit() -> None:
    try:
        RECORDS_VOLUME.commit()
    except Exception as exc:
        print(f"RLVR records volume commit failed: {type(exc).__name__}: {exc}", flush=True)


def _volume_reload() -> None:
    try:
        RECORDS_VOLUME.reload()
    except Exception as exc:
        print(f"RLVR records volume reload failed: {type(exc).__name__}: {exc}", flush=True)


def _write_cluster_signal(path: Path, payload: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    write_json(path, payload)
    _volume_commit()


def _cluster_control_paths(output_dir: str | Path) -> dict[str, Path]:
    control_dir = ensure_dir(Path(output_dir) / "_cluster_control")
    requests_dir = ensure_dir(control_dir / "publish_requests")
    responses_dir = ensure_dir(control_dir / "publish_responses")
    bundles_dir = ensure_dir(control_dir / "publish_bundles")
    return {
        "dir": control_dir,
        "ready": control_dir / "cluster_ready.json",
        "error": control_dir / "cluster_error.json",
        "done": control_dir / "cluster_done.json",
        "requests": requests_dir,
        "responses": responses_dir,
        "bundles": bundles_dir,
    }


def _wait_for_cluster_signal(
    *,
    ready_path: Path,
    error_path: Path,
    timeout_seconds: float = DEFAULT_CLUSTER_SIGNAL_TIMEOUT_S,
) -> dict[str, Any]:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        _volume_reload()
        if error_path.exists():
            return json.loads(error_path.read_text(encoding="utf-8"))
        if ready_path.exists():
            return json.loads(ready_path.read_text(encoding="utf-8"))
        time.sleep(1.0)
    raise RuntimeError(f"timed out waiting for clustered signal: {ready_path}")


def _wait_for_health(url: str, *, require_upstream_ready: bool = False) -> dict[str, Any]:
    deadline = time.time() + DEFAULT_REQUEST_TIMEOUT_S
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            with httpx.Client(timeout=10.0, follow_redirects=True) as client:
                response = client.get(url)
            if response.status_code == 200:
                payload: dict[str, Any]
                if response.content:
                    parsed = response.json()
                    if isinstance(parsed, dict):
                        payload = parsed
                    else:
                        payload = {"status": "ok", "body": parsed}
                else:
                    payload = {"status": "ok"}
                if not require_upstream_ready:
                    return payload
                if bool(payload.get("upstream_ready", False)):
                    return payload
                last_error = RuntimeError(f"upstream not ready: {payload}")
            else:
                last_error = RuntimeError(f"health returned HTTP {response.status_code}")
        except Exception as exc:
            last_error = exc
        time.sleep(1.0)
    raise RuntimeError(f"timed out waiting for {url}: {last_error!r}")


def _wait_for_task_info(base_url: str) -> dict[str, Any]:
    deadline = time.time() + DEFAULT_REQUEST_TIMEOUT_S
    last_error: Exception | None = None
    url = f"{base_url.rstrip('/')}/task_info"
    while time.time() < deadline:
        try:
            with httpx.Client(timeout=10.0, follow_redirects=True) as client:
                response = client.get(url)
            response.raise_for_status()
            payload = response.json()
            if isinstance(payload, dict):
                return payload
            last_error = RuntimeError(f"/task_info returned non-object payload: {type(payload).__name__}")
        except Exception as exc:
            last_error = exc
        time.sleep(1.0)
    raise RuntimeError(f"timed out waiting for {url}: {last_error!r}")


def _probe_inference_chat(*, inference_base_url: str, api_key: str, model: str) -> dict[str, Any]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Reply with OK."}],
        "max_tokens": 1,
        "temperature": 0.0,
        **build_thinking_budget_request_overrides(enable_thinking=False, thinking_budget=0),
    }
    last_error: Exception | None = None
    for attempt in range(1, 4):
        try:
            with httpx.Client(timeout=600.0, follow_redirects=True) as client:
                response = client.post(
                    f"{inference_base_url.rstrip('/')}/v1/chat/completions",
                    headers=headers,
                    json=payload,
                )
                response.raise_for_status()
                body = response.json()
            break
        except Exception as exc:
            last_error = exc
            if attempt >= 3:
                raise
            time.sleep(float(attempt))
    else:
        raise RuntimeError(f"inference chat probe failed: {last_error!r}")
    if not isinstance(body, dict):
        raise RuntimeError("inference chat probe returned non-object payload")
    return {
        "status": "ok",
        "id": body.get("id"),
        "model": body.get("model"),
        "choices": len(body.get("choices") or []),
    }


def _probe_container_roundtrip(
    *,
    container_url: str,
    inference_url: str,
    api_key: str,
    request_model: str,
) -> dict[str, Any]:
    rollouts, summary = asyncio.run(
        collect_rollouts_concurrently_with_summary(
            container_url=container_url,
            inference_url=inference_url,
            model=request_model,
            api_key=api_key,
            seeds=[0],
            max_steps=1,
            system_prompt=DEFAULT_SYSTEM_PROMPT,
            temperature=0.0,
            max_tokens=256,
            enable_thinking=False,
            thinking_budget_tokens=0,
            policy_version="preflight",
            target_action_batch_size=1,
            min_action_batch_size=1,
            request_timeout_seconds=600.0,
            max_concurrent_rollouts=1,
            trace_prefix="rlvr_preflight_roundtrip",
            rollout_concurrency=1,
            rollout_semaphore_limit=1,
        )
    )
    if not rollouts:
        raise RuntimeError(f"container roundtrip preflight returned no rollouts: {summary}")
    rollout = rollouts[0]
    return {
        "status": str(rollout.get("success_status") or "unknown"),
        "reward": rollout.get("reward_info", {}).get("outcome_reward")
        if isinstance(rollout.get("reward_info"), dict)
        else None,
        "trace_correlation_id": rollout.get("trace_correlation_id"),
        "llm_call_count": rollout.get("metadata", {}).get("llm_call_count")
        if isinstance(rollout.get("metadata"), dict)
        else None,
        "summary": summary,
    }


def _write_local_preflight_failure(*, output_dir: str, payload: dict[str, Any]) -> Path:
    requested = str(output_dir or "").strip()
    if requested and not requested.startswith("/vol/"):
        destination = ensure_dir(requested)
    else:
        destination = ensure_dir(_default_local_preflight_failure_dir())
    target = destination / "preflight_failure.json"
    write_json(target, payload)
    return target


def _remote_runtime_config_path(*, config: str, config_text: str, config_filename: str) -> str:
    runtime_config_path = f"{REMOTE_ROOT}/{config}"
    submitted_config_text = str(config_text or "")
    if not submitted_config_text:
        return runtime_config_path
    submitted_dir = ensure_dir(f"{REMOTE_ROOT}/configs")
    original_name = Path(config_filename or config).name or "submitted_rlvr_config.yaml"
    submitted_name = f"__submitted_{original_name}"
    runtime_config = Path(submitted_dir) / submitted_name
    runtime_config.write_text(submitted_config_text, encoding="utf-8")
    return str(runtime_config)


def _clustered_ipv4_addresses(cluster_info: modal_exp.ClusterInfo) -> list[str]:
    addresses = [
        str(ip or "").strip()
        for ip in getattr(cluster_info, "container_ipv4_ips", [])
        if str(ip or "").strip()
    ]
    if len(addresses) < CLUSTER_SIZE:
        raise RuntimeError(
            "clustered RLVR requires IPv4 addresses for all ranks; "
            f"got {getattr(cluster_info, 'container_ipv4_ips', [])!r}"
        )
    return addresses


@app.cls(
    image=runtime_image,
    timeout=60 * 60,
    min_containers=1,
    max_containers=1,
    scaledown_window=60 * 10,
    volumes=volume_mounts(),
)
class CrafterService:
    @modal.web_server(port=CRAFTER_PORT, startup_timeout=60 * 10)
    def serve(self) -> None:
        runtime_env = {
            **os.environ,
            "PYTHONUNBUFFERED": "1",
            "PYTHONPATH": _pythonpath_with_repo(),
            "NANOHORIZON_CRAFTER_BIND_HOST": "0.0.0.0",
            "NANOHORIZON_CRAFTER_BIND_PORT": str(CRAFTER_PORT),
        }
        cmd = [str(REMOTE_CRAFTER_BIN)]
        print("Launching Crafter service:", " ".join(shlex.quote(part) for part in cmd), flush=True)
        subprocess.Popen(cmd, env=runtime_env)


def _start_local_vllm(
    *,
    model: str,
    served_model_name: str,
    api_key: str,
    max_model_len: int,
    max_lora_rank: int,
) -> subprocess.Popen[bytes]:
    vllm_bin = str(RUNTIME_VLLM_BIN)
    runtime_env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    runtime_env["PYTHONPATH"] = _pythonpath_with_repo()
    runtime_env["VLLM_ALLOW_RUNTIME_LORA_UPDATING"] = "1"
    runtime_env["HF_HOME"] = str(os.environ.get("HF_HOME") or "/root/.cache/huggingface")
    runtime_env["HF_HUB_ENABLE_HF_TRANSFER"] = str(
        os.environ.get("HF_HUB_ENABLE_HF_TRANSFER") or "1"
    )
    runtime_env["TORCHINDUCTOR_CACHE_DIR"] = str(
        os.environ.get("TORCHINDUCTOR_CACHE_DIR") or VLLM_COMPILE_CACHE_DIR
    )
    runtime_env["TRITON_CACHE_DIR"] = str(os.environ.get("TRITON_CACHE_DIR") or TRITON_CACHE_DIR)
    runtime_env["VLLM_SERVER_DEV_MODE"] = "1"
    runtime_env.pop("VLLM_USE_V1", None)
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
        str(max(1024, int(max_model_len))),
        "--max-num-seqs",
        "64",
        "--gpu-memory-utilization",
        "0.92",
        "--uvicorn-log-level",
        "info",
        "--enable-prefix-caching",
        "--language-model-only",
        "--reasoning-parser",
        "qwen3",
        "--enable-auto-tool-choice",
        "--tool-call-parser",
        "qwen3_coder",
        "--enable-lora",
        "--max-lora-rank",
        str(max(1, int(max_lora_rank))),
        "--api-key",
        api_key,
        "--enforce-eager",
    ]
    cmd, runtime_env = enable_thinking_budget_support(cmd=cmd, env=runtime_env, model_ref=model)
    print("Launching clustered RLVR vLLM:", " ".join(shlex.quote(part) for part in cmd), flush=True)
    return subprocess.Popen(cmd, env=runtime_env)


def _local_listener_diagnostics(process: subprocess.Popen[bytes] | None) -> dict[str, Any]:
    socket_probe: dict[str, Any] = {"host": "127.0.0.1", "port": VLLM_PORT, "connectable": False}
    try:
        with socket.create_connection(("127.0.0.1", VLLM_PORT), timeout=1.0):
            socket_probe["connectable"] = True
    except Exception as exc:
        socket_probe["error"] = f"{type(exc).__name__}: {exc}"
    listener_dump = ""
    try:
        completed = subprocess.run(
            [
                "bash",
                "-lc",
                "if command -v ss >/dev/null 2>&1; then ss -ltnp; "
                "elif command -v netstat >/dev/null 2>&1; then netstat -ltnp; "
                "else echo 'no_listener_tool'; fi",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
        listener_dump = (completed.stdout or completed.stderr or "").strip()[:12000]
    except Exception as exc:
        listener_dump = f"listener inspection failed: {type(exc).__name__}: {exc}"
    process_dump = ""
    try:
        completed = subprocess.run(
            ["ps", "-ef"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
        process_dump = (completed.stdout or completed.stderr or "").strip()[:12000]
    except Exception as exc:
        process_dump = f"process inspection failed: {type(exc).__name__}: {exc}"
    return {
        "vllm_pid": process.pid if process is not None else None,
        "vllm_running": bool(process is not None and process.poll() is None),
        "vllm_returncode": process.returncode if process is not None else None,
        "socket_probe": socket_probe,
        "listener_dump": listener_dump,
        "process_dump": process_dump,
    }


def _wait_for_local_vllm(
    process: subprocess.Popen[bytes],
    *,
    timeout_seconds: float = 60 * 20,
    probe_hosts: list[str] | None = None,
) -> None:
    deadline = time.time() + timeout_seconds
    last_error: Exception | None = None
    attempt = 0
    hosts = [host for host in (probe_hosts or ["127.0.0.1"]) if str(host).strip()]
    while time.time() < deadline:
        attempt += 1
        if process.poll() is not None:
            raise RuntimeError(f"clustered RLVR vLLM exited early with code {process.returncode}")
        for host in hosts:
            try:
                with httpx.Client(timeout=5.0) as client:
                    response = client.get(f"http://{host}:{VLLM_PORT}/health")
                if response.status_code == 200:
                    print(
                        f"clustered RLVR vLLM healthy after {attempt} checks via host={host}",
                        flush=True,
                    )
                    return
                last_error = RuntimeError(f"/health returned HTTP {response.status_code} via host={host}")
            except Exception as exc:
                last_error = exc
        if attempt == 1 or attempt % 15 == 0:
            print(
                f"waiting for clustered RLVR vLLM health: attempt={attempt} "
                f"last_error={last_error!r}",
                flush=True,
            )
        time.sleep(1.0)
    diagnostics = _local_listener_diagnostics(process)
    raise RuntimeError(
        "timed out waiting for clustered RLVR vLLM health: "
        f"{last_error!r}; diagnostics={json.dumps(diagnostics, sort_keys=True)}"
    )


def _load_lora_bundle_into_local_vllm(
    *,
    api_key: str,
    lora_name: str,
    policy_version: str,
    bundle_bytes: bytes,
) -> dict[str, Any]:
    runtime_dir = extract_lora_bundle(
        bundle_bytes=bundle_bytes,
        dest_root=RUNTIME_LORA_DIR,
        bundle_name=lora_name,
    )
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    with httpx.Client(timeout=120.0) as client:
        response = client.post(
            f"http://127.0.0.1:{VLLM_PORT}/v1/load_lora_adapter",
            json={
                "lora_name": lora_name,
                "lora_path": str(runtime_dir),
                "policy_version": policy_version,
                "load_inplace": True,
            },
            headers=headers,
        )
        response.raise_for_status()
        payload: dict[str, Any]
        if response.content:
            try:
                data = response.json()
            except ValueError:
                data = {"raw_body": response.text}
            payload = dict(data) if isinstance(data, dict) else {"raw_body": data}
        else:
            payload = {"status": "ok", "empty_body": True}
    payload["runtime_lora_dir"] = str(runtime_dir)
    return payload


def _process_publish_requests(
    *,
    control_paths: dict[str, Path],
    api_key: str,
) -> None:
    for request_path in sorted(control_paths["requests"].glob("*.json")):
        response_path = control_paths["responses"] / request_path.name
        if response_path.exists():
            continue
        try:
            request = json.loads(request_path.read_text(encoding="utf-8"))
            bundle_path = Path(str(request["bundle_path"]))
            bundle_bytes = bundle_path.read_bytes()
            payload = _load_lora_bundle_into_local_vllm(
                api_key=api_key,
                lora_name=str(request["lora_name"]),
                policy_version=str(request["policy_version"]),
                bundle_bytes=bundle_bytes,
            )
            response_payload = {
                "status": "ok",
                "request_id": str(request.get("request_id") or request_path.stem),
                "payload": payload,
            }
        except Exception as exc:
            response_payload = {
                "status": "error",
                "request_id": request_path.stem,
                "error": f"{type(exc).__name__}: {exc}",
            }
        write_json(response_path, response_payload)
        _volume_commit()


class _ClusteredRemoteMethod:
    def __init__(self, callback: Any) -> None:
        self._callback = callback

    def remote(self, **kwargs: Any) -> Any:
        return self._callback(**kwargs)


class _ClusteredInferenceHandle:
    def __init__(self, control_paths: dict[str, Path]) -> None:
        self._control_paths = control_paths
        self.load_lora_bundle = _ClusteredRemoteMethod(self._load_lora_bundle)

    def _load_lora_bundle(
        self,
        *,
        lora_name: str,
        policy_version: str,
        bundle_bytes: bytes,
    ) -> dict[str, Any]:
        request_id = uuid.uuid4().hex
        bundle_path = self._control_paths["bundles"] / f"{request_id}.tar"
        request_path = self._control_paths["requests"] / f"{request_id}.json"
        response_path = self._control_paths["responses"] / f"{request_id}.json"
        bundle_path.write_bytes(bundle_bytes)
        write_json(
            request_path,
            {
                "request_id": request_id,
                "lora_name": lora_name,
                "policy_version": policy_version,
                "bundle_path": str(bundle_path),
            },
        )
        _volume_commit()
        deadline = time.time() + 60 * 10
        while time.time() < deadline:
            _volume_reload()
            if self._control_paths["error"].exists():
                payload = json.loads(self._control_paths["error"].read_text(encoding="utf-8"))
                raise RuntimeError(f"clustered inference worker failed: {payload}")
            if response_path.exists():
                payload = json.loads(response_path.read_text(encoding="utf-8"))
                if str(payload.get("status")) != "ok":
                    raise RuntimeError(str(payload.get("error") or "unknown clustered publish error"))
                response = payload.get("payload")
                if isinstance(response, dict):
                    return dict(response)
                return {"status": "ok"}
            time.sleep(1.0)
        raise RuntimeError(f"timed out waiting for clustered LoRA publish response: {response_path}")


def _clustered_inference_worker(payload: dict[str, Any], cluster_info: modal_exp.ClusterInfo) -> None:
    control_paths = _cluster_control_paths(str(payload["output_dir"]))
    process: subprocess.Popen[bytes] | None = None
    forward_cm: Any | None = None
    try:
        model = str(payload["inference_model"])
        served_model_name = str(payload["served_model_name"])
        api_key = str(payload["inference_api_key"])
        max_model_len = int(payload["max_model_len"])
        max_lora_rank = int(payload["max_lora_rank"])
        process = _start_local_vllm(
            model=model,
            served_model_name=served_model_name,
            api_key=api_key,
            max_model_len=max_model_len,
            max_lora_rank=max_lora_rank,
        )
        ipv4_addrs = _clustered_ipv4_addresses(cluster_info)
        local_ipv4 = str(ipv4_addrs[int(cluster_info.rank)])
        _wait_for_local_vllm(process, probe_hosts=["127.0.0.1", local_ipv4])
        internal_inference_base_url = f"http://{local_ipv4}:{VLLM_PORT}"
        forward_cm = modal.forward(VLLM_PORT)
        forwarded_tunnel = forward_cm.__enter__()
        rollout_inference_base_url = str(forwarded_tunnel.url).rstrip("/")
        _write_cluster_signal(
            control_paths["ready"],
            {
                "status": "ready",
                "cluster_rank": int(cluster_info.rank),
                "container_ipv4": local_ipv4,
                "inference_base_url": rollout_inference_base_url,
                "inference_url": f"{rollout_inference_base_url}/v1/chat/completions",
                "inference_admin_url": f"{rollout_inference_base_url}/v1",
                "internal_inference_base_url": internal_inference_base_url,
                "internal_inference_url": f"{internal_inference_base_url}/v1/chat/completions",
                "served_model_name": served_model_name,
                "pid": int(process.pid),
            },
        )
        print(
            "clustered RLVR inference worker ready",
            json.dumps(
                {
                    "cluster_rank": int(cluster_info.rank),
                    "internal_inference_base_url": internal_inference_base_url,
                    "rollout_inference_base_url": rollout_inference_base_url,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        while True:
            _volume_reload()
            if control_paths["done"].exists():
                return
            _process_publish_requests(control_paths=control_paths, api_key=api_key)
            time.sleep(1.0)
    except Exception as exc:
        _write_cluster_signal(
            control_paths["error"],
            {
                "status": "error",
                "cluster_rank": int(cluster_info.rank),
                "error": f"{type(exc).__name__}: {exc}",
                "diagnostics": _local_listener_diagnostics(process),
            },
        )
        print(f"clustered RLVR inference worker failed: {type(exc).__name__}: {exc}", flush=True)
    finally:
        if forward_cm is not None:
            try:
                forward_cm.__exit__(None, None, None)
            except Exception as exc:
                print(f"clustered RLVR forward cleanup failed: {type(exc).__name__}: {exc}", flush=True)
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)


def _clustered_controller(payload: dict[str, Any], cluster_info: modal_exp.ClusterInfo) -> dict[str, Any]:
    os.chdir(REMOTE_ROOT)
    output_dir = ensure_dir(str(payload["output_dir"]))
    control_paths = _cluster_control_paths(output_dir)
    shutil.rmtree(control_paths["dir"], ignore_errors=True)
    control_paths = _cluster_control_paths(output_dir)
    runtime_config_path = _remote_runtime_config_path(
        config=str(payload["config"]),
        config_text=str(payload.get("config_text") or ""),
        config_filename=str(payload.get("config_filename") or ""),
    )
    ready_payload = _wait_for_cluster_signal(
        ready_path=control_paths["ready"],
        error_path=control_paths["error"],
    )
    if str(ready_payload.get("status")) != "ready":
        failure_payload = {
            "container_url": str(payload["container_url"]),
            "output_dir": str(output_dir),
            "error": ready_payload,
        }
        failure_path = output_dir / "preflight_failure.json"
        write_json(failure_path, failure_payload)
        _volume_commit()
        raise RuntimeError(f"clustered inference worker failed during startup: {ready_payload}")

    container_url = str(payload["container_url"]).rstrip("/")
    inference_base_url = str(ready_payload["inference_base_url"]).rstrip("/")
    served_model_name = str(payload["served_model_name"])
    inference_api_key = str(payload["inference_api_key"])
    bootstrap_info: dict[str, Any] = {
        "container_url": container_url,
        "inference_base_url": inference_base_url,
        "inference_url": f"{inference_base_url}/v1/chat/completions",
        "inference_admin_url": f"{inference_base_url}/v1",
        "served_model_name": served_model_name,
        "cluster": {
            "rank": int(cluster_info.rank),
            "container_ipv4_ips": [str(item) for item in getattr(cluster_info, "container_ipv4_ips", ())],
            "container_ips": [str(item) for item in getattr(cluster_info, "container_ips", ())],
            "ready_payload": ready_payload,
        },
        "preflight": {},
    }
    try:
        bootstrap_info["preflight"]["container_health"] = _wait_for_health(f"{container_url}/health")
        bootstrap_info["preflight"]["container_task_info"] = _wait_for_task_info(container_url)
        bootstrap_info["preflight"]["inference_public_health"] = _wait_for_health(
            f"{inference_base_url}/health"
        )
        bootstrap_info["preflight"]["inference_chat_probe"] = _probe_inference_chat(
            inference_base_url=inference_base_url,
            api_key=inference_api_key,
            model=served_model_name,
        )
        bootstrap_info["preflight"]["container_roundtrip_probe"] = _probe_container_roundtrip(
            container_url=container_url,
            inference_url=f"{inference_base_url}/v1/chat/completions",
            api_key=inference_api_key,
            request_model=served_model_name,
        )
    except Exception as exc:
        failure_payload = {**bootstrap_info, "error": f"{type(exc).__name__}: {exc}"}
        failure_path = output_dir / "preflight_failure.json"
        write_json(failure_path, failure_payload)
        _volume_commit()
        raise RuntimeError(f"clustered RLVR preflight failed: {exc}") from exc

    external_inference_server = _ClusteredInferenceHandle(control_paths)
    try:
        return run_training(
            config_path=runtime_config_path,
            output_dir=str(output_dir),
            container_url=container_url,
            inference_url=f"{inference_base_url}/v1/chat/completions",
            inference_admin_url=f"{inference_base_url}/v1",
            inference_api_key=inference_api_key,
            request_model=served_model_name,
            external_inference_server=external_inference_server,
            bootstrap_info=bootstrap_info,
        )
    finally:
        _write_cluster_signal(
            control_paths["done"],
            {"status": "done", "finished_at": now_utc_iso()},
        )


@app.function(
    image=runtime_image,
    gpu=GPU_RLVR,
    timeout=60 * 60,
    volumes=volume_mounts(),
)
@modal_exp.clustered(size=CLUSTER_SIZE)
def run_clustered_rlvr(payload: dict[str, Any]) -> dict[str, Any] | None:
    cluster_info = modal_exp.get_cluster_info()
    if int(cluster_info.rank) == 1:
        _clustered_inference_worker(payload, cluster_info)
        return None
    return _clustered_controller(payload, cluster_info)


@app.local_entrypoint()
def main(
    config: str = "configs/crafter_rlvr_qwen35_4b_2xa100_20min.yaml",
    output_dir: str = "",
) -> None:
    config_path = PROJECT_ROOT / config
    submitted_config_path = config_path.expanduser().resolve()
    config_payload = load_config(config_path)
    model_name = str(config_payload.get("model", {}).get("model") or DEFAULT_INFERENCE_MODEL)
    served_model_name = str(
        config_payload.get("model", {}).get("served_model_name") or DEFAULT_SERVED_MODEL_NAME
    ).strip() or DEFAULT_SERVED_MODEL_NAME
    max_model_len = int(
        config_payload.get("inference", {}).get("max_model_len")
        or config_payload.get("training", {}).get("max_length")
        or DEFAULT_MAX_MODEL_LEN
    )
    max_lora_rank = int(config_payload.get("training", {}).get("lora_rank", DEFAULT_MAX_LORA_RANK))
    if model_name != DEFAULT_INFERENCE_MODEL:
        raise RuntimeError(
            f"RLVR Modal inference currently supports model={DEFAULT_INFERENCE_MODEL!r}; got {model_name!r}"
        )
    if served_model_name != DEFAULT_SERVED_MODEL_NAME:
        raise RuntimeError(
            "RLVR Modal inference currently supports "
            f"served_model_name={DEFAULT_SERVED_MODEL_NAME!r}; got {served_model_name!r}"
        )
    if max_model_len != DEFAULT_MAX_MODEL_LEN:
        raise RuntimeError(
            "RLVR Modal inference currently supports "
            f"max_model_len={DEFAULT_MAX_MODEL_LEN}; got {max_model_len}"
        )
    if max_lora_rank != DEFAULT_MAX_LORA_RANK:
        raise RuntimeError(
            "RLVR Modal inference currently supports "
            f"training.lora_rank={DEFAULT_MAX_LORA_RANK}; got {max_lora_rank}"
        )
    inference_api_key = str(
        os.getenv("NANOHORIZON_RLVR_INFERENCE_API_KEY")
        or config_payload.get("inference", {}).get("api_key")
        or DEFAULT_INFERENCE_API_KEY
    ).strip() or DEFAULT_INFERENCE_API_KEY
    resolved_output_dir = str(output_dir or _default_output_dir())
    crafter_service = CrafterService()
    crafter_web_url = crafter_service.serve.get_web_url()
    if not crafter_web_url:
        raise RuntimeError("Crafter service did not provide a web URL")
    container_url = crafter_web_url.rstrip("/")
    payload = {
        "config": config,
        "config_text": submitted_config_path.read_text(encoding="utf-8"),
        "config_filename": submitted_config_path.name,
        "output_dir": resolved_output_dir,
        "container_url": container_url,
        "inference_api_key": inference_api_key,
        "inference_model": model_name,
        "served_model_name": served_model_name,
        "max_model_len": max_model_len,
        "max_lora_rank": max_lora_rank,
    }
    try:
        result = run_clustered_rlvr.remote(payload)
    except Exception as exc:
        failure_payload = {
            "container_url": container_url,
            "config": config,
            "output_dir": resolved_output_dir,
            "error": f"{type(exc).__name__}: {exc}",
        }
        failure_path = _write_local_preflight_failure(
            output_dir=resolved_output_dir,
            payload=failure_payload,
        )
        raise RuntimeError(
            f"clustered RLVR run failed; details written to {failure_path}: {exc}"
        ) from exc
    print(json.dumps(result, indent=2, sort_keys=True))
