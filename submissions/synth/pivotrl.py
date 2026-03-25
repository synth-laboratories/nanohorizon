from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import platform
import random
import re
import shlex
import signal
import subprocess
import sys
import time
import urllib.request
from collections.abc import Iterator
from contextlib import contextmanager, nullcontext, suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean
from typing import Any, TypedDict

import httpx
import modal


REMOTE_ROOT = "/root/nanohorizon"
_THIS_FILE = Path(__file__).resolve()


def _resolve_repo_root() -> Path:
    if len(_THIS_FILE.parents) >= 3 and _THIS_FILE.parent.name == "synth":
        return _THIS_FILE.parents[2]
    remote_root = Path(os.getenv("NANOHORIZON_PIVOTRL_REMOTE_ROOT", REMOTE_ROOT))
    if remote_root.exists():
        return remote_root
    return Path.cwd().resolve()


REPO_ROOT = _resolve_repo_root()
PROJECT_ROOT = REPO_ROOT
HF_CACHE_DIR = "/root/.cache/huggingface"
VLLM_CACHE_DIR = "/root/.cache/vllm"
TRITON_CACHE_DIR = "/root/.triton"
ARTIFACT_DIR = "/vol/artifacts"
CRAFTAX_PORT = 8903
DEFAULT_REQUEST_TIMEOUT_S = 60 * 20
DEFAULT_VLLM_PORT = 8003
DEFAULT_VLLM_GPU_UTILIZATION = 0.9
DEFAULT_MODAL_GPU = os.getenv("NANOHORIZON_MODAL_GPU_PIVOTRL", "A100-40GB")
APP_NAME = os.getenv("NANOHORIZON_MODAL_PIVOTRL_APP_NAME", "nanohorizon-pivotrl")

HF_CACHE_VOLUME = modal.Volume.from_name("nanohorizon-hf-cache", create_if_missing=True)
VLLM_CACHE_VOLUME = modal.Volume.from_name("nanohorizon-vllm-cache", create_if_missing=True)
TRITON_CACHE_VOLUME = modal.Volume.from_name("nanohorizon-triton-cache", create_if_missing=True)
ARTIFACT_VOLUME = modal.Volume.from_name("nanohorizon-artifacts", create_if_missing=True)

CRAFTAX_ACTION_ENUM = [
    "move_left",
    "move_right",
    "move_up",
    "move_down",
    "do",
    "sleep",
    "place_table",
    "place_stone",
    "place_furnace",
    "place_plant",
    "make_wood_pickaxe",
    "make_stone_pickaxe",
    "make_iron_pickaxe",
    "make_wood_sword",
    "make_stone_sword",
    "make_iron_sword",
]

CRAFTAX_INTERACT_TOOL = {
    "type": "function",
    "function": {
        "name": "craftax_interact",
        "description": "Choose the next short Craftax macro-action sequence.",
        "parameters": {
            "type": "object",
            "properties": {
                "actions_list": {
                    "type": "array",
                    "items": {"type": "string", "enum": CRAFTAX_ACTION_ENUM},
                    "minItems": 1,
                    "maxItems": 10,
                }
            },
            "required": ["actions_list"],
            "additionalProperties": False,
        },
    },
}


ACTION_SET = set(CRAFTAX_ACTION_ENUM)
DEFAULT_BASE_MODEL = "Qwen/Qwen3.5-4B"
DEFAULT_TEACHER_MODEL = "Qwen/Qwen3.5-4B"
TRACK_BASELINE_SCORE = 2.5

RUBRIC_TARGETS = {
    "collect_wood",
    "place_table",
    "make_wood_pickaxe",
    "collect_stone",
    "make_stone_pickaxe",
}

RESOURCE_KEYS = (
    "wood",
    "stone",
    "coal",
    "iron",
    "sapling",
    "drink",
    "food",
    "health",
    "energy",
)


class Rubric(TypedDict):
    target_achievement: str
    accept_actions: list[str]
    weak_accept_actions: list[str]
    reject_actions: list[str]
    forbidden_actions: list[str]
    inventory_requirements: dict[str, int]
    position_or_object_requirements: list[str]
    notes: str


@dataclass
class OfflineSample:
    group_id: str
    pivot_id: str
    prompt_messages: list[dict[str, str]]
    completion_text: str
    actions: list[str]
    reward: float
    advantage: float
    old_logprob: float
    reference_logprob: float


def now_utc_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def ensure_dir(path: str | Path) -> Path:
    target = Path(path).expanduser().resolve()
    target.mkdir(parents=True, exist_ok=True)
    return target


def write_json(path: str | Path, payload: Any) -> None:
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw in Path(path).expanduser().resolve().read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        payload = json.loads(line)
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def normalize_inference_url(raw_url: str) -> str:
    value = str(raw_url or "").strip().rstrip("/")
    if not value:
        return ""
    if value.endswith("/chat/completions"):
        return value
    if value.endswith("/v1"):
        return f"{value}/chat/completions"
    return f"{value}/v1/chat/completions"


@dataclass(frozen=True)
class LocalVLLMConfig:
    model: str
    served_model_name: str = ""
    lora_name: str = ""
    lora_path: str = ""
    max_lora_rank: int = 16
    port: int = DEFAULT_VLLM_PORT
    max_model_len: int = 8192
    max_new_tokens: int = 1024
    gpu_memory_utilization: float = DEFAULT_VLLM_GPU_UTILIZATION
    max_num_seqs: int = 16
    max_num_batched_tokens: int = 4096
    enable_thinking: bool = True
    enforce_eager: bool = False
    tool_call_parser: str = "qwen3_coder"
    reasoning_parser: str = "qwen3"
    vllm_bin: str = "vllm"


def infer_lora_rank(adapter_dir: str | Path) -> int:
    config_path = Path(adapter_dir).expanduser().resolve() / "adapter_config.json"
    if not config_path.exists():
        return 16
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    rank = payload.get("r")
    return int(rank) if isinstance(rank, int) and rank > 0 else 16


def build_vllm_serve_command(config: LocalVLLMConfig) -> list[str]:
    cmd = [
        config.vllm_bin,
        "serve",
        config.model,
        "--served-model-name",
        config.served_model_name or config.model,
        "--host",
        "127.0.0.1",
        "--port",
        str(config.port),
        "--max-model-len",
        str(config.max_model_len),
        "--max-num-seqs",
        str(config.max_num_seqs),
        "--max-num-batched-tokens",
        str(config.max_num_batched_tokens),
        "--gpu-memory-utilization",
        str(config.gpu_memory_utilization),
        "--uvicorn-log-level",
        "info",
        "--enable-auto-tool-choice",
        "--tool-call-parser",
        config.tool_call_parser,
        "--enable-prefix-caching",
    ]
    if config.enable_thinking:
        cmd += ["--reasoning-parser", config.reasoning_parser]
    if config.lora_path:
        cmd += [
            "--enable-lora",
            "--max-lora-rank",
            str(config.max_lora_rank),
            "--lora-modules",
            f"{config.lora_name}={config.lora_path}",
        ]
    if config.enforce_eager:
        cmd.append("--enforce-eager")
    return cmd


def wait_for_local_vllm(
    *,
    port: int,
    process: subprocess.Popen[str] | None = None,
    timeout_seconds: float = DEFAULT_REQUEST_TIMEOUT_S,
) -> None:
    deadline = time.time() + timeout_seconds
    last_error: Exception | None = None
    while time.time() < deadline:
        if process is not None and process.poll() is not None:
            raise RuntimeError(f"local vLLM exited before health check passed with code {process.returncode}")
        try:
            with httpx.Client(timeout=5.0) as client:
                response = client.get(f"http://127.0.0.1:{int(port)}/health")
                if response.status_code == 200:
                    return
                last_error = RuntimeError(f"/health returned HTTP {response.status_code}")
        except Exception as exc:
            last_error = exc
        time.sleep(1.0)
    raise RuntimeError(f"timed out waiting for local vLLM health: {last_error!r}")


@contextmanager
def local_vllm_server(*, config: LocalVLLMConfig, log_path: str | Path | None = None) -> Iterator[dict[str, Any]]:
    log_file = None
    process: subprocess.Popen[str] | None = None
    try:
        if log_path:
            target = Path(log_path).expanduser().resolve()
            target.parent.mkdir(parents=True, exist_ok=True)
            log_file = target.open("w", encoding="utf-8", buffering=1)
        command = build_vllm_serve_command(config)
        print(f"[pivotrl] starting local vLLM: {' '.join(shlex.quote(part) for part in command)}", flush=True)
        process = subprocess.Popen(
            command,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
            stdout=log_file or sys.stdout,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        wait_for_local_vllm(port=config.port, process=process)
        print(f"[pivotrl] local vLLM healthy on port {config.port}", flush=True)
        yield {"process": process, "base_url": f"http://127.0.0.1:{config.port}/v1"}
    finally:
        if process is not None:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=30)
        if log_file is not None:
            log_file.close()


def resolve_output_root(raw_output_root: str) -> Path:
    if raw_output_root.strip():
        return ensure_dir(raw_output_root)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return ensure_dir(REPO_ROOT / ".out" / "pivotrl" / timestamp)


def mean_or_zero(values: list[float]) -> float:
    return mean(values) if values else 0.0


def flatten_messages(messages: list[dict[str, Any]]) -> str:
    rendered: list[str] = []
    for message in messages:
        role = str(message.get("role") or "user").strip() or "user"
        content = message.get("content")
        if isinstance(content, list):
            content_text = "\n".join(
                str(item.get("text", "")) if isinstance(item, dict) else str(item)
                for item in content
            )
        else:
            content_text = str(content or "")
        rendered.append(f"{role}: {content_text}")
    return "\n".join(rendered).strip()


def rollout_turns(rollout: dict[str, Any]) -> list[dict[str, Any]]:
    artifact = rollout.get("artifact")
    if isinstance(artifact, list):
        for entry in artifact:
            if isinstance(entry, dict) and isinstance(entry.get("turns"), list):
                return [turn for turn in entry["turns"] if isinstance(turn, dict)]
    trace = rollout.get("trace")
    if isinstance(trace, dict):
        inference = trace.get("inference")
        if isinstance(inference, dict) and isinstance(inference.get("turns"), list):
            return [turn for turn in inference["turns"] if isinstance(turn, dict)]
    return []


def is_rollout_payload(rollout: dict[str, Any]) -> bool:
    if not isinstance(rollout, dict):
        return False
    reward_info = rollout.get("reward_info")
    trace = rollout.get("trace")
    success_status = str(rollout.get("success_status") or "").strip().lower()
    return isinstance(reward_info, dict) and isinstance(trace, dict) and success_status in {"success", "ok"}


def rollout_outcome_reward(rollout: dict[str, Any]) -> float:
    reward_info = rollout.get("reward_info")
    if isinstance(reward_info, dict):
        outcome_objectives = reward_info.get("outcome_objectives")
        if isinstance(outcome_objectives, dict):
            for key in ("unique_achievements", "reward"):
                try:
                    value = outcome_objectives.get(key)
                    if value is not None:
                        return float(value)
                except (TypeError, ValueError):
                    pass
        details = reward_info.get("details")
        if isinstance(details, dict):
            achievements = details.get("achievements")
            if isinstance(achievements, list):
                unique = {str(item).strip() for item in achievements if str(item).strip()}
                if unique:
                    return float(len(unique))
        try:
            return float(reward_info.get("outcome_reward", 0.0))
        except (TypeError, ValueError):
            return 0.0
    return 0.0


def rollout_achievements(rollout: dict[str, Any]) -> list[str]:
    metadata = rollout.get("metadata")
    if isinstance(metadata, dict):
        achievements = metadata.get("achievements")
        if isinstance(achievements, list):
            return [str(item).strip() for item in achievements if str(item).strip()]
    reward_info = rollout.get("reward_info")
    if isinstance(reward_info, dict):
        details = reward_info.get("details")
        if isinstance(details, dict):
            achievements = details.get("achievements")
            if isinstance(achievements, list):
                return [str(item).strip() for item in achievements if str(item).strip()]
    return []


def _is_synthtunnel_url(url: str) -> bool:
    try:
        parsed = httpx.URL(url)
        hostname = parsed.host or ""
        path = parsed.path or ""
    except Exception:
        return False
    return hostname == "st.usesynth.ai" or hostname.endswith(".st.usesynth.ai") or "/s/rt_" in path


def _is_cloudflare_quick_tunnel_url(url: str) -> bool:
    try:
        hostname = httpx.URL(url).host or ""
    except Exception:
        return False
    return hostname.endswith(".trycloudflare.com")


def _container_headers(
    *,
    container_url: str,
    container_worker_token: str | None,
    environment_api_key: str | None,
) -> dict[str, str]:
    headers: dict[str, str] = {"Content-Type": "application/json"}
    if _is_synthtunnel_url(container_url):
        worker_token = (container_worker_token or "").strip()
        if not worker_token:
            raise ValueError("container_worker_token is required for SynthTunnel container_url")
        headers["Authorization"] = f"Bearer {worker_token}"
        return headers
    env_key = (environment_api_key or "").strip()
    if env_key:
        headers["x-api-key"] = env_key
    return headers


def build_rollout_request(
    *,
    inference_url: str,
    model: str,
    api_key: str,
    seed: int,
    max_steps: int,
    trace_correlation_id: str,
    system_prompt: str,
    temperature: float = 0.0,
    max_tokens: int = 180,
    max_model_len: int = 8192,
    enable_thinking: bool = True,
    thinking_budget_tokens: int = 0,
    policy_version: str = "bootstrap",
    target_action_batch_size: int = 8,
    min_action_batch_size: int = 5,
    timeout_s: int = 45,
) -> dict[str, Any]:
    safe_max_tokens = min(int(max_tokens), max(256, int(max_model_len) - 2048))
    return {
        "trace_correlation_id": trace_correlation_id,
        "trial_id": trace_correlation_id,
        "env": {
            "seed": int(seed),
            "config": {
                "max_steps": int(max_steps),
                "episode_max_steps": int(max_steps),
            },
        },
        "policy": {
            "config": {
                "model": model,
                "api_key": api_key,
                "inference_url": inference_url,
                "temperature": float(temperature),
                "max_tokens": int(safe_max_tokens),
                "system_prompt": system_prompt,
                "enable_thinking": bool(enable_thinking),
                "thinking_budget_tokens": int(thinking_budget_tokens),
                "use_tools": True,
                "policy_version": str(policy_version),
                "route": "teacher" if "teacher" in str(policy_version).lower() else "student",
                "target_action_batch_size": int(target_action_batch_size),
                "min_action_batch_size": int(min_action_batch_size),
                "timeout_s": int(timeout_s),
            }
        },
    }


async def collect_rollouts_concurrently_with_summary(
    *,
    container_url: str,
    container_worker_token: str = "",
    environment_api_key: str = "",
    inference_url: str,
    model: str,
    api_key: str,
    seeds: list[int],
    max_steps: int,
    system_prompt: str,
    temperature: float,
    max_tokens: int,
    max_model_len: int,
    enable_thinking: bool,
    thinking_budget_tokens: int = 0,
    policy_version: str,
    target_action_batch_size: int,
    min_action_batch_size: int,
    request_timeout_seconds: float,
    max_concurrent_rollouts: int,
    trace_prefix: str,
    rollout_concurrency: int | None = None,
    rollout_semaphore_limit: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    worker_count = max(1, int(rollout_concurrency if rollout_concurrency is not None else max_concurrent_rollouts))
    permit_limit = max(
        1,
        int(rollout_semaphore_limit if rollout_semaphore_limit is not None else max_concurrent_rollouts),
    )
    semaphore = asyncio.Semaphore(permit_limit)
    timeout = httpx.Timeout(float(request_timeout_seconds), connect=min(30.0, float(request_timeout_seconds)))
    container_base = str(container_url).rstrip("/")
    headers = _container_headers(
        container_url=container_base,
        container_worker_token=container_worker_token,
        environment_api_key=environment_api_key,
    )
    is_quick_tunnel = _is_cloudflare_quick_tunnel_url(container_base)
    if is_quick_tunnel:
        headers = {**headers, "Connection": "close"}
    rollout_queue: asyncio.Queue[tuple[int, int] | None] = asyncio.Queue()
    for index, seed in enumerate(seeds):
        rollout_queue.put_nowait((index, int(seed)))
    results: list[dict[str, Any] | None] = [None] * len(seeds)
    request_latencies_s: list[float] = []
    active_rollouts = 0
    high_watermark = 0
    requests_started = 0
    requests_finished = 0
    started_at = time.perf_counter()

    async def _run_one(client: httpx.AsyncClient, seed: int, index: int) -> dict[str, Any]:
        request_body = build_rollout_request(
            inference_url=inference_url,
            model=model,
            api_key=api_key,
            seed=seed,
            max_steps=max_steps,
            trace_correlation_id=f"{trace_prefix}_{index:05d}_{seed}",
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            max_model_len=max_model_len,
            enable_thinking=enable_thinking,
            thinking_budget_tokens=thinking_budget_tokens,
            policy_version=policy_version,
            target_action_batch_size=target_action_batch_size,
            min_action_batch_size=min_action_batch_size,
            timeout_s=max(1, math.ceil(request_timeout_seconds)),
        )
        try:
            response = await client.post(
                f"{container_base}/rollout",
                headers=headers,
                json=request_body,
                follow_redirects=False,
            )
            deadline = time.perf_counter() + float(request_timeout_seconds)
            while True:
                if response.status_code == 303:
                    location = response.headers.get("location", "").strip()
                    if not location:
                        raise RuntimeError("rollout redirect missing Location header")
                    result_url = str(response.request.url.join(location))
                    response = await client.get(result_url, headers=headers, follow_redirects=False)
                    continue
                if response.status_code == 408 and "__modal_function_call_id=" in str(response.request.url):
                    if time.perf_counter() >= deadline:
                        raise RuntimeError(f"modal result url timed out: {response.request.url}")
                    await asyncio.sleep(1.0)
                    response = await client.get(str(response.request.url), headers=headers, follow_redirects=False)
                    continue
                break
            response.raise_for_status()
            payload = response.json()
            if isinstance(payload, dict):
                payload.setdefault("trace_correlation_id", request_body["trace_correlation_id"])
                payload.setdefault("trial_id", request_body["trial_id"])
                payload.setdefault("_request_seed", seed)
                return payload
            return {
                "error": "rollout response was not an object",
                "seed": seed,
                "trace_correlation_id": request_body["trace_correlation_id"],
            }
        except Exception as exc:
            error_text = str(exc).strip() or f"{type(exc).__name__}: no detail"
            return {
                "error": error_text,
                "seed": seed,
                "trace_correlation_id": request_body["trace_correlation_id"],
            }

    async def _worker(client: httpx.AsyncClient) -> None:
        nonlocal active_rollouts, high_watermark, requests_started, requests_finished
        while True:
            item = await rollout_queue.get()
            if item is None:
                rollout_queue.task_done()
                return
            index, seed = item
            async with semaphore:
                active_rollouts += 1
                high_watermark = max(high_watermark, active_rollouts)
                requests_started += 1
                request_started_at = time.perf_counter()
                try:
                    results[index] = await _run_one(client, seed, index)
                finally:
                    request_latencies_s.append(time.perf_counter() - request_started_at)
                    requests_finished += 1
                    active_rollouts -= 1
            rollout_queue.task_done()

    client_kwargs: dict[str, Any] = {"timeout": timeout}
    if is_quick_tunnel:
        client_kwargs.update(
            {
                "limits": httpx.Limits(max_connections=max(worker_count, permit_limit), max_keepalive_connections=0),
                "headers": {"Connection": "close"},
                "http2": False,
            }
        )

    async with httpx.AsyncClient(**client_kwargs) as client:
        workers = [asyncio.create_task(_worker(client)) for _ in range(worker_count)]
        await rollout_queue.join()
        for _worker_task in workers:
            rollout_queue.put_nowait(None)
        await asyncio.gather(*workers)

    completed_rollouts = [item for item in results if isinstance(item, dict)]
    valid_rollouts = [item for item in completed_rollouts if not item.get("error") and is_rollout_payload(item)]
    rewards = [rollout_outcome_reward(item) for item in valid_rollouts]
    elapsed_s = max(time.perf_counter() - started_at, 1e-9)
    summary = {
        "requested_rollouts": len(seeds),
        "completed_rollouts": len(completed_rollouts),
        "num_errors": len(completed_rollouts) - len(valid_rollouts),
        "num_structured_rollouts": len(valid_rollouts),
        "mean_outcome_reward": mean(rewards) if rewards else 0.0,
        "max_outcome_reward": max(rewards) if rewards else 0.0,
        "elapsed_s": elapsed_s,
        "rollouts_per_minute": len(valid_rollouts) / (elapsed_s / 60.0),
        "rollout_concurrency": worker_count,
        "rollout_semaphore_limit": permit_limit,
        "rollout_requests_started": requests_started,
        "rollout_requests_finished": requests_finished,
        "active_rollout_high_watermark": high_watermark,
        "mean_request_latency_s": mean(request_latencies_s) if request_latencies_s else 0.0,
        "max_request_latency_s": max(request_latencies_s) if request_latencies_s else 0.0,
    }
    normalized_results = [item if isinstance(item, dict) else {"error": "missing rollout result"} for item in results]
    return normalized_results, summary


def rollout_system_prompt(
    *,
    thinking_budget_tokens: int,
    target_action_batch_size: int,
    min_action_batch_size: int,
) -> str:
    if target_action_batch_size == min_action_batch_size:
        action_instruction = f"Return exactly {target_action_batch_size} valid Craftax actions."
    else:
        action_instruction = (
            f"Return a short useful macro-action with {min_action_batch_size}-{target_action_batch_size} "
            "valid Craftax actions."
        )
    return (
        "You are a Craftax teacher policy.\n"
        f"You may think for up to about {thinking_budget_tokens} tokens before answering.\n"
        f"{action_instruction}\n"
        "Use movement to explore when nothing useful is adjacent.\n"
        "Use 'do' only when facing a useful nearby object or resource.\n"
        "Read the recent action history and avoid repeating unproductive loops.\n"
        "Use the provided `craftax_interact` tool exactly once for the final answer.\n"
        "Do not return plain text actions or JSON.\n"
        "Your final assistant action must be a tool call with valid Craftax actions."
    )


def release_cuda_memory() -> None:
    import gc

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            with suppress(Exception):
                torch.cuda.ipc_collect()
    except Exception:
        pass


def _cuda_base_image() -> modal.Image:
    return (
        modal.Image.from_registry(
            "nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04",
            add_python="3.11",
        )
        .apt_install("git", "curl", "build-essential", "ninja-build", "libgl1", "libglib2.0-0")
    )


def _attach_repo(image: modal.Image) -> modal.Image:
    def _ignore_local_copy(path: Path) -> bool:
        parts = set(path.parts)
        if "__pycache__" in parts:
            return True
        if path.suffix in {".pyc", ".pyo"}:
            return True
        if any(part in {".git", ".out", "artifacts"} for part in parts):
            return True
        return False

    return (
        image.add_local_dir(
            (PROJECT_ROOT / "src").as_posix(),
            remote_path=f"{REMOTE_ROOT}/src",
            copy=True,
            ignore=_ignore_local_copy,
        )
        .add_local_dir(
            (PROJECT_ROOT / "scripts").as_posix(),
            remote_path=f"{REMOTE_ROOT}/scripts",
            copy=True,
            ignore=_ignore_local_copy,
        )
        .add_local_dir(
            (PROJECT_ROOT / "configs").as_posix(),
            remote_path=f"{REMOTE_ROOT}/configs",
            copy=True,
            ignore=_ignore_local_copy,
        )
        .add_local_dir(
            (PROJECT_ROOT / "data").as_posix(),
            remote_path=f"{REMOTE_ROOT}/data",
            copy=True,
            ignore=_ignore_local_copy,
        )
        .add_local_file((PROJECT_ROOT / "pyproject.toml").as_posix(), remote_path=f"{REMOTE_ROOT}/pyproject.toml", copy=True)
        .add_local_file((PROJECT_ROOT / "README.md").as_posix(), remote_path=f"{REMOTE_ROOT}/README.md", copy=True)
        .add_local_file(Path(__file__).as_posix(), remote_path=f"{REMOTE_ROOT}/submissions/synth/pivotrl.py", copy=True)
    )


def pivotrl_runtime_image() -> modal.Image:
    image = (
        _cuda_base_image()
        .pip_install(
            "httpx>=0.28.1",
            "pyyaml>=6.0.2",
            "modal>=1.0.0",
            "accelerate>=1.10.0",
            "datasets>=4.1.0",
            "peft>=0.15.0",
            "torch>=2.6.0",
            "transformers>=4.56.1,<4.57",
            "vllm>=0.10.0",
        )
        .env({"PYTHONPATH": f"{REMOTE_ROOT}/src"})
    )
    image = _attach_repo(image)
    return image


app = modal.App(APP_NAME)
image = pivotrl_runtime_image()


def _load_text_only_causal_lm(*, base_model: str, device: str, use_cache: bool = True) -> Any:
    import torch
    from transformers import AutoModelForCausalLM

    model_kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "torch_dtype": torch.bfloat16 if device == "cuda" else torch.float32,
    }
    if device == "cuda":
        model_kwargs["device_map"] = "auto"
    try:
        model = AutoModelForCausalLM.from_pretrained(base_model, **model_kwargs)
    except Exception:
        from transformers import Qwen3_5ForCausalLM

        model = Qwen3_5ForCausalLM.from_pretrained(base_model, **model_kwargs)
    model.config.use_cache = use_cache
    return model


DEFAULT_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
    "in_proj_qkv",
    "in_proj_z",
    "in_proj_b",
    "in_proj_a",
    "out_proj",
]

FALLBACK_TARGET_MODULES = [
    "c_attn",
    "c_proj",
    "c_fc",
]


def infer_target_modules(model: Any) -> list[str]:
    available_suffixes = {
        str(name).split(".")[-1]
        for name, _module in model.named_modules()
    }
    preferred = [name for name in DEFAULT_TARGET_MODULES if name in available_suffixes]
    if preferred:
        return preferred
    fallback = [name for name in FALLBACK_TARGET_MODULES if name in available_suffixes]
    if fallback:
        return fallback
    linear_like_suffixes: list[str] = []
    for name, module in model.named_modules():
        suffix = str(name).split(".")[-1]
        if suffix in linear_like_suffixes:
            continue
        class_name = type(module).__name__.lower()
        if "linear" in class_name or "conv1d" in class_name:
            linear_like_suffixes.append(suffix)
    if linear_like_suffixes:
        return linear_like_suffixes[:6]
    raise RuntimeError("unable to infer LoRA target modules for model")


def initialize_policy(*, base_model: str, lora_rank: int, learning_rate: float) -> tuple[Any, Any, Any]:
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = _load_text_only_causal_lm(base_model=base_model, device=device, use_cache=False)
    target_modules = infer_target_modules(model)
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_rank * 2,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )
    model = get_peft_model(model, lora_config)
    with suppress(Exception):
        model.enable_input_require_grads()
    with suppress(Exception):
        model.gradient_checkpointing_enable()
    with suppress(Exception):
        model.config.use_cache = False
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    return tokenizer, model, optimizer


def initialize_reference_policy(*, base_model: str) -> tuple[Any, Any]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    device = "cuda"
    try:
        import torch

        if not torch.cuda.is_available():
            device = "cpu"
    except Exception:
        device = "cpu"
    model = _load_text_only_causal_lm(base_model=base_model, device=device, use_cache=True)
    model.eval()
    return tokenizer, model


def normalize_messages_for_chat_template(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for item in messages:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role") or "user").strip() or "user"
        normalized.append(
            {
                "role": role,
                "content": str(item.get("content") or ""),
            }
        )
    return normalized


def render_prompt_text(tokenizer: Any, messages: list[dict[str, str]]) -> str:
    normalized = normalize_messages_for_chat_template(messages)
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            rendered = tokenizer.apply_chat_template(
                normalized,
                tokenize=False,
                add_generation_prompt=True,
            )
            if isinstance(rendered, str):
                return rendered
        except Exception:
            pass
    rendered_lines = [f"<|{item['role']}|>\n{item['content']}" for item in normalized]
    rendered_lines.append("<|assistant|>\n")
    return "\n".join(rendered_lines)


def tokenize_prompt_and_completion(
    tokenizer: Any,
    prompt_messages: list[dict[str, str]],
    completion_text: str,
    *,
    max_length: int,
) -> dict[str, Any]:
    import torch

    prompt_text = render_prompt_text(tokenizer, prompt_messages)
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    completion_ids = tokenizer(str(completion_text or ""), add_special_tokens=False)["input_ids"]
    full_ids = (prompt_ids + completion_ids)[:max_length]
    labels = ([-100] * len(prompt_ids) + completion_ids)[:max_length]
    attention_mask = [1] * len(full_ids)
    return {
        "prompt_text": prompt_text,
        "input_ids": torch.tensor([full_ids], dtype=torch.long),
        "labels": torch.tensor([labels], dtype=torch.long),
        "attention_mask": torch.tensor([attention_mask], dtype=torch.long),
    }


def selected_token_logprobs(logits: Any, shifted_labels: Any) -> tuple[Any, Any]:
    import torch

    mask = shifted_labels != -100
    safe_targets = torch.clamp(shifted_labels, min=0)
    selected_logits = logits.gather(dim=-1, index=safe_targets.unsqueeze(-1)).squeeze(-1)
    lse = torch.logsumexp(logits, dim=-1)
    return (selected_logits - lse) * mask, mask


def sequence_logprob(
    *,
    tokenizer: Any,
    model: Any,
    prompt_messages: list[dict[str, str]],
    completion_text: str,
    max_length: int,
    disable_adapter: bool = False,
) -> float:
    batch = tokenize_prompt_and_completion(
        tokenizer,
        prompt_messages,
        completion_text,
        max_length=max_length,
    )
    batch = {key: value.to(model.device) for key, value in batch.items() if key in {"input_ids", "labels", "attention_mask"}}
    context = model.disable_adapter() if disable_adapter and hasattr(model, "disable_adapter") else nullcontext()
    with context:
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            use_cache=False,
        )
    shifted_logits = outputs.logits[:, :-1, :]
    shifted_labels = batch["labels"][:, 1:]
    selected_log_probs, _mask = selected_token_logprobs(shifted_logits, shifted_labels)
    return float(selected_log_probs.sum(dim=1).detach().cpu().item())


def sample_completion_text(
    *,
    tokenizer: Any,
    model: Any,
    prompt_messages: list[dict[str, str]],
    max_length: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> str:
    encoded_prompt = render_prompt_text(tokenizer, prompt_messages)
    inputs = tokenizer(encoded_prompt, return_tensors="pt", add_special_tokens=False)
    inputs = {key: value.to(model.device) for key, value in inputs.items()}
    generation_kwargs: dict[str, Any] = {
        "max_new_tokens": int(max_new_tokens),
        "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "use_cache": True,
    }
    if float(temperature) > 0.0:
        generation_kwargs.update(
            {
                "do_sample": True,
                "temperature": float(temperature),
                "top_p": float(top_p),
            }
        )
    else:
        generation_kwargs["do_sample"] = False
    with suppress(Exception):
        generation_kwargs["attention_mask"] = inputs.get("attention_mask")
    outputs = model.generate(inputs["input_ids"], **generation_kwargs)
    completion_ids = outputs[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(completion_ids, skip_special_tokens=True).strip()


def build_tool_call_text(actions: list[str]) -> str:
    payload = {
        "name": "craftax_interact",
        "arguments": {
            "actions_list": actions,
        },
    }
    return f"<tool_call>{json.dumps(payload, separators=(',', ':'))}</tool_call>"


def extract_actions_from_text(text: str) -> list[str]:
    raw = str(text or "")
    if not raw.strip():
        return []

    tool_call_match = re.search(r"<tool_call>\s*(.*?)\s*</tool_call>", raw, flags=re.DOTALL)
    if tool_call_match:
        block = tool_call_match.group(1).strip()
        try:
            payload = json.loads(block)
        except Exception:
            payload = None
        if isinstance(payload, dict):
            arguments = payload.get("arguments", {})
            if isinstance(arguments, dict):
                values = arguments.get("actions_list")
                if isinstance(values, list):
                    return [str(item).strip().lower() for item in values if str(item).strip().lower() in ACTION_SET]

    sequence: list[str] = []
    pattern = r"\b(" + "|".join(re.escape(action) for action in sorted(ACTION_SET, key=len, reverse=True)) + r")\b"
    for match in re.finditer(pattern, raw.lower()):
        token = match.group(1)
        if token in ACTION_SET:
            sequence.append(token)
    return sequence


def latest_user_text(messages: list[dict[str, Any]]) -> str:
    for message in reversed(messages):
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "").strip().lower()
        if role != "user":
            continue
        content = message.get("content")
        if isinstance(content, list):
            parts = [str(item.get("text") or "") for item in content if isinstance(item, dict)]
            return "\n".join(parts).strip()
        return str(content or "").strip()
    return flatten_messages([item for item in messages if isinstance(item, dict)])


def parse_inventory_from_text(text: str) -> dict[str, int]:
    normalized = str(text or "").lower()
    inventory: dict[str, int] = {}
    for key in RESOURCE_KEYS:
        match = re.search(rf"\b{re.escape(key)}\s*=\s*(-?\d+)\b", normalized)
        if match:
            inventory[key] = max(0, int(match.group(1)))
    return inventory


def summarize_inventory(inventory: dict[str, int]) -> str:
    if not inventory:
        return "inventory unavailable"
    ordered = [f"{key}={inventory[key]}" for key in sorted(inventory) if inventory[key] > 0]
    return ", ".join(ordered) if ordered else "inventory empty"


def infer_target_achievement(
    *,
    current_turn: dict[str, Any],
    next_turn: dict[str, Any],
) -> tuple[str | None, str]:
    current_actions = current_turn.get("actions")
    actions = (
        [str(item).strip().lower() for item in current_actions if str(item).strip()]
        if isinstance(current_actions, list)
        else extract_actions_from_text(str(current_turn.get("assistant_text") or ""))
    )
    current_text = latest_user_text([item for item in current_turn.get("prompt_messages", []) if isinstance(item, dict)])
    next_text = latest_user_text([item for item in next_turn.get("prompt_messages", []) if isinstance(item, dict)])
    current_inventory = parse_inventory_from_text(current_text)
    next_inventory = parse_inventory_from_text(next_text)

    if "make_stone_pickaxe" in actions:
        return "make_stone_pickaxe", "action matched target craft"
    if "make_wood_pickaxe" in actions:
        return "make_wood_pickaxe", "action matched target craft"
    if "place_table" in actions:
        return "place_table", "action matched table placement"
    if next_inventory.get("stone", 0) > current_inventory.get("stone", 0):
        return "collect_stone", "inventory delta on stone"
    if next_inventory.get("wood", 0) > current_inventory.get("wood", 0):
        return "collect_wood", "inventory delta on wood"
    if "do" in actions and "tree" in current_text.lower():
        return "collect_wood", "interaction near tree"
    if "do" in actions and "stone" in current_text.lower():
        return "collect_stone", "interaction near stone"
    return None, ""


def infer_bootstrap_target_from_rollout(
    *,
    rollout: dict[str, Any],
    state_text: str,
    inventory: dict[str, int],
) -> tuple[str | None, str]:
    normalized = state_text.lower()
    achievements = {item.strip().lower() for item in rollout_achievements(rollout)}
    if inventory.get("wood", 0) <= 0 and ("tree" in normalized or "collect wood" in normalized):
        return "collect_wood", "bootstrap fallback from nearby tree state"
    if inventory.get("wood", 0) >= 2 and "place_table" not in achievements:
        return "place_table", "bootstrap fallback from inventory prerequisites"
    if inventory.get("wood_pickaxe", 0) <= 0 and inventory.get("wood", 0) >= 1:
        return "make_wood_pickaxe", "bootstrap fallback from craft prerequisites"
    if inventory.get("stone", 0) <= 0 and inventory.get("wood_pickaxe", 0) > 0:
        return "collect_stone", "bootstrap fallback from progression ordering"
    if inventory.get("stone_pickaxe", 0) <= 0 and inventory.get("stone", 0) >= 1:
        return "make_stone_pickaxe", "bootstrap fallback from craft prerequisites"
    if "tree" in normalized:
        return "collect_wood", "bootstrap fallback from visible tree"
    if "stone" in normalized:
        return "collect_stone", "bootstrap fallback from visible stone"
    return None, ""


def build_rubric(
    *,
    target_achievement: str,
    state_text: str,
    inventory: dict[str, int],
) -> Rubric:
    normalized_state = state_text.lower()
    shared_reject = [
        "sleep",
        "place_furnace",
        "place_plant",
        "make_wood_sword",
        "make_stone_sword",
        "make_iron_sword",
        "make_iron_pickaxe",
    ]
    if target_achievement == "collect_wood":
        return {
            "target_achievement": target_achievement,
            "accept_actions": ["do"],
            "weak_accept_actions": ["move_left", "move_right", "move_up", "move_down"],
            "reject_actions": [*shared_reject, "place_table", "place_stone", "make_wood_pickaxe", "make_stone_pickaxe"],
            "forbidden_actions": [],
            "inventory_requirements": {},
            "position_or_object_requirements": ["tree adjacent"] if "tree" in normalized_state else [],
            "notes": "Prefer immediate harvesting when a tree is adjacent; otherwise move to line up a harvest.",
        }
    if target_achievement == "collect_stone":
        return {
            "target_achievement": target_achievement,
            "accept_actions": ["do"],
            "weak_accept_actions": ["move_left", "move_right", "move_up", "move_down"],
            "reject_actions": [*shared_reject, "place_table", "place_stone"],
            "forbidden_actions": [],
            "inventory_requirements": {},
            "position_or_object_requirements": ["stone adjacent"] if "stone" in normalized_state else [],
            "notes": "Prefer direct stone collection when stone is reachable; otherwise move toward stone.",
        }
    if target_achievement == "place_table":
        return {
            "target_achievement": target_achievement,
            "accept_actions": ["place_table"],
            "weak_accept_actions": ["move_left", "move_right", "move_up", "move_down"],
            "reject_actions": [*shared_reject, "place_stone", "make_wood_pickaxe", "make_stone_pickaxe"],
            "forbidden_actions": [],
            "inventory_requirements": {"wood": max(2, inventory.get("wood", 0)) if inventory.get("wood", 0) else 2},
            "position_or_object_requirements": [],
            "notes": "Table placement is the direct precondition for early crafting.",
        }
    if target_achievement == "make_wood_pickaxe":
        return {
            "target_achievement": target_achievement,
            "accept_actions": ["make_wood_pickaxe"],
            "weak_accept_actions": ["place_table", "do"],
            "reject_actions": [*shared_reject, "place_stone", "make_stone_pickaxe"],
            "forbidden_actions": [],
            "inventory_requirements": {"wood": max(1, inventory.get("wood", 0)) if inventory.get("wood", 0) else 1},
            "position_or_object_requirements": ["table adjacent"] if "table" in normalized_state else [],
            "notes": "If the table is ready and wood is available, craft the wood pickaxe immediately.",
        }
    if target_achievement == "make_stone_pickaxe":
        return {
            "target_achievement": target_achievement,
            "accept_actions": ["make_stone_pickaxe"],
            "weak_accept_actions": ["make_wood_pickaxe", "do"],
            "reject_actions": [*shared_reject, "place_stone", "make_wood_sword"],
            "forbidden_actions": [],
            "inventory_requirements": {
                "wood": max(1, inventory.get("wood", 0)) if inventory.get("wood", 0) else 1,
                "stone": max(1, inventory.get("stone", 0)) if inventory.get("stone", 0) else 1,
            },
            "position_or_object_requirements": ["table adjacent"] if "table" in normalized_state else [],
            "notes": "Stone pickaxe crafting is the direct target; related setup remains weakly positive.",
        }
    raise ValueError(f"unsupported rubric target: {target_achievement}")


def requirements_satisfied(
    *,
    rubric: Rubric,
    state_text: str,
    inventory: dict[str, int],
) -> bool:
    normalized_state = state_text.lower()
    for phrase in rubric["position_or_object_requirements"]:
        if phrase and phrase.lower() not in normalized_state:
            return False
    for key, required in rubric["inventory_requirements"].items():
        if inventory.get(key, 0) < int(required):
            return False
    return True


def score_actions_against_rubric(
    *,
    actions: list[str],
    rubric: Rubric,
    state_text: str,
    inventory: dict[str, int],
) -> tuple[float, str]:
    if not actions:
        return -1.0, "no valid actions parsed"
    if any(action in rubric["forbidden_actions"] for action in actions):
        return -1.0, "forbidden action present"
    has_accept = any(action in rubric["accept_actions"] for action in actions)
    has_weak = any(action in rubric["weak_accept_actions"] for action in actions)
    has_reject = any(action in rubric["reject_actions"] for action in actions)
    if has_accept and requirements_satisfied(rubric=rubric, state_text=state_text, inventory=inventory):
        return 1.0, "strong accept action with satisfied requirements"
    if has_accept:
        return -0.5, "strong accept action attempted without requirements"
    if has_weak:
        return 0.5, "weak accept action"
    if has_reject:
        return -0.5, "reject action"
    return 0.0, "neutral action"


def build_pivot_prompt_messages(pivot: dict[str, Any]) -> list[dict[str, str]]:
    rubric = pivot["rubric"]
    inventory_summary = pivot.get("pre_achievement_inventory_or_summary", "")
    if isinstance(inventory_summary, dict):
        inventory_text = summarize_inventory(
            {
                str(key): int(value)
                for key, value in inventory_summary.items()
                if isinstance(value, (int, float))
            }
        )
    else:
        inventory_text = str(inventory_summary or "")
    prompt = (
        "State observation:\n"
        f"{pivot['state_text']}\n\n"
        f"Target achievement: {pivot['target_achievement']}\n"
        f"Inventory summary: {inventory_text or 'inventory unavailable'}\n"
        "Rubric:\n"
        f"- Strong accept actions: {', '.join(rubric['accept_actions']) or 'none'}\n"
        f"- Weak accept actions: {', '.join(rubric['weak_accept_actions']) or 'none'}\n"
        f"- Reject actions: {', '.join(rubric['reject_actions']) or 'none'}\n"
        f"- Forbidden actions: {', '.join(rubric['forbidden_actions']) or 'none'}\n"
        f"- Inventory requirements: {json.dumps(rubric['inventory_requirements'], sort_keys=True)}\n"
        f"- Position requirements: {', '.join(rubric['position_or_object_requirements']) or 'none'}\n"
        f"- Notes: {rubric['notes']}\n\n"
        "Respond with exactly one tool-call block and no extra text:\n"
        '<tool_call>{"name":"craftax_interact","arguments":{"actions_list":["move_up","do"]}}</tool_call>'
    )
    return [
        {
            "role": "system",
            "content": (
                "You are a Craftax pivot policy.\n"
                "Choose a short macro-action sequence that maximizes the rubric reward for the current state.\n"
                "Do not write analysis outside the final tool call."
            ),
        },
        {
            "role": "user",
            "content": prompt,
        },
    ]


def build_candidate_pivots(
    *,
    rollouts: list[dict[str, Any]],
    lookback: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    target_counts = {target: 0 for target in sorted(RUBRIC_TARGETS)}
    skipped_missing_target = 0
    fallback_count = 0
    for rollout in rollouts:
        if not isinstance(rollout, dict):
            continue
        turns = rollout_turns(rollout)
        if len(turns) <= lookback:
            trace_id = str(rollout.get("trace_correlation_id") or rollout.get("trial_id") or "")
            rollout_id = str(rollout.get("rollout_id") or trace_id)
            seed = int(rollout.get("_request_seed") or 0)
            outcome_reward = float(rollout_outcome_reward(rollout))
            metadata = rollout.get("metadata")
            inventory = {}
            if isinstance(metadata, dict) and isinstance(metadata.get("inventory"), dict):
                inventory = {
                    str(key): int(value)
                    for key, value in metadata["inventory"].items()
                    if isinstance(value, (int, float))
                }
            state_text = str(
                (
                    (rollout.get("reward_info") or {}).get("details", {}).get("last_inference_error")
                    if isinstance(rollout.get("reward_info"), dict)
                    and isinstance((rollout.get("reward_info") or {}).get("details"), dict)
                    else ""
                )
                or rollout.get("status_detail")
                or json.dumps(metadata or {}, sort_keys=True)
            )
            if not inventory:
                inventory = parse_inventory_from_text(state_text)
            target_achievement, transition_reason = infer_bootstrap_target_from_rollout(
                rollout=rollout,
                state_text=state_text,
                inventory=inventory,
            )
            if target_achievement not in RUBRIC_TARGETS:
                continue
            rubric = build_rubric(
                target_achievement=target_achievement,
                state_text=state_text,
                inventory=inventory,
            )
            candidate = {
                "pivot_id": f"{trace_id or rollout_id}_bootstrap_fallback",
                "trace_id": trace_id,
                "rollout_id": rollout_id,
                "seed": seed,
                "turn_index": -1,
                "state_messages": [],
                "state_text": state_text,
                "demonstrated_action": [],
                "target_achievement": target_achievement,
                "pre_achievement_inventory_or_summary": inventory,
                "rubric": rubric,
                "transition_reason": transition_reason,
                "source_outcome_reward": outcome_reward,
                "next_state_text": "",
                "training_messages": build_pivot_prompt_messages(
                    {
                        "state_text": state_text,
                        "target_achievement": target_achievement,
                        "pre_achievement_inventory_or_summary": inventory,
                        "rubric": rubric,
                    }
                ),
            }
            target_counts[target_achievement] += 1
            fallback_count += 1
            candidates.append(candidate)
            continue
        trace_id = str(rollout.get("trace_correlation_id") or rollout.get("trial_id") or "")
        rollout_id = str(rollout.get("rollout_id") or trace_id)
        seed = int(rollout.get("_request_seed") or 0)
        outcome_reward = float(rollout_outcome_reward(rollout))
        for idx in range(lookback, len(turns)):
            pivot_turn = turns[idx - lookback]
            achievement_turn = turns[idx]
            target_achievement, transition_reason = infer_target_achievement(
                current_turn=pivot_turn,
                next_turn=achievement_turn,
            )
            if target_achievement not in RUBRIC_TARGETS:
                skipped_missing_target += 1
                continue
            prompt_messages = [
                item
                for item in pivot_turn.get("prompt_messages", [])
                if isinstance(item, dict)
            ]
            state_text = latest_user_text(prompt_messages)
            inventory = parse_inventory_from_text(state_text)
            demonstrated_actions = (
                [str(item).strip().lower() for item in pivot_turn.get("actions", []) if str(item).strip()]
                if isinstance(pivot_turn.get("actions"), list)
                else extract_actions_from_text(str(pivot_turn.get("assistant_text") or ""))
            )
            rubric = build_rubric(
                target_achievement=target_achievement,
                state_text=state_text,
                inventory=inventory,
            )
            candidate = {
                "pivot_id": f"{trace_id or rollout_id}_turn_{int(pivot_turn.get('turn_index') or idx - lookback):04d}",
                "trace_id": trace_id,
                "rollout_id": rollout_id,
                "seed": seed,
                "turn_index": int(pivot_turn.get("turn_index") or idx - lookback),
                "state_messages": prompt_messages,
                "state_text": state_text,
                "demonstrated_action": demonstrated_actions,
                "target_achievement": target_achievement,
                "pre_achievement_inventory_or_summary": inventory,
                "rubric": rubric,
                "transition_reason": transition_reason,
                "source_outcome_reward": outcome_reward,
                "next_state_text": latest_user_text(
                    [item for item in achievement_turn.get("prompt_messages", []) if isinstance(item, dict)]
                ),
                "training_messages": build_pivot_prompt_messages(
                    {
                        "state_text": state_text,
                        "target_achievement": target_achievement,
                        "pre_achievement_inventory_or_summary": inventory,
                        "rubric": rubric,
                    }
                ),
            }
            target_counts[target_achievement] += 1
            candidates.append(candidate)
    summary = {
        "candidate_count": len(candidates),
        "skipped_missing_target": skipped_missing_target,
        "fallback_count": fallback_count,
        "target_counts": target_counts,
    }
    return candidates, summary


def profile_pivots(
    *,
    pivots: list[dict[str, Any]],
    base_model: str,
    profile_k: int,
    lambda_diff: float,
    max_length: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    max_pivots: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    selected_pivots = list(pivots)
    if max_pivots > 0:
        selected_pivots = selected_pivots[:max_pivots]
    if profile_k <= 0:
        passthrough = [
            {
                **pivot,
                "profile_stats": {
                    "reference_rewards": [],
                    "reference_action_samples": [],
                    "mu_hat": 0.0,
                    "sigma_hat_sq": 0.0,
                    "kept": True,
                    "profile_k": 0,
                },
            }
            for pivot in selected_pivots
        ]
        return passthrough, {
            "profiled_count": len(passthrough),
            "kept_count": len(passthrough),
            "dropped_count": 0,
            "lambda_diff": float(lambda_diff),
            "profile_k": 0,
        }

    tokenizer, model = initialize_reference_policy(base_model=base_model)
    kept: list[dict[str, Any]] = []
    reward_means: list[float] = []
    reward_variances: list[float] = []
    for pivot in selected_pivots:
        reference_rewards: list[float] = []
        reference_action_samples: list[dict[str, Any]] = []
        state_text = str(pivot["state_text"])
        inventory = {
            str(key): int(value)
            for key, value in dict(pivot["pre_achievement_inventory_or_summary"]).items()
        }
        training_messages = [
            {"role": str(item["role"]), "content": str(item["content"])}
            for item in pivot["training_messages"]
        ]
        for _index in range(profile_k):
            raw_completion = sample_completion_text(
                tokenizer=tokenizer,
                model=model,
                prompt_messages=training_messages,
                max_length=max_length,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )
            actions = extract_actions_from_text(raw_completion)
            if not actions:
                actions = ["sleep"]
            reward, reward_reason = score_actions_against_rubric(
                actions=actions,
                rubric=pivot["rubric"],
                state_text=state_text,
                inventory=inventory,
            )
            reference_rewards.append(reward)
            reference_action_samples.append(
                {
                    "actions": actions,
                    "raw_completion": raw_completion,
                    "reward": reward,
                    "reward_reason": reward_reason,
                }
            )
        mu_hat = mean_or_zero(reference_rewards)
        sigma_hat_sq = (
            sum((reward - mu_hat) ** 2 for reward in reference_rewards) / float(len(reference_rewards))
            if reference_rewards
            else 0.0
        )
        profiled = {
            **pivot,
            "profile_stats": {
                "reference_rewards": reference_rewards,
                "reference_action_samples": reference_action_samples,
                "mu_hat": mu_hat,
                "sigma_hat_sq": sigma_hat_sq,
                "kept": bool(sigma_hat_sq > 0.0 and mu_hat < float(lambda_diff)),
                "profile_k": int(profile_k),
            },
        }
        reward_means.append(mu_hat)
        reward_variances.append(sigma_hat_sq)
        if sigma_hat_sq > 0.0 and mu_hat < float(lambda_diff):
            kept.append(profiled)
    release_cuda_memory()
    return kept, {
        "profiled_count": len(selected_pivots),
        "kept_count": len(kept),
        "dropped_count": len(selected_pivots) - len(kept),
        "lambda_diff": float(lambda_diff),
        "profile_k": int(profile_k),
        "mean_mu_hat": mean_or_zero(reward_means),
        "mean_sigma_hat_sq": mean_or_zero(reward_variances),
    }


def group_advantages(rewards: list[float]) -> list[float]:
    if not rewards:
        return []
    mean_reward = sum(rewards) / len(rewards)
    variance = sum((reward - mean_reward) ** 2 for reward in rewards) / len(rewards)
    std = math.sqrt(max(variance, 1e-12))
    if std <= 1e-6:
        return [float(reward - mean_reward) for reward in rewards]
    return [float((reward - mean_reward) / std) for reward in rewards]


def build_group_samples(
    *,
    tokenizer: Any,
    model: Any,
    pivots: list[dict[str, Any]],
    group_size: int,
    max_length: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> tuple[list[OfflineSample], dict[str, Any]]:
    samples: list[OfflineSample] = []
    per_group_rewards: list[float] = []
    per_group_variances: list[float] = []
    for group_index, pivot in enumerate(pivots):
        rewards: list[float] = []
        completions: list[tuple[str, list[str], str]] = []
        training_messages = [
            {"role": str(item["role"]), "content": str(item["content"])}
            for item in pivot["training_messages"]
        ]
        state_text = str(pivot["state_text"])
        inventory = {
            str(key): int(value)
            for key, value in dict(pivot["pre_achievement_inventory_or_summary"]).items()
        }
        for _sample_idx in range(max(1, int(group_size))):
            raw_completion = sample_completion_text(
                tokenizer=tokenizer,
                model=model,
                prompt_messages=training_messages,
                max_length=max_length,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )
            actions = extract_actions_from_text(raw_completion)
            if not actions:
                actions = ["sleep"]
            canonical_completion = build_tool_call_text(actions)
            reward, _reason = score_actions_against_rubric(
                actions=actions,
                rubric=pivot["rubric"],
                state_text=state_text,
                inventory=inventory,
            )
            rewards.append(reward)
            completions.append((canonical_completion, actions, raw_completion))
        advantages = group_advantages(rewards)
        mu_hat = mean_or_zero(rewards)
        sigma_hat_sq = (
            sum((reward - mu_hat) ** 2 for reward in rewards) / float(len(rewards))
            if rewards
            else 0.0
        )
        per_group_rewards.extend(rewards)
        per_group_variances.append(sigma_hat_sq)
        for sample_idx, ((completion_text, actions, _raw_completion), reward, advantage) in enumerate(
            zip(completions, rewards, advantages, strict=False)
        ):
            old_logprob = sequence_logprob(
                tokenizer=tokenizer,
                model=model,
                prompt_messages=training_messages,
                completion_text=completion_text,
                max_length=max_length,
                disable_adapter=False,
            )
            reference_logprob = sequence_logprob(
                tokenizer=tokenizer,
                model=model,
                prompt_messages=training_messages,
                completion_text=completion_text,
                max_length=max_length,
                disable_adapter=True,
            )
            samples.append(
                OfflineSample(
                    group_id=f"group_{group_index:05d}",
                    pivot_id=str(pivot["pivot_id"]),
                    prompt_messages=training_messages,
                    completion_text=completion_text,
                    actions=actions,
                    reward=float(reward),
                    advantage=float(advantage),
                    old_logprob=float(old_logprob),
                    reference_logprob=float(reference_logprob),
                )
            )
    return samples, {
        "group_count": len(pivots),
        "sample_count": len(samples),
        "mean_group_reward": mean_or_zero(per_group_rewards),
        "max_group_reward": max(per_group_rewards) if per_group_rewards else 0.0,
        "mean_group_variance": mean_or_zero(per_group_variances),
    }


def train_iteration(
    *,
    tokenizer: Any,
    model: Any,
    optimizer: Any,
    samples: list[OfflineSample],
    clip_epsilon: float,
    kl_coef: float,
    max_length: int,
    max_steps: int,
) -> dict[str, Any]:
    import torch

    if not samples or max_steps <= 0:
        return {
            "optimizer_steps": 0,
            "mean_loss": 0.0,
            "mean_ratio": 0.0,
            "mean_reward": 0.0,
            "mean_advantage": 0.0,
            "mean_reference_gap": 0.0,
            "clip_fraction": 0.0,
            "skipped": True,
        }

    shuffled = list(samples)
    random.shuffle(shuffled)
    losses: list[float] = []
    ratios: list[float] = []
    rewards: list[float] = []
    advantages: list[float] = []
    reference_gaps: list[float] = []
    clipped_count = 0
    optimizer_steps = 0

    for step_index in range(max(1, int(max_steps))):
        sample = shuffled[step_index % len(shuffled)]
        batch = tokenize_prompt_and_completion(
            tokenizer,
            sample.prompt_messages,
            sample.completion_text,
            max_length=max_length,
        )
        tensor_batch = {
            key: value.to(model.device)
            for key, value in batch.items()
            if key in {"input_ids", "labels", "attention_mask"}
        }
        outputs = model(
            input_ids=tensor_batch["input_ids"],
            attention_mask=tensor_batch["attention_mask"],
            use_cache=False,
        )
        shifted_logits = outputs.logits[:, :-1, :]
        shifted_labels = tensor_batch["labels"][:, 1:]
        selected_log_probs, _mask = selected_token_logprobs(shifted_logits, shifted_labels)
        sequence_new_logprob = selected_log_probs.sum(dim=1)
        old_logprob_tensor = torch.tensor([sample.old_logprob], device=model.device, dtype=sequence_new_logprob.dtype)
        reference_logprob_tensor = torch.tensor(
            [sample.reference_logprob],
            device=model.device,
            dtype=sequence_new_logprob.dtype,
        )
        advantage_tensor = torch.tensor([sample.advantage], device=model.device, dtype=sequence_new_logprob.dtype)
        ratio = torch.exp(sequence_new_logprob - old_logprob_tensor)
        clipped_ratio = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon)
        objective = torch.minimum(ratio * advantage_tensor, clipped_ratio * advantage_tensor)
        reference_gap = sequence_new_logprob - reference_logprob_tensor
        kl_penalty = torch.square(reference_gap).mean()
        loss = -objective.mean() + float(kl_coef) * kl_penalty
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        optimizer_steps += 1
        ratio_value = float(ratio.detach().cpu().item())
        losses.append(float(loss.detach().cpu().item()))
        ratios.append(ratio_value)
        rewards.append(float(sample.reward))
        advantages.append(float(sample.advantage))
        reference_gaps.append(float(reference_gap.detach().cpu().item()))
        if ratio_value < (1.0 - clip_epsilon) or ratio_value > (1.0 + clip_epsilon):
            clipped_count += 1

    return {
        "optimizer_steps": optimizer_steps,
        "mean_loss": mean_or_zero(losses),
        "mean_ratio": mean_or_zero(ratios),
        "mean_reward": mean_or_zero(rewards),
        "mean_advantage": mean_or_zero(advantages),
        "mean_reference_gap": mean_or_zero(reference_gaps),
        "clip_fraction": (float(clipped_count) / float(optimizer_steps)) if optimizer_steps else 0.0,
        "sample_count": len(samples),
        "skipped": False,
    }


def save_adapter(*, model: Any, tokenizer: Any, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(destination)
    tokenizer.save_pretrained(destination)


def build_result_manifest(
    *,
    output_root: Path,
    bootstrap_summary: dict[str, Any],
    pivot_profile_summary: dict[str, Any],
    training_summary: dict[str, Any],
    final_eval_summary: dict[str, Any] | None,
) -> dict[str, Any]:
    final_score = (
        float(final_eval_summary.get("mean_outcome_reward", 0.0))
        if isinstance(final_eval_summary, dict)
        else None
    )
    return {
        "method_name": "pivotrl_preachievement_offline",
        "timestamp_utc": now_utc_iso(),
        "output_root": str(output_root),
        "offline_only_after_bootstrap": True,
        "bootstrap_requested_rollouts": int(bootstrap_summary.get("requested_rollouts", 0)),
        "bootstrap_successful_rollouts": int(bootstrap_summary.get("successful_rollouts", 0)),
        "pivot_profiled_count": int(pivot_profile_summary.get("profiled_count", 0)),
        "pivot_kept_count": int(pivot_profile_summary.get("kept_count", 0)),
        "pivot_dropped_count": int(pivot_profile_summary.get("dropped_count", 0)),
        "train_optimizer_steps": int(training_summary.get("optimizer_steps_total", 0)),
        "train_iterations_completed": int(training_summary.get("iterations_completed", 0)),
        "heldout_mean_reward": final_score,
        "track_reference_baseline_score": TRACK_BASELINE_SCORE,
        "delta_vs_track_reference_baseline": (
            float(final_score) - TRACK_BASELINE_SCORE if final_score is not None else None
        ),
    }


def resolve_container_url(args: argparse.Namespace) -> str:
    value = str(
        args.container_url
        or os.getenv("NANOHORIZON_CRAFTAX_CONTAINER_URL")
        or os.getenv("NANOHORIZON_CONTAINER_URL")
        or ""
    ).strip()
    if not value:
        raise RuntimeError("container URL is required via --container-url or NANOHORIZON_CRAFTAX_CONTAINER_URL")
    return value


def resolve_container_worker_token(args: argparse.Namespace) -> str:
    return str(
        args.container_worker_token
        or os.getenv("NANOHORIZON_CRAFTAX_CONTAINER_WORKER_TOKEN")
        or ""
    ).strip()


def wait_for_http_health(url: str, *, timeout_seconds: float = DEFAULT_REQUEST_TIMEOUT_S) -> dict[str, Any]:
    deadline = time.time() + timeout_seconds
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            with httpx.Client(timeout=10.0, follow_redirects=True) as client:
                response = client.get(url)
            if response.status_code == 200:
                if response.content:
                    payload = response.json()
                    if isinstance(payload, dict):
                        return payload
                return {"status": "ok"}
            last_error = RuntimeError(f"health returned HTTP {response.status_code}")
        except Exception as exc:
            last_error = exc
        time.sleep(1.0)
    raise RuntimeError(f"timed out waiting for {url}: {last_error!r}")


@contextmanager
def local_craftax_runtime(*, log_path: str | Path | None = None) -> Iterator[str]:
    log_file = None
    process: subprocess.Popen[str] | None = None
    try:
        if log_path:
            target = Path(log_path).expanduser().resolve()
            target.parent.mkdir(parents=True, exist_ok=True)
            log_file = target.open("w", encoding="utf-8", buffering=1)
        process = subprocess.Popen(
            [sys.executable, "-m", "nanohorizon.craftax_core.http_shim"],
            env={
                **os.environ,
                "NANOHORIZON_CRAFTAX_BIND_HOST": "0.0.0.0",
                "NANOHORIZON_CRAFTAX_BIND_PORT": str(CRAFTAX_PORT),
            },
            stdout=log_file or sys.stdout,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            start_new_session=True,
        )
        base_url = f"http://127.0.0.1:{CRAFTAX_PORT}"
        wait_for_http_health(f"{base_url}/health", timeout_seconds=60.0)
        yield base_url
    finally:
        if process is not None:
            with suppress(ProcessLookupError, OSError):
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            with suppress(subprocess.TimeoutExpired):
                process.wait(timeout=10)
        if log_file is not None:
            log_file.close()


def evaluate_adapter(
    *,
    base_model: str,
    adapter_dir: str | Path,
    container_url: str,
    output_dir: str | Path,
    seed_start: int,
    num_rollouts: int,
    max_steps: int,
    max_concurrent_rollouts: int,
    max_length: int,
    max_new_tokens: int,
    thinking_budget_tokens: int,
    enable_thinking: bool,
    enforce_eager: bool,
    summary_name: str,
) -> dict[str, Any]:
    out_dir = ensure_dir(output_dir)
    release_cuda_memory()
    config = LocalVLLMConfig(
        model=base_model,
        served_model_name=base_model,
        lora_name="policy-lora",
        lora_path=str(Path(adapter_dir).expanduser().resolve()),
        max_lora_rank=infer_lora_rank(adapter_dir),
        max_model_len=max_length,
        max_new_tokens=max_new_tokens,
        enable_thinking=enable_thinking,
        enforce_eager=enforce_eager,
    )
    seeds = [int(seed_start) + idx for idx in range(max(1, int(num_rollouts)))]
    system_prompt = (
        "You are a Craftax policy.\n"
        f"You may think for up to about {int(thinking_budget_tokens)} tokens before answering.\n"
        "Return a short useful macro-action with 5-10 valid Craftax actions.\n"
        "Use the provided `craftax_interact` tool exactly once for the final answer.\n"
        "Do not return plain text actions or JSON."
    )
    with local_vllm_server(config=config, log_path=out_dir / f"{summary_name.removesuffix('.json')}_vllm.log") as server:
        rollouts, collection_summary = asyncio.run(
            collect_rollouts_concurrently_with_summary(
                container_url=container_url,
                inference_url=f"{str(server['base_url']).rstrip('/')}/chat/completions",
                model=base_model,
                api_key="",
                seeds=seeds,
                max_steps=max_steps,
                system_prompt=system_prompt,
                temperature=0.0,
                max_tokens=max_new_tokens,
                max_model_len=max_length,
                enable_thinking=enable_thinking,
                thinking_budget_tokens=thinking_budget_tokens,
                policy_version="pivotrl-eval",
                target_action_batch_size=8,
                min_action_batch_size=5,
                request_timeout_seconds=DEFAULT_REQUEST_TIMEOUT_S,
                max_concurrent_rollouts=max_concurrent_rollouts,
                trace_prefix=summary_name.removesuffix(".json"),
                rollout_concurrency=max_concurrent_rollouts,
                rollout_semaphore_limit=max_concurrent_rollouts,
            )
        )
    valid_rollouts = [item for item in rollouts if isinstance(item, dict) and not item.get("error") and is_rollout_payload(item)]
    rewards = [rollout_outcome_reward(item) for item in valid_rollouts]
    summary = {
        "requested_num_eval_rollouts": len(rollouts),
        "num_eval_rollouts": len(valid_rollouts),
        "num_rollout_errors": len(rollouts) - len(valid_rollouts),
        "mean_outcome_reward": mean_or_zero(rewards),
        "max_outcome_reward": max(rewards) if rewards else 0.0,
        "collection": collection_summary,
        "details": [
            {
                "seed": int(item.get("_request_seed") or 0),
                "trace_correlation_id": str(item.get("trace_correlation_id") or ""),
                "outcome_reward": rollout_outcome_reward(item),
                "achievements": rollout_achievements(item),
                "success_status": item.get("success_status"),
                "error": item.get("error"),
            }
            for item in rollouts
        ],
    }
    write_json(Path(out_dir) / summary_name, summary)
    write_jsonl(Path(out_dir) / f"{summary_name.removesuffix('.json')}_rollouts.jsonl", [item for item in rollouts if isinstance(item, dict)])
    return summary


def run_bootstrap(args: argparse.Namespace, *, output_root: Path) -> tuple[Path, dict[str, Any]]:
    print("[pivotrl] bootstrap phase starting", flush=True)
    artifacts_dir = ensure_dir(output_root / "artifacts")
    logs_dir = ensure_dir(output_root / "logs")
    rollouts_path = artifacts_dir / "bootstrap_rollouts.jsonl"
    successful_path = artifacts_dir / "bootstrap_successful_rollouts.jsonl"
    summary_path = artifacts_dir / "bootstrap_summary.json"

    container_url = resolve_container_url(args)
    container_worker_token = resolve_container_worker_token(args)
    seeds = [int(args.bootstrap_seed_start) + offset for offset in range(max(1, int(args.bootstrap_seed_count)))]
    teacher_model = str(args.teacher_model or DEFAULT_TEACHER_MODEL)
    teacher_api_key = str(args.teacher_api_key or os.getenv("NANOHORIZON_TEACHER_API_KEY") or "").strip()
    teacher_inference_url = normalize_inference_url(
        str(
            args.teacher_inference_url
            or os.getenv("NANOHORIZON_TEACHER_INFERENCE_URL")
            or os.getenv("NANOHORIZON_TEACHER_BASE_URL")
            or ""
        )
    )
    system_prompt = rollout_system_prompt(
        thinking_budget_tokens=int(args.bootstrap_thinking_budget_tokens),
        target_action_batch_size=int(args.bootstrap_target_action_batch_size),
        min_action_batch_size=int(args.bootstrap_min_action_batch_size),
    )

    async def _collect(inference_url: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        return await collect_rollouts_concurrently_with_summary(
            container_url=container_url,
            container_worker_token=container_worker_token,
            inference_url=inference_url,
            model=teacher_model,
            api_key=teacher_api_key,
            seeds=seeds,
            max_steps=int(args.bootstrap_max_steps),
            system_prompt=system_prompt,
            temperature=float(args.bootstrap_temperature),
            max_tokens=int(args.bootstrap_max_new_tokens),
            max_model_len=int(args.max_length),
            enable_thinking=bool(args.enable_thinking),
            thinking_budget_tokens=int(args.bootstrap_thinking_budget_tokens),
            policy_version="pivotrl-teacher-bootstrap",
            target_action_batch_size=int(args.bootstrap_target_action_batch_size),
            min_action_batch_size=int(args.bootstrap_min_action_batch_size),
            request_timeout_seconds=float(args.request_timeout_seconds),
            max_concurrent_rollouts=int(args.bootstrap_rollout_concurrency),
            trace_prefix="pivotrl_bootstrap",
            rollout_concurrency=int(args.bootstrap_rollout_concurrency),
            rollout_semaphore_limit=int(args.bootstrap_rollout_semaphore_limit),
        )

    if teacher_inference_url:
        rollouts, collection_summary = asyncio.run(_collect(teacher_inference_url))
    else:
        config = LocalVLLMConfig(
            model=teacher_model,
            served_model_name=teacher_model,
            max_model_len=int(args.max_length),
            max_new_tokens=int(args.bootstrap_max_new_tokens),
            enable_thinking=bool(args.enable_thinking),
            enforce_eager=bool(args.enforce_eager),
        )
        with local_vllm_server(config=config, log_path=logs_dir / "bootstrap_teacher_vllm.log") as server:
            rollouts, collection_summary = asyncio.run(_collect(f"{str(server['base_url']).rstrip('/')}/chat/completions"))

    successful_rollouts = [
        rollout
        for rollout in rollouts
        if isinstance(rollout, dict) and not rollout.get("error") and is_rollout_payload(rollout)
    ]
    write_jsonl(rollouts_path, [row for row in rollouts if isinstance(row, dict)])
    write_jsonl(successful_path, successful_rollouts)
    rewards = [float(rollout_outcome_reward(rollout)) for rollout in successful_rollouts]
    bootstrap_summary = {
        **collection_summary,
        "requested_rollouts": len(seeds),
        "successful_rollouts": len(successful_rollouts),
        "mean_successful_reward": mean_or_zero(rewards),
        "max_successful_reward": max(rewards) if rewards else 0.0,
        "teacher_model": teacher_model,
        "teacher_inference_url": teacher_inference_url,
        "container_url": container_url,
        "achievements": sorted(
            {
                achievement
                for rollout in successful_rollouts
                for achievement in rollout_achievements(rollout)
            }
        ),
        "artifacts": {
            "bootstrap_rollouts": str(rollouts_path),
            "bootstrap_successful_rollouts": str(successful_path),
        },
    }
    dataset_source_path = successful_path if successful_rollouts else rollouts_path
    bootstrap_summary["dataset_source_path"] = str(dataset_source_path)
    write_json(summary_path, bootstrap_summary)
    return dataset_source_path, bootstrap_summary


def run_build_dataset(
    args: argparse.Namespace,
    *,
    output_root: Path,
    bootstrap_rollouts_path: Path | None = None,
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    print("[pivotrl] build-dataset phase starting", flush=True)
    artifacts_dir = ensure_dir(output_root / "artifacts")
    candidate_path = bootstrap_rollouts_path or (
        Path(args.bootstrap_rollouts_path).expanduser().resolve()
        if str(args.bootstrap_rollouts_path).strip()
        else artifacts_dir / "bootstrap_successful_rollouts.jsonl"
    )
    if not candidate_path.exists():
        raise FileNotFoundError(f"bootstrap rollouts not found: {candidate_path}")
    rollouts = read_jsonl(candidate_path)
    candidates, candidate_summary = build_candidate_pivots(
        rollouts=rollouts,
        lookback=max(1, int(args.lookback)),
    )
    profiled, profile_summary = profile_pivots(
        pivots=candidates,
        base_model=str(args.base_model or DEFAULT_BASE_MODEL),
        profile_k=int(args.profile_k),
        lambda_diff=float(args.lambda_diff),
        max_length=int(args.max_length),
        max_new_tokens=int(args.profile_max_new_tokens),
        temperature=float(args.profile_temperature),
        top_p=float(args.profile_top_p),
        max_pivots=int(args.max_pivots),
    )
    dataset_path = artifacts_dir / "pivot_dataset.jsonl"
    profile_summary_path = artifacts_dir / "pivot_profile_summary.json"
    write_jsonl(dataset_path, profiled)
    write_json(
        profile_summary_path,
        {
            **candidate_summary,
            **profile_summary,
            "bootstrap_rollouts_path": str(candidate_path),
        },
    )
    return dataset_path, candidate_summary, profile_summary


def run_train(
    args: argparse.Namespace,
    *,
    output_root: Path,
    dataset_path: Path | None = None,
) -> tuple[Path, dict[str, Any]]:
    print("[pivotrl] train phase starting", flush=True)
    artifacts_dir = ensure_dir(output_root / "artifacts")
    resolved_dataset_path = dataset_path or (
        Path(args.dataset_path).expanduser().resolve()
        if str(args.dataset_path).strip()
        else artifacts_dir / "pivot_dataset.jsonl"
    )
    if not resolved_dataset_path.exists():
        raise FileNotFoundError(f"pivot dataset not found: {resolved_dataset_path}")
    pivots = read_jsonl(resolved_dataset_path)
    if not pivots:
        raise RuntimeError(f"pivot dataset is empty: {resolved_dataset_path}")

    tokenizer, model, optimizer = initialize_policy(
        base_model=str(args.base_model or DEFAULT_BASE_MODEL),
        lora_rank=int(args.lora_rank),
        learning_rate=float(args.learning_rate),
    )
    train_pivots = list(pivots)
    if int(args.max_pivots) > 0:
        train_pivots = train_pivots[: int(args.max_pivots)]
    iteration_summaries: list[dict[str, Any]] = []
    remaining_steps = max(0, int(args.max_train_steps))

    for iteration_index in range(max(1, int(args.train_iterations))):
        if remaining_steps <= 0:
            break
        if int(args.pivots_per_iteration) > 0 and len(train_pivots) > int(args.pivots_per_iteration):
            selected_pivots = random.sample(train_pivots, k=int(args.pivots_per_iteration))
        else:
            selected_pivots = list(train_pivots)
        samples, sampling_summary = build_group_samples(
            tokenizer=tokenizer,
            model=model,
            pivots=selected_pivots,
            group_size=int(args.group_size),
            max_length=int(args.max_length),
            max_new_tokens=int(args.sample_max_new_tokens),
            temperature=float(args.sample_temperature),
            top_p=float(args.sample_top_p),
        )
        steps_this_iteration = min(remaining_steps, max(1, int(args.train_steps_per_iteration)))
        iteration_summary = train_iteration(
            tokenizer=tokenizer,
            model=model,
            optimizer=optimizer,
            samples=samples,
            clip_epsilon=float(args.clip_epsilon),
            kl_coef=float(args.kl_coef),
            max_length=int(args.max_length),
            max_steps=steps_this_iteration,
        )
        iteration_summaries.append(
            {
                "iteration_index": iteration_index,
                "sampling": sampling_summary,
                "training": iteration_summary,
            }
        )
        remaining_steps -= int(iteration_summary.get("optimizer_steps", 0))

    adapter_dir = artifacts_dir / "pivotrl_adapter"
    save_adapter(model=model, tokenizer=tokenizer, destination=adapter_dir)
    release_cuda_memory()
    training_summary = {
        "method_name": "pivotrl_preachievement_offline",
        "dataset_path": str(resolved_dataset_path),
        "pivot_count": len(train_pivots),
        "group_size": int(args.group_size),
        "iterations_completed": len(iteration_summaries),
        "optimizer_steps_total": sum(
            int(item["training"].get("optimizer_steps", 0))
            for item in iteration_summaries
        ),
        "mean_iteration_reward": mean_or_zero(
            [
                float(item["sampling"].get("mean_group_reward", 0.0))
                for item in iteration_summaries
            ]
        ),
        "mean_iteration_loss": mean_or_zero(
            [
                float(item["training"].get("mean_loss", 0.0))
                for item in iteration_summaries
                if not item["training"].get("skipped")
            ]
        ),
        "offline_only_after_bootstrap": True,
        "adapter_dir": str(adapter_dir),
        "iterations": iteration_summaries,
    }
    write_json(artifacts_dir / "training_summary.json", training_summary)
    return adapter_dir, training_summary


def run_eval(
    args: argparse.Namespace,
    *,
    output_root: Path,
    adapter_dir: Path | None = None,
) -> dict[str, Any]:
    print("[pivotrl] eval phase starting", flush=True)
    artifacts_dir = ensure_dir(output_root / "artifacts")
    if int(args.eval_rollouts) <= 0:
        summary = {
            "skipped": True,
            "reason": "eval_rollouts <= 0",
            "heldout_mean_reward": None,
        }
        write_json(artifacts_dir / "final_eval_summary.json", summary)
        return summary
    resolved_adapter_dir = adapter_dir or (
        Path(args.adapter_dir).expanduser().resolve()
        if str(args.adapter_dir).strip()
        else artifacts_dir / "pivotrl_adapter"
    )
    if not resolved_adapter_dir.exists():
        raise FileNotFoundError(f"adapter directory not found: {resolved_adapter_dir}")
    container_url = resolve_container_url(args)
    summary = evaluate_adapter(
        base_model=str(args.base_model or DEFAULT_BASE_MODEL),
        adapter_dir=resolved_adapter_dir,
        container_url=container_url,
        output_dir=artifacts_dir,
        seed_start=int(args.eval_seed_start),
        num_rollouts=int(args.eval_rollouts),
        max_steps=int(args.eval_max_steps),
        max_concurrent_rollouts=int(args.eval_concurrency),
        max_length=int(args.max_length),
        max_new_tokens=int(args.eval_max_new_tokens),
        thinking_budget_tokens=int(args.eval_thinking_budget_tokens),
        enable_thinking=bool(args.enable_thinking),
        enforce_eager=bool(args.enforce_eager),
        summary_name="final_eval_summary.json",
    )
    return summary


def run_end_to_end(args: argparse.Namespace) -> dict[str, Any]:
    print("[pivotrl] end-to-end run starting", flush=True)
    output_root = resolve_output_root(args.output_root)
    ensure_dir(output_root / "artifacts")
    bootstrap_rollouts_path: Path
    bootstrap_summary: dict[str, Any]
    if str(args.bootstrap_rollouts_path).strip():
        bootstrap_rollouts_path = Path(args.bootstrap_rollouts_path).expanduser().resolve()
        bootstrap_summary = {
            "requested_rollouts": 0,
            "successful_rollouts": len(read_jsonl(bootstrap_rollouts_path)),
            "bootstrap_rollouts_path": str(bootstrap_rollouts_path),
            "skipped_live_bootstrap": True,
        }
        write_json(output_root / "artifacts" / "bootstrap_summary.json", bootstrap_summary)
    else:
        bootstrap_rollouts_path, bootstrap_summary = run_bootstrap(args, output_root=output_root)
    dataset_path, _candidate_summary, profile_summary = run_build_dataset(
        args,
        output_root=output_root,
        bootstrap_rollouts_path=bootstrap_rollouts_path,
    )
    adapter_dir, training_summary = run_train(
        args,
        output_root=output_root,
        dataset_path=dataset_path,
    )
    final_eval_summary = run_eval(
        args,
        output_root=output_root,
        adapter_dir=adapter_dir,
    )
    result_manifest = build_result_manifest(
        output_root=output_root,
        bootstrap_summary=bootstrap_summary,
        pivot_profile_summary=profile_summary,
        training_summary=training_summary,
        final_eval_summary=final_eval_summary,
    )
    write_json(output_root / "artifacts" / "result_manifest.json", result_manifest)
    return result_manifest


def _default_modal_output_dir() -> str:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{ARTIFACT_DIR}/pivotrl/{stamp}"


@app.function(
    image=image,
    gpu=DEFAULT_MODAL_GPU,
    timeout=60 * 60 * 4,
    volumes={
        HF_CACHE_DIR: HF_CACHE_VOLUME,
        VLLM_CACHE_DIR: VLLM_CACHE_VOLUME,
        TRITON_CACHE_DIR: TRITON_CACHE_VOLUME,
        ARTIFACT_DIR: ARTIFACT_VOLUME,
    },
)
def run_modal_job(payload: dict[str, Any]) -> dict[str, Any]:
    os.chdir(REMOTE_ROOT)
    output_root = str(payload.get("output_root") or _default_modal_output_dir())
    payload = {**payload, "output_root": output_root}
    args = argparse.Namespace(**payload)
    logs_dir = ensure_dir(Path(output_root) / "logs")
    container_url = str(payload.get("container_url") or "").strip()
    if container_url:
        return run_end_to_end(args)
    with local_craftax_runtime(log_path=logs_dir / "craftax_runtime.log") as local_container_url:
        setattr(args, "container_url", local_container_url)
        return run_end_to_end(args)


@app.local_entrypoint()
def modal_main(
    output_root: str = "",
    base_model: str = DEFAULT_BASE_MODEL,
    teacher_model: str = DEFAULT_TEACHER_MODEL,
    max_length: int = 8192,
    bootstrap_seed_start: int = 0,
    bootstrap_seed_count: int = 32,
    bootstrap_max_steps: int = 48,
    bootstrap_rollout_concurrency: int = 4,
    bootstrap_rollout_semaphore_limit: int = 4,
    bootstrap_max_new_tokens: int = 3072,
    bootstrap_thinking_budget_tokens: int = 2000,
    profile_k: int = 4,
    lambda_diff: float = 0.75,
    max_pivots: int = 128,
    group_size: int = 4,
    pivots_per_iteration: int = 16,
    train_iterations: int = 2,
    train_steps_per_iteration: int = 8,
    max_train_steps: int = 16,
    eval_rollouts: int = 8,
    eval_concurrency: int = 4,
    container_url: str = "",
    container_worker_token: str = "",
    teacher_inference_url: str = "",
    teacher_api_key: str = "",
    eval_max_new_tokens: int = 2048,
    eval_thinking_budget_tokens: int = 2000,
    enable_thinking: bool = True,
    enforce_eager: bool = False,
) -> None:
    payload = {
        "command": "run",
        "output_root": output_root or _default_modal_output_dir(),
        "base_model": base_model,
        "teacher_model": teacher_model,
        "max_length": max_length,
        "bootstrap_seed_start": bootstrap_seed_start,
        "bootstrap_seed_count": bootstrap_seed_count,
        "bootstrap_max_steps": bootstrap_max_steps,
        "bootstrap_rollout_concurrency": bootstrap_rollout_concurrency,
        "bootstrap_rollout_semaphore_limit": bootstrap_rollout_semaphore_limit,
        "bootstrap_max_new_tokens": bootstrap_max_new_tokens,
        "bootstrap_thinking_budget_tokens": bootstrap_thinking_budget_tokens,
        "profile_k": profile_k,
        "lambda_diff": lambda_diff,
        "max_pivots": max_pivots,
        "group_size": group_size,
        "pivots_per_iteration": pivots_per_iteration,
        "train_iterations": train_iterations,
        "train_steps_per_iteration": train_steps_per_iteration,
        "max_train_steps": max_train_steps,
        "eval_rollouts": eval_rollouts,
        "eval_concurrency": eval_concurrency,
        "container_url": container_url,
        "container_worker_token": container_worker_token,
        "teacher_inference_url": teacher_inference_url,
        "teacher_api_key": teacher_api_key,
        "eval_max_new_tokens": eval_max_new_tokens,
        "eval_thinking_budget_tokens": eval_thinking_budget_tokens,
        "enable_thinking": enable_thinking,
        "enforce_eager": enforce_eager,
        "request_timeout_seconds": DEFAULT_REQUEST_TIMEOUT_S,
        "bootstrap_rollouts_path": "",
        "bootstrap_temperature": 0.2,
        "bootstrap_target_action_batch_size": 4,
        "bootstrap_min_action_batch_size": 3,
        "lookback": 1,
        "profile_temperature": 0.8,
        "profile_top_p": 0.95,
        "profile_max_new_tokens": 96,
        "dataset_path": "",
        "sample_max_new_tokens": 96,
        "sample_temperature": 0.8,
        "sample_top_p": 0.95,
        "learning_rate": 1e-5,
        "lora_rank": 16,
        "clip_epsilon": 0.2,
        "kl_coef": 0.02,
        "adapter_dir": "",
        "eval_seed_start": 10000,
        "eval_max_steps": 48,
    }
    result = run_modal_job.remote(payload)
    print(json.dumps(result, indent=2, sort_keys=True))


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output-root", default="")
    parser.add_argument("--base-model", default=os.getenv("NANOHORIZON_PIVOTRL_BASE_MODEL", DEFAULT_BASE_MODEL))
    parser.add_argument("--teacher-model", default=os.getenv("NANOHORIZON_PIVOTRL_TEACHER_MODEL", DEFAULT_TEACHER_MODEL))
    parser.add_argument("--teacher-inference-url", default="")
    parser.add_argument("--teacher-api-key", default="")
    parser.add_argument("--container-url", default="")
    parser.add_argument("--container-worker-token", default="")
    parser.add_argument("--request-timeout-seconds", type=float, default=300.0)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--enforce-eager", action="store_true")

    parser.add_argument("--bootstrap-rollouts-path", default="")
    parser.add_argument("--bootstrap-seed-start", type=int, default=0)
    parser.add_argument("--bootstrap-seed-count", type=int, default=8)
    parser.add_argument("--bootstrap-max-steps", type=int, default=48)
    parser.add_argument("--bootstrap-temperature", type=float, default=0.2)
    parser.add_argument("--bootstrap-max-new-tokens", type=int, default=3072)
    parser.add_argument("--bootstrap-thinking-budget-tokens", type=int, default=2000)
    parser.add_argument("--bootstrap-rollout-concurrency", type=int, default=4)
    parser.add_argument("--bootstrap-rollout-semaphore-limit", type=int, default=4)
    parser.add_argument("--bootstrap-target-action-batch-size", type=int, default=4)
    parser.add_argument("--bootstrap-min-action-batch-size", type=int, default=3)

    parser.add_argument("--lookback", type=int, default=1)
    parser.add_argument("--profile-k", type=int, default=4)
    parser.add_argument("--lambda-diff", type=float, default=0.75)
    parser.add_argument("--profile-temperature", type=float, default=0.8)
    parser.add_argument("--profile-top-p", type=float, default=0.95)
    parser.add_argument("--profile-max-new-tokens", type=int, default=96)
    parser.add_argument("--max-pivots", type=int, default=256)
    parser.add_argument("--dataset-path", default="")

    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--train-iterations", type=int, default=3)
    parser.add_argument("--pivots-per-iteration", type=int, default=32)
    parser.add_argument("--max-train-steps", type=int, default=48)
    parser.add_argument("--train-steps-per-iteration", type=int, default=16)
    parser.add_argument("--sample-max-new-tokens", type=int, default=96)
    parser.add_argument("--sample-temperature", type=float, default=0.8)
    parser.add_argument("--sample-top-p", type=float, default=0.95)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--clip-epsilon", type=float, default=0.2)
    parser.add_argument("--kl-coef", type=float, default=0.02)
    parser.add_argument("--adapter-dir", default="")

    parser.add_argument("--eval-seed-start", type=int, default=10_000)
    parser.add_argument("--eval-rollouts", type=int, default=8)
    parser.add_argument("--eval-max-steps", type=int, default=48)
    parser.add_argument("--eval-concurrency", type=int, default=4)
    parser.add_argument("--eval-max-new-tokens", type=int, default=2048)
    parser.add_argument("--eval-thinking-budget-tokens", type=int, default=2000)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Offline PivotRL submission for NanoHorizon Craftax.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    bootstrap_parser = subparsers.add_parser("bootstrap", help="Collect teacher rollouts from the live Craftax env.")
    add_common_args(bootstrap_parser)

    dataset_parser = subparsers.add_parser("build-dataset", help="Build and profile the offline pivot dataset.")
    add_common_args(dataset_parser)

    train_parser = subparsers.add_parser("train", help="Run offline PivotRL/GRPO training.")
    add_common_args(train_parser)

    eval_parser = subparsers.add_parser("eval", help="Run held-out live evaluation for the trained adapter.")
    add_common_args(eval_parser)

    run_parser = subparsers.add_parser("run", help="Bootstrap, build dataset, train, and evaluate.")
    add_common_args(run_parser)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    output_root = resolve_output_root(args.output_root)

    if args.command == "bootstrap":
        run_bootstrap(args, output_root=output_root)
        return
    if args.command == "build-dataset":
        run_build_dataset(args, output_root=output_root)
        return
    if args.command == "train":
        run_train(args, output_root=output_root)
        return
    if args.command == "eval":
        run_eval(args, output_root=output_root)
        return
    if args.command == "run":
        result_manifest = run_end_to_end(args)
        print(json.dumps(result_manifest, indent=2, sort_keys=True))
        return
    raise RuntimeError(f"unsupported command: {args.command}")


if __name__ == "__main__":
    main()
