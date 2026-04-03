from __future__ import annotations

import argparse
import asyncio
import io
import json
import math
import os
import platform
import random
import shlex
import shutil
import socket
import subprocess
import sys
import tarfile
import time
import uuid
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from statistics import mean
from typing import Any, cast

import httpx
import modal
import yaml
from modal import experimental as modal_exp

REMOTE_SRC = Path("/root/nanohorizon/src")
if REMOTE_SRC.exists():
    sys.path.insert(0, str(REMOTE_SRC))

from nanohorizon.craftax_core.metadata import DEFAULT_ACTION_NAMES, PRIMARY_TOOL_NAME
from nanohorizon.shared.craftax_data import summarize_achievement_frequencies
from nanohorizon.shared.modal_common import ARTIFACT_DIR, ARTIFACT_VOLUME, RECORDS_DIR, RECORDS_VOLUME


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


CRAFTAX_ACTION_ENUM = list(DEFAULT_ACTION_NAMES)

CRAFTAX_INTERACT_TOOL = {
    "type": "function",
    "function": {
        "name": PRIMARY_TOOL_NAME,
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
            raise ValueError(
                "container_worker_token is required for SynthTunnel container_url. "
                "Pass the worker token emitted by the tunnel helper."
            )
        headers["Authorization"] = f"Bearer {worker_token}"
        return headers
    env_key = (environment_api_key or "").strip()
    if env_key:
        headers["x-api-key"] = env_key
    return headers


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
    success_status = rollout.get("success_status")
    return (
        isinstance(reward_info, dict)
        and isinstance(trace, dict)
        and isinstance(success_status, str)
        and bool(str(success_status).strip())
    )


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
    enable_thinking: bool = False,
    thinking_budget_tokens: int = 0,
    policy_version: str = "bootstrap",
    target_action_batch_size: int = 8,
    min_action_batch_size: int = 5,
    timeout_s: int = 45,
) -> dict[str, Any]:
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
                "max_tokens": int(max_tokens),
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
    worker_count = max(
        1,
        int(rollout_concurrency if rollout_concurrency is not None else max_concurrent_rollouts),
    )
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
                print(
                    f"[rollout] START seed={seed} index={index} "
                    f"active={active_rollouts}/{permit_limit} "
                    f"started={requests_started} finished={requests_finished}",
                    flush=True,
                )
                try:
                    results[index] = await _run_one(client, seed, index)
                finally:
                    request_latencies_s.append(time.perf_counter() - request_started_at)
                    requests_finished += 1
                    active_rollouts -= 1
                    elapsed = request_latencies_s[-1]
                    print(
                        f"[rollout] DONE  seed={seed} index={index} "
                        f"elapsed={elapsed:.1f}s active={active_rollouts}/{permit_limit} "
                        f"finished={requests_finished}/{len(seeds)}",
                        flush=True,
                    )
            rollout_queue.task_done()

    client_kwargs: dict[str, Any] = {"timeout": timeout}
    if is_quick_tunnel:
        client_kwargs.update(
            {
                "limits": httpx.Limits(
                    max_connections=max(worker_count, permit_limit),
                    max_keepalive_connections=0,
                ),
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
    valid_rollouts = [
        item for item in completed_rollouts if not item.get("error") and is_rollout_payload(item)
    ]
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


def build_lora_bundle(adapter_dir: Path) -> tuple[bytes, list[str]]:
    resolved_dir = Path(adapter_dir).expanduser().resolve()
    if not resolved_dir.is_dir():
        raise FileNotFoundError(f"adapter_dir does not exist: {resolved_dir}")
    file_paths = sorted(path for path in resolved_dir.rglob("*") if path.is_file())
    if not file_paths:
        raise FileNotFoundError(f"adapter_dir contains no files: {resolved_dir}")
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for path in file_paths:
            archive.add(str(path), arcname=str(path.relative_to(resolved_dir)))
    return buffer.getvalue(), [str(path.relative_to(resolved_dir)) for path in file_paths]


DEFAULT_TARGET_MODULES = [
    # Attention
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    # MLP
    "gate_proj",
    "up_proj",
    "down_proj",
    # Gated Delta Net (Qwen3.5 hybrid recurrent layers)
    "in_proj_qkv",
    "in_proj_z",
    "in_proj_b",
    "in_proj_a",
    "out_proj",
]


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

DEFAULT_SYSTEM_PROMPT = (
    "You are a Craftax RL policy.\n"
    f"Use the provided `{PRIMARY_TOOL_NAME}` tool exactly once for the final answer.\n"
    "Return a short useful macro-action with 5-10 valid full-Craftax actions.\n"
    "Use movement to explore when nothing useful is adjacent.\n"
    "Use 'do' only when facing a useful nearby object or resource.\n"
    "Read the recent action history and avoid repeating unproductive loops.\n"
    "Do not return plain text actions or JSON.\n"
    "After reasoning, your final assistant action must be a tool call."
)


@dataclass
class TurnSample:
    group_id: str
    seed: int
    rollout_id: str
    trace_correlation_id: str
    turn_index: int
    prompt_messages: list[dict[str, Any]]
    full_messages: list[dict[str, Any]]
    old_logprob: float
    advantage: float
    outcome_reward: float
    decision_reward: float
    policy_version: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NanoHorizon Craftax RLVR training")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", default="")
    parser.add_argument(
        "--container-url",
        default=os.getenv("NANOHORIZON_RLVR_CONTAINER_URL", "direct://local"),
    )
    parser.add_argument("--inference-url", default=os.getenv("NANOHORIZON_RLVR_INFERENCE_URL", ""))
    parser.add_argument("--inference-admin-url", default=os.getenv("NANOHORIZON_RLVR_INFERENCE_ADMIN_URL", ""))
    parser.add_argument("--inference-api-key", default=os.getenv("NANOHORIZON_RLVR_INFERENCE_API_KEY", ""))
    parser.add_argument("--request-model", default=os.getenv("NANOHORIZON_RLVR_REQUEST_MODEL", ""))
    return parser.parse_args()


def _normalize_inference_url(raw_url: str) -> str:
    value = str(raw_url or "").strip().rstrip("/")
    if not value:
        return ""
    if value.endswith("/chat/completions"):
        return value
    if value.endswith("/v1"):
        return f"{value}/chat/completions"
    return f"{value}/v1/chat/completions"


def _normalize_admin_url(raw_url: str) -> str:
    value = str(raw_url or "").strip().rstrip("/")
    if not value:
        return ""
    if value.endswith("/v1"):
        return value
    return f"{value}/v1"


def _normalize_tool_calls(tool_calls: Any) -> list[dict[str, Any]]:
    if not isinstance(tool_calls, list):
        return []
    normalized: list[dict[str, Any]] = []
    for index, item in enumerate(tool_calls):
        if not isinstance(item, dict):
            continue
        item_dict = cast(dict[str, Any], item)
        raw_function = item_dict.get("function")
        function = raw_function if isinstance(raw_function, dict) else {}
        if not isinstance(function, dict):
            continue
        name = str(function.get("name") or "").strip()
        if not name:
            continue
        arguments = function.get("arguments", {})
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except Exception:
                arguments = arguments
        normalized.append(
            {
                "id": str(item_dict.get("id") or f"call_{index}"),
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": arguments,
                },
            }
        )
    return normalized


def _normalize_messages_for_chat_template(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for item in messages:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role") or "user").strip() or "user"
        entry: dict[str, Any] = {
            "role": role,
            "content": item.get("content") if item.get("content") is not None else "",
        }
        tool_calls = _normalize_tool_calls(item.get("tool_calls"))
        if tool_calls:
            entry["tool_calls"] = tool_calls
            if role == "assistant" and not isinstance(entry["content"], str):
                entry["content"] = ""
        reasoning_content = item.get("reasoning_content")
        if reasoning_content is not None:
            entry["reasoning_content"] = str(reasoning_content)
        normalized.append(entry)
    return normalized


def _render_messages(tokenizer: Any, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> str:
    safe_messages = _normalize_messages_for_chat_template(messages)
    if hasattr(tokenizer, "apply_chat_template"):
        rendered = tokenizer.apply_chat_template(
            safe_messages,
            tools=tools or None,
            tokenize=False,
            add_generation_prompt=False,
        )
        if isinstance(rendered, str):
            return rendered
    rendered_lines: list[str] = []
    for item in safe_messages:
        rendered_lines.append(f"<|{item.get('role', 'user')}|>\n{item.get('content', '')}")
    return "\n".join(rendered_lines)


def _tokenize_messages_with_assistant_mask(
    tokenizer: Any,
    prompt_messages: list[dict[str, Any]],
    full_messages: list[dict[str, Any]],
    *,
    max_length: int,
) -> dict[str, Any]:
    import torch

    normalized_prompt_messages = _normalize_messages_for_chat_template(prompt_messages)
    normalized_full_messages = _normalize_messages_for_chat_template(full_messages)
    prompt_text = tokenizer.apply_chat_template(
        normalized_prompt_messages,
        tools=[CRAFTAX_INTERACT_TOOL],
        tokenize=False,
        add_generation_prompt=True,
    )
    full_text = _render_messages(tokenizer, normalized_full_messages, [CRAFTAX_INTERACT_TOOL])
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
    if len(full_ids) < len(prompt_ids):
        raise ValueError("full chat template render shorter than prompt render")
    input_ids = full_ids[:max_length]
    labels = ([-100] * len(prompt_ids) + full_ids[len(prompt_ids) :])[:max_length]
    attention_mask = [1] * len(input_ids)
    return {
        "input_ids": torch.tensor([input_ids], dtype=torch.long),
        "labels": torch.tensor([labels], dtype=torch.long),
        "attention_mask": torch.tensor([attention_mask], dtype=torch.long),
    }


def _selected_token_logprobs(logits: Any, shifted_labels: Any) -> tuple[Any, Any]:
    import torch

    mask = shifted_labels != -100
    safe_targets = torch.clamp(shifted_labels, min=0)
    selected_logits = logits.gather(dim=-1, index=safe_targets.unsqueeze(-1)).squeeze(-1)
    lse = torch.logsumexp(logits, dim=-1)
    return (selected_logits - lse) * mask, mask


def _assistant_message_from_turn(turn: dict[str, Any], *, trace_correlation_id: str, rollout_id: str) -> dict[str, Any] | None:
    raw_actions = turn.get("actions")
    if not isinstance(raw_actions, list):
        return None
    actions = [str(item).strip() for item in raw_actions if str(item).strip()]
    if not actions:
        return None
    turn_index = int(turn.get("turn_index") or 0)
    return {
        "role": "assistant",
        "content": "",
        "reasoning_content": str(turn.get("reasoning_text") or turn.get("assistant_text") or "").strip(),
        "tool_calls": [
            {
                "id": f"call_{trace_correlation_id or rollout_id}_{turn_index}",
                "type": "function",
                "function": {
                    "name": PRIMARY_TOOL_NAME,
                    "arguments": {
                        "actions_list": actions,
                    },
                },
            }
        ],
    }


def _load_seed_manifest(path: Path) -> tuple[list[int], list[int]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"seed manifest must decode to an object: {path}")
    train_seeds = [int(item) for item in payload.get("train_seeds") or []]
    eval_seeds = [int(item) for item in payload.get("eval_seeds") or []]
    if not train_seeds:
        raise ValueError(f"seed manifest missing train_seeds: {path}")
    if not eval_seeds:
        raise ValueError(f"seed manifest missing eval_seeds: {path}")
    return train_seeds, eval_seeds


def _iteration_train_seeds(*, all_train_seeds: list[int], iteration_index: int, groups_per_iteration: int) -> list[int]:
    if not all_train_seeds:
        return []
    size = max(1, int(groups_per_iteration))
    start = (iteration_index * size) % len(all_train_seeds)
    seeds: list[int] = []
    for offset in range(size):
        seeds.append(int(all_train_seeds[(start + offset) % len(all_train_seeds)]))
    return seeds


def _build_rollout_seed_schedule(*, seeds: list[int], group_size: int) -> list[int]:
    schedule: list[int] = []
    for seed in seeds:
        schedule.extend([int(seed)] * max(1, int(group_size)))
    return schedule


def _group_advantages(rewards: list[float]) -> list[float]:
    if not rewards:
        return []
    mean_reward = sum(rewards) / len(rewards)
    variance = sum((reward - mean_reward) ** 2 for reward in rewards) / len(rewards)
    std = math.sqrt(max(variance, 1e-12))
    if std <= 1e-6:
        return [float(reward - mean_reward) for reward in rewards]
    return [float((reward - mean_reward) / std) for reward in rewards]


def _build_turn_samples(*, rollouts: list[dict[str, Any]], group_size: int) -> tuple[list[TurnSample], dict[str, Any]]:
    samples: list[TurnSample] = []
    valid_rollouts = [
        rollout
        for rollout in rollouts
        if isinstance(rollout, dict) and not rollout.get("error") and is_rollout_payload(rollout)
    ]
    grouped_rewards: list[float] = []
    group_count = 0
    for group_start in range(0, len(rollouts), max(1, int(group_size))):
        group_rollouts = rollouts[group_start : group_start + max(1, int(group_size))]
        if len(group_rollouts) < max(1, int(group_size)):
            continue
        valid_group = [
            rollout
            for rollout in group_rollouts
            if isinstance(rollout, dict) and not rollout.get("error") and is_rollout_payload(rollout)
        ]
        if len(valid_group) != max(1, int(group_size)):
            continue
        rewards = [float(rollout_outcome_reward(rollout)) for rollout in valid_group]
        advantages = _group_advantages(rewards)
        group_count += 1
        grouped_rewards.extend(rewards)
        for rollout, advantage, reward in zip(valid_group, advantages, rewards, strict=False):
            trace_correlation_id = str(rollout.get("trace_correlation_id") or "")
            rollout_id = str(rollout.get("rollout_id") or "")
            policy_version = str(rollout.get("policy_version") or "bootstrap")
            seed = int(rollout.get("_request_seed") or 0)
            for turn in rollout_turns(rollout):
                if not bool(turn.get("trainable", True)):
                    continue
                old_logprob_raw = turn.get("behavior_sequence_logprob")
                if old_logprob_raw is None:
                    continue
                try:
                    old_logprob = float(old_logprob_raw)
                except (TypeError, ValueError):
                    continue
                prompt_messages = turn.get("prompt_messages")
                if not isinstance(prompt_messages, list) or not prompt_messages:
                    continue
                assistant_message = _assistant_message_from_turn(
                    turn,
                    trace_correlation_id=trace_correlation_id,
                    rollout_id=rollout_id,
                )
                if assistant_message is None:
                    continue
                full_messages = [*prompt_messages, assistant_message]
                samples.append(
                    TurnSample(
                        group_id=f"group_{group_count:05d}",
                        seed=seed,
                        rollout_id=rollout_id,
                        trace_correlation_id=trace_correlation_id,
                        turn_index=int(turn.get("turn_index") or 0),
                        prompt_messages=[item for item in prompt_messages if isinstance(item, dict)],
                        full_messages=[item for item in full_messages if isinstance(item, dict)],
                        old_logprob=old_logprob,
                        advantage=float(advantage),
                        outcome_reward=float(reward),
                        decision_reward=float(turn.get("decision_reward") or 0.0),
                        policy_version=policy_version,
                    )
                )
    summary = {
        "valid_rollouts": len(valid_rollouts),
        "group_count": group_count,
        "trainable_turns": len(samples),
        "mean_outcome_reward": mean(grouped_rewards) if grouped_rewards else 0.0,
        "max_outcome_reward": max(grouped_rewards) if grouped_rewards else 0.0,
    }
    return samples, summary


def _evaluate_policy(
    *,
    container_url: str,
    inference_url: str,
    inference_api_key: str,
    request_model: str,
    seeds: list[int],
    max_steps: int,
    max_concurrent_rollouts: int,
    request_timeout_seconds: float,
    max_tokens: int,
    thinking_budget_tokens: int,
    enable_thinking: bool,
    output_dir: Path,
    label: str,
    policy_version: str,
) -> dict[str, Any]:
    rollouts, collection_summary = asyncio_run(
        collect_rollouts_concurrently_with_summary(
            container_url=container_url,
            inference_url=inference_url,
            model=request_model,
            api_key=inference_api_key,
            seeds=seeds,
            max_steps=max_steps,
            system_prompt=DEFAULT_SYSTEM_PROMPT,
            temperature=0.0,
            max_tokens=max_tokens,
            enable_thinking=enable_thinking,
            thinking_budget_tokens=thinking_budget_tokens,
            policy_version=policy_version,
            target_action_batch_size=8,
            min_action_batch_size=5,
            request_timeout_seconds=request_timeout_seconds,
            max_concurrent_rollouts=max_concurrent_rollouts,
            trace_prefix=label,
            rollout_concurrency=max_concurrent_rollouts,
            rollout_semaphore_limit=max_concurrent_rollouts,
        )
    )
    valid = [
        rollout
        for rollout in rollouts
        if isinstance(rollout, dict) and not rollout.get("error") and is_rollout_payload(rollout)
    ]
    requested_rollout_count = len(rollouts)
    rewards = [float(rollout_outcome_reward(item)) for item in valid]
    achievement_frequencies = summarize_achievement_frequencies(
        rollouts,
        denominator=requested_rollout_count,
    )
    summary = {
        "label": label,
        "policy_version": policy_version,
        "seed_count": len(seeds),
        "requested_num_eval_rollouts": requested_rollout_count,
        "num_eval_rollouts": len(valid),
        "num_rollout_errors": len(rollouts) - len(valid),
        "mean_outcome_reward": mean(rewards) if rewards else 0.0,
        "mean_outcome_reward_over_requested_rollouts": (
            sum(rewards) / float(requested_rollout_count)
        ) if requested_rollout_count else 0.0,
        "max_outcome_reward": max(rewards) if rewards else 0.0,
        "achievement_frequencies": achievement_frequencies,
        "collection": collection_summary,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "summary.json", summary)
    with (output_dir / "rollouts.jsonl").open("w", encoding="utf-8") as handle:
        for rollout in rollouts:
            handle.write(json.dumps(rollout, sort_keys=True) + "\n")
    return summary


def _reload_adapter(
    *,
    inference_admin_url: str,
    inference_api_key: str,
    adapter_dir: str,
    adapter_name: str,
    policy_version: str,
    external_inference_server: Any | None = None,
) -> dict[str, Any]:
    if external_inference_server is not None:
        bundle_bytes, bundle_files = build_lora_bundle(Path(adapter_dir))
        response = external_inference_server.load_lora_bundle.remote(
            lora_name=adapter_name,
            policy_version=policy_version,
            bundle_bytes=bundle_bytes,
        )
        payload: dict[str, Any]
        if isinstance(response, dict):
            payload = dict(cast(dict[str, Any], response))
        else:
            payload = {"status": "ok"}
        payload["bundle_file_count"] = len(bundle_files)
        payload["attempt"] = 1
        return payload
    headers = {"Content-Type": "application/json"}
    if inference_api_key:
        headers["Authorization"] = f"Bearer {inference_api_key}"
    payload = {
        "lora_name": adapter_name,
        "lora_path": adapter_dir,
        "policy_version": policy_version,
        "load_inplace": True,
    }
    deadline = time.time() + 120.0
    last_error: str = ""
    attempt = 0
    with httpx.Client(timeout=120.0) as client:
        while time.time() < deadline:
            attempt += 1
            response = client.post(
                f"{inference_admin_url.rstrip('/')}/load_lora_adapter",
                json=payload,
                headers=headers,
            )
            if response.status_code < 400:
                if response.content:
                    body = response.json()
                    if isinstance(body, dict):
                        body["attempt"] = attempt
                        return body
                    return {"attempt": attempt, "body": body}
                return {"status": "ok", "attempt": attempt, "empty_body": True}
            response_text = (response.text or "").strip()
            last_error = f"HTTP {response.status_code}: {response_text[:1000]}"
            retryable = response.status_code in {404, 409, 425, 429, 500, 502, 503, 504}
            adapter_missing = "No adapter found" in response_text or "adapter_config.json" in response_text
            if retryable or adapter_missing:
                time.sleep(2.0)
                continue
            response.raise_for_status()
    raise RuntimeError(
        f"timed out loading adapter {adapter_name!r} for policy {policy_version!r} after {attempt} attempts: "
        f"{last_error}"
    )


def _commit_output_volume(path: Path) -> None:
    normalized = str(path)
    try:
        if normalized == RECORDS_DIR or normalized.startswith(f"{RECORDS_DIR}/"):
            RECORDS_VOLUME.commit()
        elif normalized == ARTIFACT_DIR or normalized.startswith(f"{ARTIFACT_DIR}/"):
            ARTIFACT_VOLUME.commit()
    except Exception as exc:
        raise RuntimeError(f"failed to commit output volume for {path}: {type(exc).__name__}: {exc}") from exc


def _reset_to_base(
    *,
    inference_admin_url: str,
    inference_api_key: str,
) -> dict[str, Any]:
    return {
        "status": "ok",
        "policy_version": "bootstrap",
        "skipped": True,
        "detail": "fresh vLLM instance starts on the base model",
        "inference_admin_url": inference_admin_url,
    }


def _initialize_model_and_optimizer(*, base_model: str, lora_rank: int, learning_rate: float) -> tuple[Any, Any, Any]:
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = _load_text_only_causal_lm(base_model=base_model, device=device, use_cache=False)
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_rank * 2,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=DEFAULT_TARGET_MODULES,
    )
    model = get_peft_model(model, lora_config)
    with suppress(Exception):
        model.gradient_checkpointing_enable()
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    return tokenizer, model, optimizer


def _train_iteration(
    *,
    tokenizer: Any,
    model: Any,
    optimizer: Any,
    samples: list[TurnSample],
    clip_epsilon: float,
    max_length: int,
    max_steps: int,
) -> dict[str, Any]:
    import torch

    if not samples:
        return {
            "optimizer_steps": 0,
            "mean_loss": 0.0,
            "mean_old_logprob": 0.0,
            "mean_new_logprob": 0.0,
            "mean_ratio": 0.0,
            "clip_fraction": 0.0,
            "num_samples": 0,
            "skipped": True,
        }

    shuffled = list(samples)
    random.shuffle(shuffled)
    step_limit = max(1, int(max_steps))
    losses: list[float] = []
    old_logprobs: list[float] = []
    new_logprobs: list[float] = []
    ratios: list[float] = []
    clipped_count = 0
    optimizer_steps = 0

    for step_index in range(step_limit):
        sample = shuffled[step_index % len(shuffled)]
        batch = _tokenize_messages_with_assistant_mask(
            tokenizer,
            sample.prompt_messages,
            sample.full_messages,
            max_length=max_length,
        )
        batch = {key: value.to(model.device) for key, value in batch.items()}
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            use_cache=False,
        )
        shifted_logits = outputs.logits[:, :-1, :]
        shifted_labels = batch["labels"][:, 1:]
        selected_logprobs, mask = _selected_token_logprobs(shifted_logits, shifted_labels)
        sequence_new_logprob = selected_logprobs.sum(dim=1)
        del mask
        old_logprob_tensor = torch.tensor([sample.old_logprob], device=model.device, dtype=sequence_new_logprob.dtype)
        advantage_tensor = torch.tensor([sample.advantage], device=model.device, dtype=sequence_new_logprob.dtype)
        ratio = torch.exp(sequence_new_logprob - old_logprob_tensor)
        clipped_ratio = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon)
        objective = torch.minimum(ratio * advantage_tensor, clipped_ratio * advantage_tensor)
        loss = -objective.mean()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        optimizer_steps += 1
        loss_value = float(loss.detach().cpu().item())
        new_logprob_value = float(sequence_new_logprob.detach().cpu().item())
        ratio_value = float(ratio.detach().cpu().item())
        losses.append(loss_value)
        old_logprobs.append(float(sample.old_logprob))
        new_logprobs.append(new_logprob_value)
        ratios.append(ratio_value)
        if ratio_value < (1.0 - clip_epsilon) or ratio_value > (1.0 + clip_epsilon):
            clipped_count += 1

    return {
        "optimizer_steps": optimizer_steps,
        "mean_loss": mean(losses) if losses else 0.0,
        "mean_old_logprob": mean(old_logprobs) if old_logprobs else 0.0,
        "mean_new_logprob": mean(new_logprobs) if new_logprobs else 0.0,
        "mean_ratio": mean(ratios) if ratios else 0.0,
        "clip_fraction": (float(clipped_count) / float(optimizer_steps)) if optimizer_steps else 0.0,
        "num_samples": len(samples),
        "skipped": False,
    }


def _save_adapter(*, model: Any, tokenizer: Any, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(destination)
    tokenizer.save_pretrained(destination)
    _commit_output_volume(destination)


def _copy_config_bundle(*, config_path: Path, output_dir: Path) -> None:
    shutil.copy2(config_path, output_dir / "run_config.yaml")


def _build_metadata(*, config: dict[str, Any], config_path: Path, output_dir: Path) -> dict[str, Any]:
    return {
        "track": str(config.get("task", {}).get("track") or "rlvr_20min_2xa100_40gb"),
        "task": str(config.get("task", {}).get("name") or "craftax"),
        "baseline": "modal_craftax_grpo",
        "model": str(config.get("model", {}).get("model") or ""),
        "config_path": str(config_path),
        "output_dir": str(output_dir),
        "user_editable_training_script": "src/nanohorizon/baselines/rlvr.py",
        "runner_script": "scripts/run_craftax_rlvr_qwen35_4b_2xa100_20min.sh",
    }


def run_training(
    *,
    config_path: str | Path,
    output_dir: str | Path,
    container_url: str,
    inference_url: str,
    inference_admin_url: str,
    inference_api_key: str,
    request_model: str,
    external_inference_server: Any | None = None,
    bootstrap_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    config_path = Path(config_path).expanduser().resolve()
    config = load_config(config_path)
    config_dir = config_path.parent
    out_dir = ensure_dir(output_dir)
    timer = Timer()
    random.seed(int(config.get("training", {}).get("seed", 0) or 0))

    _copy_config_bundle(config_path=config_path, output_dir=out_dir)
    metadata = _build_metadata(config=config, config_path=config_path, output_dir=out_dir)
    write_json(out_dir / "metadata.json", metadata)
    write_json(out_dir / "bootstrap_info.json", bootstrap_info or {})
    _commit_output_volume(out_dir)

    seed_manifest_path = resolve_path(config["data"]["seed_manifest_json"], base_dir=config_dir)
    train_seeds, eval_seeds = _load_seed_manifest(seed_manifest_path)
    rollout_cfg = config.get("rollout", {})
    training_cfg = config.get("training", {})
    evaluation_cfg = config.get("evaluation", {})
    budget_minutes = float(config.get("budget", {}).get("wall_clock_minutes", 20))
    max_iterations = max(1, int(rollout_cfg.get("num_iterations", 4)))
    group_size = max(1, int(rollout_cfg.get("group_size", 4)))
    groups_per_iteration = max(1, int(rollout_cfg.get("groups_per_iteration", 2)))
    rollout_max_steps = max(1, int(rollout_cfg.get("max_steps", 48)))
    rollout_concurrency = max(1, int(rollout_cfg.get("rollout_concurrency", 8)))
    rollout_semaphore_limit = max(1, int(rollout_cfg.get("rollout_semaphore_limit", 4)))
    request_timeout_seconds = float(rollout_cfg.get("request_timeout_seconds", 300.0))
    max_tokens = max(int(rollout_cfg.get("max_tokens", 3072)), 2048)
    thinking_budget_tokens = int(rollout_cfg.get("thinking_budget_tokens", 2000))
    enable_thinking = bool(rollout_cfg.get("enable_thinking", True))
    rollout_temperature = float(rollout_cfg.get("temperature", 0.8))
    target_action_batch_size = int(rollout_cfg.get("target_action_batch_size", 4))
    min_action_batch_size = int(rollout_cfg.get("min_action_batch_size", 3))
    eval_concurrency = max(1, int(evaluation_cfg.get("max_concurrent_rollouts", 4)))
    eval_max_steps = max(1, int(evaluation_cfg.get("max_steps", rollout_max_steps)))
    eval_max_tokens = max(int(evaluation_cfg.get("max_tokens", max_tokens)), 2048)
    eval_rollouts_per_seed = max(1, int(evaluation_cfg.get("rollouts_per_seed", 1)))
    clip_epsilon = float(training_cfg.get("clip_epsilon", 0.2))
    train_max_length = int(training_cfg.get("max_length", 4096))
    train_steps_per_iteration = int(training_cfg.get("max_steps_per_iteration", training_cfg.get("max_steps", 8)))

    _reset_to_base(
        inference_admin_url=_normalize_admin_url(inference_admin_url),
        inference_api_key=inference_api_key,
    )

    tokenizer, model, optimizer = _initialize_model_and_optimizer(
        base_model=str(config["model"]["model"]),
        lora_rank=int(training_cfg.get("lora_rank", 16)),
        learning_rate=float(training_cfg.get("learning_rate", 1.0e-5)),
    )

    periodic_eval_root = out_dir / "periodic_eval"
    adapters_root = out_dir / "adapters"
    iterations_root = out_dir / "iterations"
    policy_history: list[dict[str, Any]] = []
    iteration_summaries: list[dict[str, Any]] = []

    eval_seed_schedule = [seed for seed in eval_seeds for _ in range(eval_rollouts_per_seed)]
    step0_summary = _evaluate_policy(
        container_url=container_url,
        inference_url=_normalize_inference_url(inference_url),
        inference_api_key=inference_api_key,
        request_model=request_model,
        seeds=eval_seed_schedule,
        max_steps=eval_max_steps,
        max_concurrent_rollouts=eval_concurrency,
        request_timeout_seconds=request_timeout_seconds,
        max_tokens=eval_max_tokens,
        thinking_budget_tokens=thinking_budget_tokens,
        enable_thinking=enable_thinking,
        output_dir=periodic_eval_root / "step_000",
        label="periodic_eval_step_000",
        policy_version="bootstrap",
    )

    for iteration_index in range(max_iterations):
        if timer.elapsed_minutes >= budget_minutes:
            break

        iteration_dir = ensure_dir(iterations_root / f"iter_{iteration_index:03d}")
        seeds_for_iteration = _iteration_train_seeds(
            all_train_seeds=train_seeds,
            iteration_index=iteration_index,
            groups_per_iteration=groups_per_iteration,
        )
        rollout_seed_schedule = _build_rollout_seed_schedule(seeds=seeds_for_iteration, group_size=group_size)
        rollouts, rollout_summary = asyncio_run(
            collect_rollouts_concurrently_with_summary(
                container_url=container_url,
                inference_url=_normalize_inference_url(inference_url),
                model=request_model,
                api_key=inference_api_key,
                seeds=rollout_seed_schedule,
                max_steps=rollout_max_steps,
                system_prompt=DEFAULT_SYSTEM_PROMPT,
                temperature=rollout_temperature,
                max_tokens=max_tokens,
                enable_thinking=enable_thinking,
                thinking_budget_tokens=thinking_budget_tokens,
                policy_version=policy_history[-1]["policy_version"] if policy_history else "bootstrap",
                target_action_batch_size=target_action_batch_size,
                min_action_batch_size=min_action_batch_size,
                request_timeout_seconds=request_timeout_seconds,
                max_concurrent_rollouts=rollout_concurrency,
                trace_prefix=f"rlvr_iter_{iteration_index:03d}",
                rollout_concurrency=rollout_concurrency,
                rollout_semaphore_limit=rollout_semaphore_limit,
            )
        )
        with (Path(iteration_dir) / "rollouts.jsonl").open("w", encoding="utf-8") as handle:
            for rollout in rollouts:
                handle.write(json.dumps(rollout, sort_keys=True) + "\n")

        turn_samples, sample_summary = _build_turn_samples(rollouts=rollouts, group_size=group_size)
        train_summary = _train_iteration(
            tokenizer=tokenizer,
            model=model,
            optimizer=optimizer,
            samples=turn_samples,
            clip_epsilon=clip_epsilon,
            max_length=train_max_length,
            max_steps=train_steps_per_iteration,
        )
        adapter_name = f"policy-iter-{iteration_index:03d}"
        policy_version = f"iter_{iteration_index:03d}"
        adapter_dir = adapters_root / f"iter_{iteration_index:03d}"
        _save_adapter(model=model, tokenizer=tokenizer, destination=adapter_dir)
        reload_payload: dict[str, Any]
        try:
            reload_payload = _reload_adapter(
                inference_admin_url=_normalize_admin_url(inference_admin_url),
                inference_api_key=inference_api_key,
                adapter_dir=str(adapter_dir),
                adapter_name=adapter_name,
                policy_version=policy_version,
                external_inference_server=external_inference_server,
            )
        except Exception as exc:
            failure_payload = {
                "rollout": rollout_summary,
                "sample_summary": sample_summary,
                "train": train_summary,
                "adapter_name": adapter_name,
                "adapter_dir": str(adapter_dir),
                "policy_version": policy_version,
                "reload_error": f"{type(exc).__name__}: {exc}",
            }
            write_json(Path(iteration_dir) / "summary.json", failure_payload)
            _commit_output_volume(Path(iteration_dir))
            raise
        summary_payload = {
            "rollout": rollout_summary,
            "sample_summary": sample_summary,
            "train": train_summary,
            "adapter_name": adapter_name,
            "adapter_dir": str(adapter_dir),
            "policy_version": policy_version,
            "reload": reload_payload,
        }
        write_json(Path(iteration_dir) / "summary.json", summary_payload)
        _commit_output_volume(Path(iteration_dir))
        policy_history.append(
            {
                "iteration_index": iteration_index,
                "policy_version": policy_version,
                "adapter_name": adapter_name,
                "adapter_dir": str(adapter_dir),
                "mean_outcome_reward": float(sample_summary["mean_outcome_reward"]),
            }
        )
        iteration_summaries.append(summary_payload)

        periodic_summary = _evaluate_policy(
            container_url=container_url,
            inference_url=_normalize_inference_url(inference_url),
            inference_api_key=inference_api_key,
            request_model=request_model,
            seeds=eval_seed_schedule,
            max_steps=eval_max_steps,
            max_concurrent_rollouts=eval_concurrency,
            request_timeout_seconds=request_timeout_seconds,
            max_tokens=eval_max_tokens,
            thinking_budget_tokens=thinking_budget_tokens,
            enable_thinking=enable_thinking,
            output_dir=periodic_eval_root / f"step_{iteration_index + 1:03d}",
            label=f"periodic_eval_step_{iteration_index + 1:03d}",
            policy_version=policy_version,
        )
        summary_payload["periodic_eval"] = periodic_summary
        write_json(Path(iteration_dir) / "summary.json", summary_payload)

        if timer.elapsed_minutes >= budget_minutes:
            break

    final_policy_version = policy_history[-1]["policy_version"] if policy_history else "bootstrap"
    final_eval_summary = _evaluate_policy(
        container_url=container_url,
        inference_url=_normalize_inference_url(inference_url),
        inference_api_key=inference_api_key,
        request_model=request_model,
        seeds=eval_seed_schedule,
        max_steps=eval_max_steps,
        max_concurrent_rollouts=eval_concurrency,
        request_timeout_seconds=request_timeout_seconds,
        max_tokens=eval_max_tokens,
        thinking_budget_tokens=thinking_budget_tokens,
        enable_thinking=enable_thinking,
        output_dir=out_dir / "final_eval",
        label="final_eval",
        policy_version=final_policy_version,
    )

    metrics = {
        "track": str(config.get("task", {}).get("track") or "rlvr_20min_2xa100_40gb"),
        "baseline": "modal_craftax_grpo",
        "model": str(config.get("model", {}).get("model") or ""),
        "started_at": timer.started_at,
        "ended_at": timer.ended_at,
        "elapsed_minutes": timer.elapsed_minutes,
        "budget_minutes": budget_minutes,
        "iterations_completed": len(policy_history),
        "submission_mean_outcome_reward": float(
            final_eval_summary.get(
                "mean_outcome_reward_over_requested_rollouts",
                final_eval_summary["mean_outcome_reward"],
            )
        ),
        "submission_achievement_frequencies": final_eval_summary.get("achievement_frequencies", {}),
        "step0_mean_outcome_reward": float(step0_summary["mean_outcome_reward"]),
        "step0_achievement_frequencies": step0_summary.get("achievement_frequencies", {}),
        "final_mean_outcome_reward": float(final_eval_summary["mean_outcome_reward"]),
        "final_achievement_frequencies": final_eval_summary.get("achievement_frequencies", {}),
        "reward_delta_from_bootstrap": float(final_eval_summary["mean_outcome_reward"]) - float(step0_summary["mean_outcome_reward"]),
        "final_policy_version": final_policy_version,
        "periodic_eval_count": len(list(periodic_eval_root.glob("step_*/summary.json"))),
    }
    write_json(out_dir / "metrics.json", metrics)
    write_json(out_dir / "system_info.json", system_info())
    write_json(out_dir / "policy_history.json", policy_history)
    write_json(out_dir / "iteration_summaries.json", iteration_summaries)
    write_json(out_dir / "final_eval_summary.json", final_eval_summary)
    write_json(
        out_dir / "run_timing.json",
        {
            "started_at": timer.started_at,
            "ended_at": timer.ended_at,
            "elapsed_minutes": timer.elapsed_minutes,
            "budget_minutes": budget_minutes,
        },
    )
    write_text(
        out_dir / "command.txt",
        f"NANOHORIZON_RLVR_OUTPUT_DIR={out_dir} ./scripts/run_craftax_rlvr_qwen35_4b_2xa100_20min.sh\n",
    )

    release_cuda_memory()
    return {
        "output_dir": str(out_dir),
        "metrics": metrics,
        "final_eval_summary": final_eval_summary,
        "policy_history": policy_history,
    }


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    output_dir = args.output_dir or config["output"]["root_dir"]
    result = run_training(
        config_path=args.config,
        output_dir=output_dir,
        container_url=str(args.container_url),
        inference_url=str(args.inference_url),
        inference_admin_url=str(args.inference_admin_url),
        inference_api_key=str(args.inference_api_key),
        request_model=str(args.request_model or config.get("model", {}).get("served_model_name") or config["model"]["model"]),
        bootstrap_info={},
    )
    print(json.dumps(result, indent=2, sort_keys=True))


def asyncio_run(coro: Any) -> Any:
    import asyncio

    return asyncio.run(coro)


if __name__ == "__main__":
    main()


# Modal entrypoint collapsed from former modal_rlvr.py
from nanohorizon.custom_vllm.runtime import (
    build_thinking_budget_request_overrides,
    enable_thinking_budget_support,
)
from nanohorizon.shared.lora_bundle import extract_lora_bundle
from nanohorizon.shared.modal_common import (
    GPU_RLVR,
    OFFLINE_VENV_ROOT,
    PROJECT_ROOT,
    RECORDS_DIR,
    RECORDS_VOLUME,
    REMOTE_ROOT,
    TRITON_CACHE_DIR,
    VLLM_COMPILE_CACHE_DIR,
    offline_image,
    volume_mounts,
)
APP_NAME = "nanohorizon-craftax-rlvr"
CRAFTAX_PORT = 8903
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
DEFAULT_INFERENCE_API_KEY = (
    os.getenv("NANOHORIZON_RLVR_INFERENCE_API_KEY", "nanohorizon-rlvr-key").strip()
    or "nanohorizon-rlvr-key"
)

app = modal.App(APP_NAME)


def _default_output_dir() -> str:
    stamp = now_utc_iso().replace(":", "").replace("+00:00", "Z")
    return f"{RECORDS_DIR}/rlvr_20min_2xa100_40gb/{stamp}_reference_baseline"


def _safe_output_label(raw_value: str) -> str:
    label = "".join(ch.lower() if ch.isalnum() else "-" for ch in str(raw_value or "").strip())
    collapsed = "-".join(part for part in label.split("-") if part)
    return collapsed or "run"


def _modal_cluster_output_dir(requested_output_dir: str) -> str:
    requested = str(requested_output_dir or "").strip()
    stamp = now_utc_iso().replace(":", "").replace("+00:00", "Z")
    requested_name = Path(requested).name if requested else ""
    label = _safe_output_label(requested_name or "local-entrypoint")
    return f"{RECORDS_DIR}/rlvr_20min_2xa100_40gb/modal_runs/{stamp}_{label}"


def _default_local_preflight_failure_dir() -> Path:
    stamp = now_utc_iso().replace(":", "").replace("+00:00", "Z")
    return PROJECT_ROOT / "artifacts" / "rlvr_preflight_failures" / stamp


if REMOTE_SRC.exists():
    runtime_image = modal.Image.debian_slim(python_version="3.11")
else:
    runtime_image = offline_image()


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
class CraftaxService:
    @modal.web_server(port=CRAFTAX_PORT, startup_timeout=60 * 10)
    def serve(self) -> None:
        runtime_env = {
            **os.environ,
            "PYTHONUNBUFFERED": "1",
            "PYTHONPATH": _pythonpath_with_repo(),
            "NANOHORIZON_CRAFTAX_BIND_HOST": "0.0.0.0",
            "NANOHORIZON_CRAFTAX_BIND_PORT": str(CRAFTAX_PORT),
            "NANOHORIZON_CRAFTAX_UVICORN_WORKERS": "16",
        }
        cmd = [sys.executable, "-m", "nanohorizon.craftax_core.http_shim"]
        print("Launching Craftax service:", " ".join(shlex.quote(part) for part in cmd), flush=True)
        subprocess.Popen(cmd, env=runtime_env)


def _install_lora_patch_into_venv(vllm_bin: str) -> None:
    """Patch vLLM's column_parallel_linear.py in-place to bounds-check
    lora_a/lora_b list lengths in MergedColumnParallelLinearWithLoRA.set_lora.

    This fixes a vLLM 0.18 IndexError for Qwen3.5 hybrid Mamba-attention
    models where packed GDN projections produce fewer LoRA tensor slices
    than the module's n_slices during CUDA-graph dummy-LoRA profiling.
    """
    venv_bin = Path(vllm_bin).resolve().parent
    venv_root = venv_bin.parent
    candidates = sorted(venv_root.glob(
        "lib/python*/site-packages/vllm/lora/layers/column_parallel_linear.py"
    ))
    if not candidates:
        print("warning: could not locate vLLM column_parallel_linear.py; skipping LoRA patch", flush=True)
        return
    target = candidates[-1]
    original = target.read_text(encoding="utf-8")
    # The buggy pattern: iterates n_slices without checking list length
    old = "for i in range(self.n_slices):"
    new = "for i in range(min(self.n_slices, len(lora_a) if isinstance(lora_a, list) else self.n_slices, len(lora_b) if isinstance(lora_b, list) else self.n_slices)):"
    if old not in original:
        print(f"warning: LoRA patch target pattern not found in {target}; already patched or different version", flush=True)
        return
    patched = original.replace(old, new)
    target.write_text(patched, encoding="utf-8")
    print(f"Patched vLLM LoRA set_lora bounds check in {target}", flush=True)


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
    runtime_env["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
    runtime_env["TORCHINDUCTOR_CACHE_DIR"] = str(
        os.environ.get("TORCHINDUCTOR_CACHE_DIR") or VLLM_COMPILE_CACHE_DIR
    )
    runtime_env["TRITON_CACHE_DIR"] = str(os.environ.get("TRITON_CACHE_DIR") or TRITON_CACHE_DIR)
    runtime_env["VLLM_SERVER_DEV_MODE"] = "1"
    runtime_env["VLLM_USE_V1"] = "1"
    # Install a sitecustomize.py that patches vLLM's LoRA set_lora method
    # to bounds-check lora_a/lora_b list lengths.  This fixes a vLLM 0.18
    # IndexError for Qwen3.5 hybrid Mamba-attention models during
    # CUDA-graph profiling with LoRA.
    _install_lora_patch_into_venv(vllm_bin)
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
    ]
    cmd, runtime_env = enable_thinking_budget_support(cmd=cmd, env=runtime_env, model_ref=model)
    print("Launching clustered RLVR vLLM:", " ".join(shlex.quote(part) for part in cmd), flush=True)
    return subprocess.Popen(cmd, env=runtime_env, stderr=subprocess.STDOUT)


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
            # Try to capture any remaining stdout/stderr
            try:
                remaining, _ = process.communicate(timeout=5)
                if remaining:
                    print(f"[vllm-stderr] {remaining.decode(errors='replace')[-4000:]}", flush=True)
            except Exception:
                pass
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


def _start_local_craftax() -> subprocess.Popen[bytes]:
    """Start the Craftax Python shim on the controller (rank 0).

    By running Craftax on the same container as the controller, rollout
    requests go via localhost instead of Modal's web proxy, and Craftax
    can reach the inference worker via the cluster's internal network
    instead of a Modal tunnel — eliminating the request serialization
    bottleneck.
    """
    runtime_env = {
        **os.environ,
        "PYTHONUNBUFFERED": "1",
        "PYTHONPATH": _pythonpath_with_repo(),
        "NANOHORIZON_CRAFTAX_BIND_HOST": "0.0.0.0",
        "NANOHORIZON_CRAFTAX_BIND_PORT": str(CRAFTAX_PORT),
    }
    cmd = [sys.executable, "-m", "nanohorizon.craftax_core.http_shim"]
    print("Launching local Craftax on controller:", " ".join(shlex.quote(part) for part in cmd), flush=True)
    return subprocess.Popen(cmd, env=runtime_env)


def _wait_for_local_craftax(process: subprocess.Popen[bytes], timeout_s: float | None = None) -> None:
    """Wait for the local Craftax runtime to start accepting connections."""
    resolved_timeout_s = float(
        timeout_s
        if timeout_s is not None
        else os.getenv("NANOHORIZON_RLVR_CRAFTAX_STARTUP_TIMEOUT_SECONDS", "180")
    )
    deadline = time.time() + resolved_timeout_s
    while time.time() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"local Craftax runtime exited early with code {process.returncode}")
        try:
            with socket.create_connection(("127.0.0.1", CRAFTAX_PORT), timeout=1.0):
                print("local Craftax runtime is accepting connections on port", CRAFTAX_PORT, flush=True)
                return
        except (ConnectionRefusedError, OSError):
            time.sleep(0.5)
    raise RuntimeError(f"local Craftax runtime did not start within {resolved_timeout_s}s")


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

    fallback_container_url = str(payload.get("container_url") or "").rstrip("/")
    prefer_local_craftax = (
        str(os.getenv("NANOHORIZON_RLVR_USE_LOCAL_CRAFTAX", "0")).strip().lower()
        in {"1", "true", "yes", "on"}
    )
    craftax_process: subprocess.Popen[bytes] | None = None
    local_container_url = fallback_container_url

    if prefer_local_craftax:
        # This keeps the old localhost optimization available for experiments,
        # but we do not block RLVR by default on another cold Craftax startup.
        craftax_process = _start_local_craftax()
        try:
            _wait_for_local_craftax(craftax_process)
            local_container_url = f"http://127.0.0.1:{CRAFTAX_PORT}"
        except Exception as exc:
            if craftax_process.poll() is None:
                craftax_process.terminate()
            craftax_process = None
            if not fallback_container_url:
                raise
            print(
                "local Craftax startup timed out; falling back to Craftax service endpoint:",
                fallback_container_url,
                f"({type(exc).__name__}: {exc})",
                flush=True,
            )
    else:
        if not fallback_container_url:
            raise RuntimeError("Craftax service endpoint was not provided for RLVR controller")
        print(
            "RLVR controller using Craftax service endpoint by default:",
            fallback_container_url,
            flush=True,
        )

    ready_payload = _wait_for_cluster_signal(
        ready_path=control_paths["ready"],
        error_path=control_paths["error"],
    )
    if str(ready_payload.get("status")) != "ready":
        failure_payload = {
            "container_url": local_container_url,
            "output_dir": str(output_dir),
            "error": ready_payload,
        }
        failure_path = output_dir / "preflight_failure.json"
        write_json(failure_path, failure_payload)
        _volume_commit()
        if craftax_process is not None and craftax_process.poll() is None:
            craftax_process.terminate()
        raise RuntimeError(f"clustered inference worker failed during startup: {ready_payload}")

    # Use the internal cluster IP for controller-local probes/admin, but keep a
    # service-reachable inference URL for Craftax rollout requests that execute
    # inside a different container.
    internal_inference_base_url = str(
        ready_payload.get("internal_inference_base_url") or ready_payload["inference_base_url"]
    ).rstrip("/")
    tunnel_inference_base_url = str(ready_payload["inference_base_url"]).rstrip("/")
    rollout_inference_base_url = (
        internal_inference_base_url
        if local_container_url.startswith("http://127.0.0.1:")
        else tunnel_inference_base_url
    )
    served_model_name = str(payload["served_model_name"])
    inference_api_key = str(payload["inference_api_key"])

    print(
        f"clustered controller using local Craftax={local_container_url} "
        f"internal_inference={internal_inference_base_url} "
        f"tunnel_inference={tunnel_inference_base_url} "
        f"rollout_inference={rollout_inference_base_url}",
        flush=True,
    )

    bootstrap_info: dict[str, Any] = {
        "container_url": local_container_url,
        "inference_base_url": internal_inference_base_url,
        "tunnel_inference_base_url": tunnel_inference_base_url,
        "rollout_inference_base_url": rollout_inference_base_url,
        "inference_url": f"{rollout_inference_base_url}/v1/chat/completions",
        "inference_admin_url": f"{internal_inference_base_url}/v1",
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
        bootstrap_info["preflight"]["container_health"] = _wait_for_health(f"{local_container_url}/health")
        bootstrap_info["preflight"]["container_task_info"] = _wait_for_task_info(local_container_url)
        bootstrap_info["preflight"]["inference_internal_health"] = _wait_for_health(
            f"{internal_inference_base_url}/health"
        )
        bootstrap_info["preflight"]["inference_chat_probe"] = _probe_inference_chat(
            inference_base_url=internal_inference_base_url,
            api_key=inference_api_key,
            model=served_model_name,
        )
        bootstrap_info["preflight"]["container_roundtrip_probe"] = _probe_container_roundtrip(
            container_url=local_container_url,
            inference_url=f"{rollout_inference_base_url}/v1/chat/completions",
            api_key=inference_api_key,
            request_model=served_model_name,
        )
    except Exception as exc:
        failure_payload = {**bootstrap_info, "error": f"{type(exc).__name__}: {exc}"}
        failure_path = output_dir / "preflight_failure.json"
        write_json(failure_path, failure_payload)
        _volume_commit()
        craftax_process.terminate()
        raise RuntimeError(f"clustered RLVR preflight failed: {exc}") from exc

    external_inference_server = _ClusteredInferenceHandle(control_paths)
    try:
        return run_training(
            config_path=runtime_config_path,
            output_dir=str(output_dir),
            container_url=local_container_url,
            inference_url=f"{rollout_inference_base_url}/v1/chat/completions",
            inference_admin_url=f"{internal_inference_base_url}/v1",
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
        if craftax_process is not None and craftax_process.poll() is None:
            craftax_process.terminate()
            with suppress(subprocess.TimeoutExpired):
                craftax_process.wait(timeout=10)


@app.function(
    image=runtime_image,
    gpu=GPU_RLVR,
    timeout=60 * 60 * 4,
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
def modal_main(
    config: str = "configs/craftax_rlvr_qwen35_4b_2xa100_20min.yaml",
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
    model_path = Path(model_name).expanduser()
    if model_path.exists():
        raise RuntimeError(
            "RLVR Modal inference requires `model.model` to be a Hugging Face model id or another "
            "remote-accessible model reference. Local filesystem checkpoints are not automatically "
            f"uploaded into the clustered Modal runtime: {model_name!r}"
        )
    inference_api_key = str(
        os.getenv("NANOHORIZON_RLVR_INFERENCE_API_KEY")
        or config_payload.get("inference", {}).get("api_key")
        or DEFAULT_INFERENCE_API_KEY
    ).strip() or DEFAULT_INFERENCE_API_KEY
    requested_output_dir = str(output_dir or "").strip()
    remote_output_dir = _modal_cluster_output_dir(requested_output_dir)
    craftax_service = CraftaxService()
    craftax_web_url = craftax_service.serve.get_web_url()
    if not craftax_web_url:
        raise RuntimeError("Craftax service did not provide a web URL")
    container_url = craftax_web_url.rstrip("/")
    payload = {
        "config": config,
        "config_text": submitted_config_path.read_text(encoding="utf-8"),
        "config_filename": submitted_config_path.name,
        "output_dir": remote_output_dir,
        "requested_output_dir": requested_output_dir,
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
            "output_dir": requested_output_dir or remote_output_dir,
            "remote_output_dir": remote_output_dir,
            "error": f"{type(exc).__name__}: {exc}",
        }
        failure_path = _write_local_preflight_failure(
            output_dir=requested_output_dir,
            payload=failure_payload,
        )
        raise RuntimeError(
            f"clustered RLVR run failed; details written to {failure_path}: {exc}"
        ) from exc
    if requested_output_dir:
        local_destination = ensure_dir(requested_output_dir)
        write_json(
            local_destination / "modal_rlvr_result.json",
            {
                "requested_output_dir": requested_output_dir,
                "remote_output_dir": remote_output_dir,
                "result": result,
            },
        )
    print(json.dumps(result, indent=2, sort_keys=True))
