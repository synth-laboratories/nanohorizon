from __future__ import annotations

import asyncio
import json
import math
import os
import random
import re
import signal
import subprocess
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager, nullcontext, suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean
from typing import Any, TypedDict

import httpx

from nanohorizon.craftax_core.metadata import DEFAULT_ACTION_NAMES, PRIMARY_TOOL_NAME
from nanohorizon.shared.common import ensure_dir, now_utc_iso, read_jsonl, write_json
from nanohorizon.shared.craftax_data import (
    CRAFTAX_INTERACT_TOOL,
    collect_rollouts_concurrently_with_summary,
    flatten_messages,
    is_rollout_payload,
    rollout_achievements,
    rollout_outcome_reward,
    rollout_turns,
)
from nanohorizon.shared.modal_common import ARTIFACT_DIR, OFFLINE_VENV_ROOT, REMOTE_ROOT

CRAFTAX_PORT = 8903
DEFAULT_REQUEST_TIMEOUT_S = 60 * 20
DEFAULT_BASE_MODEL = "Qwen/Qwen3.5-4B"
DEFAULT_TEACHER_MODEL = "Qwen/Qwen3.5-9B"

ACTION_SET = set(DEFAULT_ACTION_NAMES)
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
    "wood_pickaxe",
    "stone_pickaxe",
)
RUBRIC_TARGETS = {
    "collect_wood",
    "place_table",
    "make_wood_pickaxe",
    "collect_stone",
    "make_stone_pickaxe",
}


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


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def default_modal_output_dir() -> str:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{ARTIFACT_DIR}/pivotrl/{stamp}"


def normalize_inference_url(raw_url: str) -> str:
    value = str(raw_url or "").strip().rstrip("/")
    if not value:
        return ""
    if value.endswith("/chat/completions"):
        return value
    if value.endswith("/v1"):
        return f"{value}/chat/completions"
    return f"{value}/v1/chat/completions"


def _normalize_inference_base_url(raw_url: str) -> str:
    value = str(raw_url or "").strip().rstrip("/")
    if not value:
        return ""
    if value.endswith("/chat/completions"):
        return value.removesuffix("/chat/completions")
    if value.endswith("/v1"):
        return value
    return f"{value}/v1"


def mean_or_zero(values: list[float]) -> float:
    return mean(values) if values else 0.0


def resolve_output_root(raw_output_root: str) -> Path:
    if str(raw_output_root or "").strip():
        return ensure_dir(raw_output_root)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return ensure_dir(Path.cwd().resolve() / ".out" / "pivotrl" / stamp)


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


def build_local_craftax_env(*, bind_host: str = "0.0.0.0", bind_port: int = CRAFTAX_PORT) -> dict[str, str]:
    return {
        **os.environ,
        "PYTHONUNBUFFERED": "1",
        "PYTHONPATH": f"{REMOTE_ROOT}/src",
        "NANOHORIZON_CRAFTAX_BIND_HOST": bind_host,
        "NANOHORIZON_CRAFTAX_BIND_PORT": str(bind_port),
        "CUDA_VISIBLE_DEVICES": "",
        "JAX_PLATFORMS": "cpu",
        "JAX_PLATFORM_NAME": "cpu",
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.0",
    }


def _tail_text_file(path: str | Path | None, *, line_count: int = 80) -> str:
    if not path:
        return ""
    target = Path(path).expanduser().resolve()
    if not target.exists():
        return ""
    try:
        lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception:
        return ""
    return "\n".join(lines[-line_count:])


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
            env=build_local_craftax_env(),
            stdout=log_file or sys.stdout,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            start_new_session=True,
        )
        base_url = f"http://127.0.0.1:{CRAFTAX_PORT}"
        deadline = time.time() + 60.0 * 10.0
        while True:
            if process.poll() is not None:
                log_tail = _tail_text_file(log_path)
                message = f"Craftax runtime exited before health check passed with code {process.returncode}"
                if log_tail:
                    raise RuntimeError(f"{message}\n----- Craftax runtime log tail -----\n{log_tail}")
                raise RuntimeError(message)
            try:
                wait_for_http_health(f"{base_url}/health", timeout_seconds=5.0)
                break
            except Exception as exc:
                if time.time() >= deadline:
                    log_tail = _tail_text_file(log_path)
                    if log_tail:
                        raise RuntimeError(
                            f"timed out waiting for {base_url}/health: {exc!r}\n"
                            f"----- Craftax runtime log tail -----\n{log_tail}"
                        ) from exc
                    raise RuntimeError(f"timed out waiting for {base_url}/health: {exc!r}") from exc
                time.sleep(1.0)
        yield base_url
    finally:
        if process is not None:
            with suppress(ProcessLookupError, OSError):
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            with suppress(subprocess.TimeoutExpired):
                process.wait(timeout=10)
        if log_file is not None:
            log_file.close()


def _uses_proxy_edge(url: str) -> bool:
    try:
        hostname = httpx.URL(url).host or ""
    except Exception:
        return False
    return (
        hostname.endswith(".modal.run")
        or hostname.endswith(".w.modal.host")
        or hostname.endswith(".trycloudflare.com")
    )


def _is_synthtunnel_url(url: str) -> bool:
    try:
        parsed = httpx.URL(url)
        hostname = parsed.host or ""
        path = parsed.path or ""
    except Exception:
        return False
    return hostname == "st.usesynth.ai" or hostname.endswith(".st.usesynth.ai") or "/s/rt_" in path


def build_container_probe_headers(
    *,
    container_url: str,
    container_worker_token: str = "",
) -> dict[str, str]:
    headers: dict[str, str] = {}
    if _is_synthtunnel_url(container_url):
        worker_token = str(container_worker_token or "").strip()
        if not worker_token:
            raise RuntimeError(
                "container_worker_token is required for SynthTunnel Craftax preflight probes"
            )
        headers["Authorization"] = f"Bearer {worker_token}"
    if _uses_proxy_edge(container_url):
        headers["Connection"] = "close"
    return headers


def wait_for_http_health(
    url: str,
    *,
    timeout_seconds: float = DEFAULT_REQUEST_TIMEOUT_S,
    headers: dict[str, str] | None = None,
) -> dict[str, Any]:
    deadline = time.time() + timeout_seconds
    last_error: Exception | None = None
    request_headers = dict(headers or {})
    if _uses_proxy_edge(url) and "Connection" not in request_headers:
        request_headers["Connection"] = "close"
    while time.time() < deadline:
        try:
            with httpx.Client(timeout=10.0, follow_redirects=True) as client:
                response = client.get(url, headers=request_headers or None)
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


def _wait_for_task_info(
    base_url: str,
    *,
    timeout_seconds: float = DEFAULT_REQUEST_TIMEOUT_S,
    headers: dict[str, str] | None = None,
) -> dict[str, Any]:
    return wait_for_http_health(
        f"{str(base_url).rstrip('/')}/task_info",
        timeout_seconds=timeout_seconds,
        headers=headers,
    )


def _probe_inference_tool_call(
    *,
    inference_url: str,
    api_key: str,
    model: str,
    timeout_seconds: float = 120.0,
) -> dict[str, Any]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    if _uses_proxy_edge(inference_url):
        headers["Connection"] = "close"
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a Craftax student policy.\n"
                    "Use the provided `craftax_interact` tool exactly once for the final answer.\n"
                    "Do not answer in plain text."
                ),
            },
            {
                "role": "user",
                "content": "Call the tool now with the single action move_right.",
            },
        ],
        "tools": [CRAFTAX_INTERACT_TOOL],
        "tool_choice": "auto",
        "max_tokens": 64,
        "temperature": 0.0,
    }
    deadline = time.time() + max(15.0, float(timeout_seconds))
    last_error: Exception | None = None
    body: Any = None
    attempt = 0
    while time.time() < deadline:
        attempt += 1
        try:
            with httpx.Client(timeout=600.0, follow_redirects=True) as client:
                response = client.post(normalize_inference_url(inference_url), headers=headers, json=payload)
                if response.status_code in {404, 409, 425, 429, 500, 502, 503, 504}:
                    raise RuntimeError(f"teacher probe transient status={response.status_code} body={response.text[:500]}")
                response.raise_for_status()
                body = response.json()
            break
        except Exception as exc:
            last_error = exc
            time.sleep(min(5.0, float(attempt)))
    if body is None:
        raise RuntimeError(f"teacher tool-call probe failed: {last_error!r}")
    if not isinstance(body, dict):
        raise RuntimeError("teacher tool-call probe returned non-object payload")
    return {
        "status": "ok",
        "id": body.get("id"),
        "choices": len(body.get("choices") or []),
        "model": body.get("model"),
    }


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


def _probe_container_roundtrip(
    *,
    container_url: str,
    container_worker_token: str,
    inference_url: str,
    api_key: str,
    model: str,
    request_timeout_seconds: float,
) -> dict[str, Any]:
    rollouts, summary = asyncio.run(
        collect_rollouts_concurrently_with_summary(
            container_url=container_url,
            container_worker_token=container_worker_token,
            inference_url=normalize_inference_url(inference_url),
            model=model,
            api_key=api_key,
            seeds=[0],
            max_steps=1,
            system_prompt=rollout_system_prompt(
                thinking_budget_tokens=0,
                target_action_batch_size=1,
                min_action_batch_size=1,
            ),
            temperature=0.0,
            max_tokens=256,
            enable_thinking=False,
            thinking_budget_tokens=0,
            policy_version="pivotrl-preflight",
            target_action_batch_size=1,
            min_action_batch_size=1,
            request_timeout_seconds=request_timeout_seconds,
            max_concurrent_rollouts=1,
            trace_prefix="pivotrl_preflight_roundtrip",
            rollout_concurrency=1,
            rollout_semaphore_limit=1,
            request_logprobs=False,
        )
    )
    if not rollouts:
        raise RuntimeError(f"container roundtrip preflight returned no rollouts: {summary}")
    rollout = rollouts[0]
    if rollout.get("error"):
        raise RuntimeError(str(rollout.get("error")))
    if not is_rollout_payload(rollout):
        raise RuntimeError(f"container roundtrip preflight returned non-rollout payload: {rollout}")
    return {
        "status": str(rollout.get("success_status") or "unknown"),
        "outcome_reward": rollout_outcome_reward(rollout),
        "achievements": rollout_achievements(rollout),
        "summary": summary,
    }


def run_bootstrap_preflight(
    *,
    container_url: str,
    container_worker_token: str,
    inference_url: str,
    api_key: str,
    model: str,
    request_timeout_seconds: float,
    output_path: Path,
) -> dict[str, Any]:
    container_probe_timeout = min(float(request_timeout_seconds), 30.0)
    container_headers = build_container_probe_headers(
        container_url=container_url,
        container_worker_token=container_worker_token,
    )
    print("[pivotrl] preflight: checking Craftax health", flush=True)
    craftax_health = wait_for_http_health(
        f"{str(container_url).rstrip('/')}/health",
        timeout_seconds=container_probe_timeout,
        headers=container_headers,
    )
    print("[pivotrl] preflight: checking Craftax task_info", flush=True)
    task_info = _wait_for_task_info(
        container_url,
        timeout_seconds=container_probe_timeout,
        headers=container_headers,
    )
    print("[pivotrl] preflight: probing teacher tool call", flush=True)
    teacher_tool_call = _probe_inference_tool_call(
        inference_url=inference_url,
        api_key=api_key,
        model=model,
        timeout_seconds=request_timeout_seconds,
    )
    print("[pivotrl] preflight: probing Craftax roundtrip", flush=True)
    craftax_roundtrip = _probe_container_roundtrip(
        container_url=container_url,
        container_worker_token=container_worker_token,
        inference_url=inference_url,
        api_key=api_key,
        model=model,
        request_timeout_seconds=request_timeout_seconds,
    )
    payload = {
        "craftax_health": craftax_health,
        "task_info": task_info,
        "teacher_tool_call": teacher_tool_call,
        "craftax_roundtrip": craftax_roundtrip,
        "inference_base_url": _normalize_inference_base_url(inference_url),
        "completed_at": now_utc_iso(),
    }
    write_json(output_path, payload)
    return payload


def _load_text_only_causal_lm(*, base_model: str, device: str, use_cache: bool = True) -> Any:
    import torch
    from transformers import AutoModelForCausalLM

    model_kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "torch_dtype": torch.bfloat16 if device == "cuda" else torch.float32,
    }
    if device == "cuda":
        model_kwargs["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(base_model, **model_kwargs)
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
FALLBACK_TARGET_MODULES = ["c_attn", "c_proj", "c_fc"]


def infer_target_modules(model: Any) -> list[str]:
    available_suffixes = {str(name).split(".")[-1] for name, _module in model.named_modules()}
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
        normalized.append({"role": role, "content": str(item.get("content") or "")})
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
    batch = {key: value.to(model.device) for key, value in batch.items()}
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
        generation_kwargs.update({"do_sample": True, "temperature": float(temperature), "top_p": float(top_p)})
    else:
        generation_kwargs["do_sample"] = False
    with suppress(Exception):
        generation_kwargs["attention_mask"] = inputs.get("attention_mask")
    outputs = model.generate(inputs["input_ids"], **generation_kwargs)
    completion_ids = outputs[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(completion_ids, skip_special_tokens=True).strip()


def build_tool_call_text(actions: list[str]) -> str:
    payload = {
        "name": PRIMARY_TOOL_NAME,
        "arguments": {"actions_list": actions},
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


def infer_target_achievement(*, current_turn: dict[str, Any], next_turn: dict[str, Any]) -> tuple[str | None, str]:
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
    # Keep tiny/sparse bootstrap runs trainable by defaulting to the earliest
    # progression milestone when we have a valid rollout but no richer state cue.
    return "collect_wood", "bootstrap fallback from early-game default progression"


def build_rubric(*, target_achievement: str, state_text: str, inventory: dict[str, int]) -> Rubric:
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


def requirements_satisfied(*, rubric: Rubric, state_text: str, inventory: dict[str, int]) -> bool:
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
            {str(key): int(value) for key, value in inventory_summary.items() if isinstance(value, (int, float))}
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
        {"role": "user", "content": prompt},
    ]


def build_candidate_pivots(*, rollouts: list[dict[str, Any]], lookback: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    target_counts = {target: 0 for target in sorted(RUBRIC_TARGETS)}
    skipped_missing_target = 0
    fallback_count = 0

    def append_bootstrap_fallback_candidate(rollout: dict[str, Any], *, trace_id: str, rollout_id: str, seed: int, outcome_reward: float) -> bool:
        nonlocal fallback_count
        metadata = rollout.get("metadata")
        inventory = {}
        if isinstance(metadata, dict) and isinstance(metadata.get("inventory"), dict):
            inventory = {
                str(key): int(value)
                for key, value in metadata["inventory"].items()
                if isinstance(value, (int, float))
            }
        turns = rollout_turns(rollout)
        state_text = ""
        for turn in reversed(turns):
            prompt_messages = [item for item in turn.get("prompt_messages", []) if isinstance(item, dict)]
            state_text = latest_user_text(prompt_messages)
            if state_text:
                break
        state_text = str(
            state_text
            or ((rollout.get("reward_info") or {}).get("details", {}).get("last_inference_error") if isinstance((rollout.get("reward_info") or {}).get("details"), dict) else "")
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
            return False
        rubric = build_rubric(target_achievement=target_achievement, state_text=state_text, inventory=inventory)
        candidates.append(
            {
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
        )
        target_counts[target_achievement] += 1
        fallback_count += 1
        return True

    for rollout in rollouts:
        if not isinstance(rollout, dict):
            continue
        turns = rollout_turns(rollout)
        trace_id = str(rollout.get("trace_correlation_id") or rollout.get("trial_id") or "")
        rollout_id = str(rollout.get("rollout_id") or trace_id)
        seed = int(rollout.get("_request_seed") or 0)
        outcome_reward = float(rollout_outcome_reward(rollout))
        if len(turns) <= lookback:
            append_bootstrap_fallback_candidate(
                rollout,
                trace_id=trace_id,
                rollout_id=rollout_id,
                seed=seed,
                outcome_reward=outcome_reward,
            )
            continue
        candidates_before_rollout = len(candidates)
        for idx in range(lookback, len(turns)):
            pivot_turn = turns[idx - lookback]
            achievement_turn = turns[idx]
            target_achievement, transition_reason = infer_target_achievement(current_turn=pivot_turn, next_turn=achievement_turn)
            if target_achievement not in RUBRIC_TARGETS:
                skipped_missing_target += 1
                continue
            prompt_messages = [item for item in pivot_turn.get("prompt_messages", []) if isinstance(item, dict)]
            state_text = latest_user_text(prompt_messages)
            inventory = parse_inventory_from_text(state_text)
            demonstrated_actions = (
                [str(item).strip().lower() for item in pivot_turn.get("actions", []) if str(item).strip()]
                if isinstance(pivot_turn.get("actions"), list)
                else extract_actions_from_text(str(pivot_turn.get("assistant_text") or ""))
            )
            rubric = build_rubric(target_achievement=target_achievement, state_text=state_text, inventory=inventory)
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
        if len(candidates) == candidates_before_rollout:
            append_bootstrap_fallback_candidate(
                rollout,
                trace_id=trace_id,
                rollout_id=rollout_id,
                seed=seed,
                outcome_reward=outcome_reward,
            )
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
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    max_pivots: int,
    min_kept_pivots: int,
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
    profiled_pivots: list[dict[str, Any]] = []
    reward_means: list[float] = []
    reward_variances: list[float] = []
    for pivot in selected_pivots:
        reference_rewards: list[float] = []
        reference_action_samples: list[dict[str, Any]] = []
        state_text = str(pivot["state_text"])
        inventory = {str(key): int(value) for key, value in dict(pivot["pre_achievement_inventory_or_summary"]).items()}
        training_messages = [{"role": str(item["role"]), "content": str(item["content"])} for item in pivot["training_messages"]]
        for _index in range(profile_k):
            raw_completion = sample_completion_text(
                tokenizer=tokenizer,
                model=model,
                prompt_messages=training_messages,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )
            actions = extract_actions_from_text(raw_completion) or ["sleep"]
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
        kept_by_threshold = bool(sigma_hat_sq > 0.0 and mu_hat < float(lambda_diff))
        profiled = {
            **pivot,
            "profile_stats": {
                "reference_rewards": reference_rewards,
                "reference_action_samples": reference_action_samples,
                "mu_hat": mu_hat,
                "sigma_hat_sq": sigma_hat_sq,
                "kept": kept_by_threshold,
                "profile_k": int(profile_k),
            },
        }
        profiled_pivots.append(profiled)
        reward_means.append(mu_hat)
        reward_variances.append(sigma_hat_sq)
        if kept_by_threshold:
            kept.append(profiled)
    min_kept_target = max(1, min(int(min_kept_pivots), len(profiled_pivots))) if profiled_pivots else 0
    backfill_kept_count = 0
    if len(kept) < min_kept_target and profiled_pivots:
        kept_ids = {str(item["pivot_id"]) for item in kept}
        near_miss_candidates = [
            item
            for item in sorted(
                profiled_pivots,
                key=lambda entry: (
                    float(entry["profile_stats"]["mu_hat"]),
                    -float(entry["profile_stats"]["sigma_hat_sq"]),
                    str(entry["pivot_id"]),
                ),
            )
            if str(item["pivot_id"]) not in kept_ids
        ]
        for backfilled in near_miss_candidates[: max(0, min_kept_target - len(kept))]:
            backfilled["profile_stats"] = {
                **backfilled["profile_stats"],
                "kept": True,
                "backfill_kept": True,
            }
            kept.append(backfilled)
            backfill_kept_count += 1
    release_cuda_memory()
    return kept, {
        "profiled_count": len(selected_pivots),
        "kept_count": len(kept),
        "dropped_count": len(selected_pivots) - len(kept),
        "backfill_kept_count": backfill_kept_count,
        "lambda_diff": float(lambda_diff),
        "profile_k": int(profile_k),
        "min_kept_pivots": int(min_kept_target),
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
        training_messages = [{"role": str(item["role"]), "content": str(item["content"])} for item in pivot["training_messages"]]
        state_text = str(pivot["state_text"])
        inventory = {str(key): int(value) for key, value in dict(pivot["pre_achievement_inventory_or_summary"]).items()}
        for _sample_idx in range(max(1, int(group_size))):
            raw_completion = sample_completion_text(
                tokenizer=tokenizer,
                model=model,
                prompt_messages=training_messages,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )
            actions = extract_actions_from_text(raw_completion) or ["sleep"]
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
            del sample_idx
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
        batch = tokenize_prompt_and_completion(tokenizer, sample.prompt_messages, sample.completion_text, max_length=max_length)
        tensor_batch = {key: value.to(model.device) for key, value in batch.items()}
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
        reference_logprob_tensor = torch.tensor([sample.reference_logprob], device=model.device, dtype=sequence_new_logprob.dtype)
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


def ensure_live_bootstrap_teacher_args(args: Any) -> tuple[str, str]:
    teacher_inference_url = normalize_inference_url(
        str(getattr(args, "teacher_inference_url", "") or os.getenv("NANOHORIZON_TEACHER_INFERENCE_URL") or "")
    )
    teacher_api_key = str(getattr(args, "teacher_api_key", "") or os.getenv("NANOHORIZON_TEACHER_API_KEY") or "").strip()
    if not teacher_inference_url:
        raise RuntimeError(
            "teacher_inference_url is required for live PivotRL bootstrap in the Modal worker path"
        )
    return teacher_inference_url, teacher_api_key


def run_bootstrap(args: Any, *, output_root: Path) -> tuple[Path, dict[str, Any]]:
    print("[pivotrl] bootstrap phase starting", flush=True)
    artifacts_dir = ensure_dir(output_root / "artifacts")
    rollouts_path = artifacts_dir / "bootstrap_rollouts.jsonl"
    successful_path = artifacts_dir / "bootstrap_successful_rollouts.jsonl"
    summary_path = artifacts_dir / "bootstrap_summary.json"
    preflight_path = artifacts_dir / "bootstrap_preflight.json"

    container_url = str(getattr(args, "container_url", "") or "").strip()
    if not container_url:
        raise RuntimeError("container_url must be set before bootstrap; start local Craftax in the worker or pass an external container URL")
    container_worker_token = str(getattr(args, "container_worker_token", "") or "").strip()
    teacher_inference_url, teacher_api_key = ensure_live_bootstrap_teacher_args(args)
    teacher_model = str(getattr(args, "teacher_model", "") or DEFAULT_TEACHER_MODEL).strip() or DEFAULT_TEACHER_MODEL
    seeds = [int(getattr(args, "bootstrap_seed_start", 0)) + offset for offset in range(max(1, int(getattr(args, "bootstrap_seed_count", 8))))]
    request_timeout_seconds = float(getattr(args, "request_timeout_seconds", DEFAULT_REQUEST_TIMEOUT_S))
    system_prompt = rollout_system_prompt(
        thinking_budget_tokens=int(getattr(args, "bootstrap_thinking_budget_tokens", 0)),
        target_action_batch_size=int(getattr(args, "bootstrap_target_action_batch_size", 4)),
        min_action_batch_size=int(getattr(args, "bootstrap_min_action_batch_size", 3)),
    )

    run_bootstrap_preflight(
        container_url=container_url,
        container_worker_token=container_worker_token,
        inference_url=teacher_inference_url,
        api_key=teacher_api_key,
        model=teacher_model,
        request_timeout_seconds=request_timeout_seconds,
        output_path=preflight_path,
    )

    rollouts, collection_summary = asyncio.run(
        collect_rollouts_concurrently_with_summary(
            container_url=container_url,
            container_worker_token=container_worker_token,
            inference_url=teacher_inference_url,
            model=teacher_model,
            api_key=teacher_api_key,
            seeds=seeds,
            max_steps=int(getattr(args, "bootstrap_max_steps", 48)),
            system_prompt=system_prompt,
            temperature=float(getattr(args, "bootstrap_temperature", 0.2)),
            max_tokens=int(getattr(args, "bootstrap_max_new_tokens", 3072)),
            enable_thinking=bool(getattr(args, "enable_thinking", False)),
            thinking_budget_tokens=int(getattr(args, "bootstrap_thinking_budget_tokens", 0)),
            policy_version="pivotrl-teacher-bootstrap",
            target_action_batch_size=int(getattr(args, "bootstrap_target_action_batch_size", 4)),
            min_action_batch_size=int(getattr(args, "bootstrap_min_action_batch_size", 3)),
            request_timeout_seconds=request_timeout_seconds,
            max_concurrent_rollouts=int(getattr(args, "bootstrap_rollout_concurrency", 4)),
            trace_prefix="pivotrl_bootstrap",
            rollout_concurrency=int(getattr(args, "bootstrap_rollout_concurrency", 4)),
            rollout_semaphore_limit=int(getattr(args, "bootstrap_rollout_semaphore_limit", 4)),
            request_logprobs=False,
        )
    )
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
        "achievements": sorted({achievement for rollout in successful_rollouts for achievement in rollout_achievements(rollout)}),
        "artifacts": {
            "bootstrap_rollouts": str(rollouts_path),
            "bootstrap_successful_rollouts": str(successful_path),
            "bootstrap_preflight": str(preflight_path),
        },
    }
    write_json(summary_path, bootstrap_summary)
    if not successful_rollouts:
        sample_failures = [
            str(rollout.get("error") or rollout.get("status_detail") or "").strip()
            for rollout in rollouts
            if isinstance(rollout, dict) and not is_rollout_payload(rollout)
        ]
        failure_suffix = f": {' | '.join(detail for detail in sample_failures[:3] if detail)}" if sample_failures else ""
        raise RuntimeError(
            "bootstrap rollout collection produced no successful rollout payloads"
            f"{failure_suffix}"
        )
    return successful_path, bootstrap_summary


def run_build_dataset(args: Any, *, output_root: Path, bootstrap_rollouts_path: Path | None = None) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    print("[pivotrl] build-dataset phase starting", flush=True)
    artifacts_dir = ensure_dir(output_root / "artifacts")
    candidate_path = bootstrap_rollouts_path or (
        Path(str(getattr(args, "bootstrap_rollouts_path", ""))).expanduser().resolve()
        if str(getattr(args, "bootstrap_rollouts_path", "")).strip()
        else artifacts_dir / "bootstrap_successful_rollouts.jsonl"
    )
    if not candidate_path.exists():
        raise FileNotFoundError(f"bootstrap rollouts not found: {candidate_path}")
    rollouts = read_jsonl(candidate_path)
    candidates, candidate_summary = build_candidate_pivots(
        rollouts=rollouts,
        lookback=max(1, int(getattr(args, "lookback", 1))),
    )
    profiled, profile_summary = profile_pivots(
        pivots=candidates,
        base_model=str(getattr(args, "base_model", "") or DEFAULT_BASE_MODEL),
        profile_k=int(getattr(args, "profile_k", 4)),
        lambda_diff=float(getattr(args, "lambda_diff", 0.75)),
        max_new_tokens=int(getattr(args, "profile_max_new_tokens", 96)),
        temperature=float(getattr(args, "profile_temperature", 0.8)),
        top_p=float(getattr(args, "profile_top_p", 0.95)),
        max_pivots=int(getattr(args, "max_pivots", 128)),
        min_kept_pivots=int(getattr(args, "min_kept_pivots", 4)),
    )
    dataset_path = artifacts_dir / "pivot_dataset.jsonl"
    profile_summary_path = artifacts_dir / "pivot_profile_summary.json"
    write_jsonl(dataset_path, profiled)
    write_json(profile_summary_path, {**candidate_summary, **profile_summary, "bootstrap_rollouts_path": str(candidate_path)})
    if not profiled:
        raise RuntimeError(
            "pivot dataset is empty after pivot profiling: "
            f"candidate_count={len(candidates)} kept_count={len(profiled)} source={candidate_path}"
        )
    return dataset_path, candidate_summary, profile_summary


def run_train(args: Any, *, output_root: Path, dataset_path: Path | None = None) -> tuple[Path, dict[str, Any]]:
    print("[pivotrl] train phase starting", flush=True)
    artifacts_dir = ensure_dir(output_root / "artifacts")
    resolved_dataset_path = dataset_path or (
        Path(str(getattr(args, "dataset_path", ""))).expanduser().resolve()
        if str(getattr(args, "dataset_path", "")).strip()
        else artifacts_dir / "pivot_dataset.jsonl"
    )
    if not resolved_dataset_path.exists():
        raise FileNotFoundError(f"pivot dataset not found: {resolved_dataset_path}")
    pivots = read_jsonl(resolved_dataset_path)
    if not pivots:
        raise RuntimeError(f"pivot dataset is empty: {resolved_dataset_path}")

    tokenizer, model, optimizer = initialize_policy(
        base_model=str(getattr(args, "base_model", "") or DEFAULT_BASE_MODEL),
        lora_rank=int(getattr(args, "lora_rank", 16)),
        learning_rate=float(getattr(args, "learning_rate", 1e-5)),
    )
    train_pivots = list(pivots)
    max_pivots = int(getattr(args, "max_pivots", 0))
    if max_pivots > 0:
        train_pivots = train_pivots[:max_pivots]
    iteration_summaries: list[dict[str, Any]] = []
    remaining_steps = max(0, int(getattr(args, "max_train_steps", 16)))

    for iteration_index in range(max(1, int(getattr(args, "train_iterations", 2)))):
        if remaining_steps <= 0:
            break
        pivots_per_iteration = int(getattr(args, "pivots_per_iteration", 16))
        if pivots_per_iteration > 0 and len(train_pivots) > pivots_per_iteration:
            selected_pivots = random.sample(train_pivots, k=pivots_per_iteration)
        else:
            selected_pivots = list(train_pivots)
        samples, sampling_summary = build_group_samples(
            tokenizer=tokenizer,
            model=model,
            pivots=selected_pivots,
            group_size=int(getattr(args, "group_size", 4)),
            max_length=int(getattr(args, "max_length", 8192)),
            max_new_tokens=int(getattr(args, "sample_max_new_tokens", 96)),
            temperature=float(getattr(args, "sample_temperature", 0.8)),
            top_p=float(getattr(args, "sample_top_p", 0.95)),
        )
        steps_this_iteration = min(remaining_steps, max(1, int(getattr(args, "train_steps_per_iteration", 8))))
        iteration_summary = train_iteration(
            tokenizer=tokenizer,
            model=model,
            optimizer=optimizer,
            samples=samples,
            clip_epsilon=float(getattr(args, "clip_epsilon", 0.2)),
            kl_coef=float(getattr(args, "kl_coef", 0.02)),
            max_length=int(getattr(args, "max_length", 8192)),
            max_steps=steps_this_iteration,
        )
        iteration_summaries.append(
            {"iteration_index": iteration_index, "sampling": sampling_summary, "training": iteration_summary}
        )
        remaining_steps -= int(iteration_summary.get("optimizer_steps", 0))

    adapter_dir = artifacts_dir / "pivotrl_adapter"
    save_adapter(model=model, tokenizer=tokenizer, destination=adapter_dir)
    release_cuda_memory()
    training_summary = {
        "method_name": "pivotrl_preachievement_offline",
        "dataset_path": str(resolved_dataset_path),
        "pivot_count": len(train_pivots),
        "group_size": int(getattr(args, "group_size", 4)),
        "iterations_completed": len(iteration_summaries),
        "optimizer_steps_total": sum(int(item["training"].get("optimizer_steps", 0)) for item in iteration_summaries),
        "mean_iteration_reward": mean_or_zero(
            [float(item["sampling"].get("mean_group_reward", 0.0)) for item in iteration_summaries]
        ),
        "mean_iteration_loss": mean_or_zero(
            [float(item["training"].get("mean_loss", 0.0)) for item in iteration_summaries if not item["training"].get("skipped")]
        ),
        "offline_only_after_bootstrap": True,
        "adapter_dir": str(adapter_dir),
        "iterations": iteration_summaries,
    }
    write_json(artifacts_dir / "training_summary.json", training_summary)
    return adapter_dir, training_summary


def build_modal_train_result(
    *,
    output_root: Path,
    bootstrap_summary: dict[str, Any],
    pivot_profile_summary: dict[str, Any],
    training_summary: dict[str, Any],
    adapter_dir: Path,
) -> dict[str, Any]:
    artifacts_dir = output_root / "artifacts"
    return {
        "output_root": str(output_root),
        "adapter_dir": str(adapter_dir),
        "bootstrap_summary": bootstrap_summary,
        "pivot_profile_summary": pivot_profile_summary,
        "training_summary": training_summary,
        "artifacts": {
            "bootstrap_rollouts": str(artifacts_dir / "bootstrap_rollouts.jsonl"),
            "bootstrap_successful_rollouts": str(artifacts_dir / "bootstrap_successful_rollouts.jsonl"),
            "bootstrap_preflight": str(artifacts_dir / "bootstrap_preflight.json"),
            "bootstrap_summary": str(artifacts_dir / "bootstrap_summary.json"),
            "pivot_dataset": str(artifacts_dir / "pivot_dataset.jsonl"),
            "pivot_profile_summary": str(artifacts_dir / "pivot_profile_summary.json"),
            "training_summary": str(artifacts_dir / "training_summary.json"),
            "modal_train_result": str(artifacts_dir / "modal_train_result.json"),
        },
    }


def run_modal_train_pipeline(args: Any, *, output_root: Path) -> dict[str, Any]:
    print("[pivotrl] modal train pipeline starting", flush=True)
    artifacts_dir = ensure_dir(output_root / "artifacts")
    bootstrap_rollouts_value = str(getattr(args, "bootstrap_rollouts_path", "") or "").strip()
    if bootstrap_rollouts_value:
        bootstrap_rollouts_path = Path(bootstrap_rollouts_value).expanduser().resolve()
        bootstrap_rows = read_jsonl(bootstrap_rollouts_path)
        successful_bootstrap_rows = [row for row in bootstrap_rows if isinstance(row, dict) and is_rollout_payload(row)]
        bootstrap_summary = {
            "requested_rollouts": 0,
            "successful_rollouts": len(successful_bootstrap_rows),
            "bootstrap_rollouts_path": str(bootstrap_rollouts_path),
            "skipped_live_bootstrap": True,
            "artifacts": {
                "bootstrap_successful_rollouts": str(bootstrap_rollouts_path),
            },
        }
        write_json(artifacts_dir / "bootstrap_summary.json", bootstrap_summary)
        if not successful_bootstrap_rows:
            raise RuntimeError(
                f"bootstrap rollout input contains no successful rollout payloads: {bootstrap_rollouts_path}"
            )
    else:
        ensure_live_bootstrap_teacher_args(args)
        bootstrap_rollouts_path, bootstrap_summary = run_bootstrap(args, output_root=output_root)
    dataset_path, _candidate_summary, profile_summary = run_build_dataset(
        args,
        output_root=output_root,
        bootstrap_rollouts_path=bootstrap_rollouts_path,
    )
    adapter_dir, training_summary = run_train(args, output_root=output_root, dataset_path=dataset_path)
    result = build_modal_train_result(
        output_root=output_root,
        bootstrap_summary=bootstrap_summary,
        pivot_profile_summary=profile_summary,
        training_summary=training_summary,
        adapter_dir=adapter_dir,
    )
    write_json(artifacts_dir / "modal_train_result.json", result)
    return result
