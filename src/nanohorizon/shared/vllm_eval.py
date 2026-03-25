from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from nanohorizon.craftax_core.metadata import DEFAULT_ACTION_NAMES, PRIMARY_TOOL_NAME
from nanohorizon.custom_vllm import (
    build_thinking_budget_request_overrides,
    enable_thinking_budget_support,
)
from nanohorizon.shared.openai_compat import (
    create_chat_completion,
    extract_openai_tool_calls,
    sanitize_craftax_actions,
)

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
                    "items": {
                        "type": "string",
                        "enum": CRAFTAX_ACTION_ENUM,
                    },
                    "minItems": 1,
                    "maxItems": 10,
                }
            },
            "required": ["actions_list"],
            "additionalProperties": False,
        },
    },
}

DEFAULT_VLLM_PORT = 8003
REMOTE_ROOT = "/root/nanohorizon"
OFFLINE_VENV_ROOT = "/opt/nanohorizon-offline-venvs"


@dataclass(frozen=True)
class LocalVLLMEvalConfig:
    model: str
    served_model_name: str = ""
    lora_name: str = ""
    lora_path: str = ""
    max_lora_rank: int = 16
    port: int = DEFAULT_VLLM_PORT
    max_model_len: int = 4096
    max_new_tokens: int = 1024
    gpu_memory_utilization: float = 0.90
    max_num_seqs: int = 16
    max_num_batched_tokens: int = 4096
    reasoning_parser: str = "qwen3"
    tool_call_parser: str = "qwen3_coder"
    guided_decoding_backend: str = "outlines"
    enable_thinking: bool = True
    enforce_eager: bool = True
    vllm_bin: str = f"{OFFLINE_VENV_ROOT}/teacher/bin/vllm"


def infer_lora_rank(adapter_dir: str | Path) -> int:
    adapter_path = Path(adapter_dir).expanduser().resolve()
    config_path = adapter_path / "adapter_config.json"
    if not config_path.exists():
        return 16
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    rank = payload.get("r")
    if isinstance(rank, int) and rank > 0:
        return rank
    return 16


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


def build_vllm_serve_command(config: LocalVLLMEvalConfig) -> list[str]:
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
        "--language-model-only",
    ]
    if config.enable_thinking:
        cmd += [
            "--reasoning-parser",
            config.reasoning_parser,
        ]
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


def build_vllm_runtime_env(*, model_ref: str, enable_thinking: bool) -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("PYTHONPATH", _pythonpath_with_repo())
    env["PYTHONPATH"] = _pythonpath_with_repo()
    if enable_thinking:
        cmd, env = enable_thinking_budget_support(
            cmd=["vllm", "serve", model_ref],
            env=env,
            model_ref=model_ref,
        )
        del cmd
    return env


def wait_for_local_vllm(*, port: int, timeout_seconds: float = 60 * 20) -> None:
    deadline = time.time() + timeout_seconds
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            with httpx.Client(timeout=5.0) as client:
                response = client.get(f"http://127.0.0.1:{int(port)}/health")
                if response.status_code == 200:
                    return
                last_error = RuntimeError(f"/health returned HTTP {response.status_code}")
        except Exception as exc:
            last_error = exc
        time.sleep(1.0)
    raise RuntimeError(f"Timed out waiting for local vLLM health: {last_error!r}")


@contextmanager
def local_vllm_server(
    *,
    config: LocalVLLMEvalConfig,
    log_path: str | Path | None = None,
) -> Iterator[dict[str, Any]]:
    env = build_vllm_runtime_env(model_ref=config.model, enable_thinking=config.enable_thinking)
    cmd = build_vllm_serve_command(config)
    if config.enable_thinking:
        cmd, env = enable_thinking_budget_support(cmd=cmd, env=env, model_ref=config.model)
    log_file = None
    try:
        if log_path:
            target = Path(log_path).expanduser().resolve()
            target.parent.mkdir(parents=True, exist_ok=True)
            log_file = target.open("w", encoding="utf-8", buffering=1)
        print("Launching eval vLLM:", " ".join(shlex.quote(part) for part in cmd), flush=True)
        process = subprocess.Popen(
            cmd,
            env={**env, "PYTHONUNBUFFERED": "1"},
            stdout=log_file or sys.stdout,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        wait_for_local_vllm(port=config.port)
        yield {
            "process": process,
            "base_url": f"http://127.0.0.1:{config.port}/v1",
        }
    finally:
        if "process" in locals():
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=30)
        if log_file is not None:
            log_file.close()


def generate_with_vllm_server(
    *,
    base_model: str,
    prompts: list[Any],
    max_new_tokens: int,
    thinking_budget_tokens: int,
    enable_thinking: bool = True,
    server_log_path: str | Path | None = None,
    max_model_len: int = 4096,
    enforce_eager: bool = True,
    adapter_dir: str | Path | None = None,
) -> list[str]:
    if not prompts:
        return []

    lora_name = ""
    lora_path = ""
    request_model = base_model
    max_lora_rank = 16
    if adapter_dir is not None:
        lora_name = "policy-lora"
        lora_path = str(Path(adapter_dir).expanduser().resolve())
        request_model = lora_name
        max_lora_rank = infer_lora_rank(lora_path)
    config = LocalVLLMEvalConfig(
        model=base_model,
        served_model_name=base_model,
        lora_name=lora_name,
        lora_path=lora_path,
        max_lora_rank=max_lora_rank,
        max_model_len=max_model_len,
        max_new_tokens=max_new_tokens,
        enable_thinking=enable_thinking,
        enforce_eager=enforce_eager,
    )
    outputs: list[str] = []
    with local_vllm_server(config=config, log_path=server_log_path) as server:
        for prompt in prompts:
            if not isinstance(prompt, list):
                raise ValueError("vLLM eval expects chat message lists")
            normalized_messages = [
                {
                    "role": str(item.get("role") or "user"),
                    "content": str(item.get("content") or ""),
                }
                for item in prompt
                if isinstance(item, dict)
            ]
            system_directive = (
                "You are a Craftax student policy.\n"
                f"Use the provided `{PRIMARY_TOOL_NAME}` tool exactly once for the final answer.\n"
                "Do not answer in plain text.\n"
                "Do not output JSON in assistant content.\n"
                "After reasoning, your final assistant action must be a tool call."
            )
            if normalized_messages and normalized_messages[0]["role"] == "system":
                normalized_messages[0]["content"] = system_directive
            else:
                normalized_messages.insert(0, {"role": "system", "content": system_directive})
            extra_body = build_thinking_budget_request_overrides(
                enable_thinking=enable_thinking,
                thinking_budget=thinking_budget_tokens,
            )
            extra_body["guided_decoding_backend"] = config.guided_decoding_backend
            payload = create_chat_completion(
                model=request_model,
                messages=normalized_messages,
                max_tokens=max_new_tokens,
                temperature=0.0,
                base_url=str(server["base_url"]),
                api_key="",
                timeout_seconds=300.0,
                tools=[CRAFTAX_INTERACT_TOOL],
                tool_choice="auto",
                extra_body=extra_body,
            )
            tool_calls = extract_openai_tool_calls(payload, tool_name=PRIMARY_TOOL_NAME)
            actions: list[str] = []
            for tool_call in tool_calls:
                arguments = tool_call.get("arguments", {})
                if not isinstance(arguments, dict):
                    continue
                values = arguments.get("actions_list")
                if isinstance(values, list):
                    actions = sanitize_craftax_actions(values)
                    if actions:
                        break
            outputs.append("\n".join(actions) if actions else "")
    return outputs
