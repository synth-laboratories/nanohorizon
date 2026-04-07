from __future__ import annotations

import json
import os
import shlex
import shutil
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
SUPPORTED_VLLM_LORA_RANKS = (1, 8, 16, 32, 64, 128, 256, 320, 512)


@dataclass(frozen=True)
class LocalVLLMEvalConfig:
    model: str
    served_model_name: str = ""
    api_key: str = ""
    host: str = "127.0.0.1"
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
    allow_runtime_lora_updates: bool = False
    vllm_use_v1: bool = True
    server_dev_mode: bool = True
    hf_home: str = ""
    triton_cache_dir: str = ""
    torchinductor_cache_dir: str = ""
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
    vllm_bin = _resolve_vllm_bin(config.vllm_bin)
    cmd = [
        vllm_bin,
        "serve",
        config.model,
        "--served-model-name",
        config.served_model_name or config.model,
        "--host",
        config.host,
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
    if config.api_key:
        cmd += ["--api-key", config.api_key]
    if config.enable_thinking:
        cmd += [
            "--reasoning-parser",
            config.reasoning_parser,
        ]
    if config.lora_path:
        max_lora_rank = _normalize_vllm_max_lora_rank(config.max_lora_rank)
        cmd += [
            "--enable-lora",
            "--max-lora-rank",
            str(max_lora_rank),
            "--lora-modules",
            f"{config.lora_name}={config.lora_path}",
        ]
    if config.enforce_eager:
        cmd.append("--enforce-eager")
    return cmd


def _resolve_vllm_bin(configured_bin: str) -> str:
    candidate = str(configured_bin or "").strip()
    if candidate:
        candidate_path = Path(candidate)
        if candidate_path.is_file():
            return str(candidate_path)
        resolved = shutil.which(candidate)
        if resolved:
            return resolved
    resolved = shutil.which("vllm")
    if resolved:
        return resolved
    return candidate or "vllm"


def _normalize_vllm_max_lora_rank(value: int) -> int:
    requested = max(1, int(value))
    for supported in SUPPORTED_VLLM_LORA_RANKS:
        if requested <= supported:
            return supported
    return SUPPORTED_VLLM_LORA_RANKS[-1]


def _install_lora_patch_into_venv(vllm_bin: str) -> None:
    resolved_vllm_bin = _resolve_vllm_bin(vllm_bin)
    venv_bin = Path(resolved_vllm_bin).resolve().parent
    venv_root = venv_bin.parent
    candidates = sorted(
        venv_root.glob("lib/python*/site-packages/vllm/lora/layers/column_parallel_linear.py")
    )
    if not candidates:
        return
    target = candidates[-1]
    original = target.read_text(encoding="utf-8")
    old = "for i in range(self.n_slices):"
    new = (
        "for i in range(min(self.n_slices, "
        "len(lora_a) if isinstance(lora_a, list) else self.n_slices, "
        "len(lora_b) if isinstance(lora_b, list) else self.n_slices)):"
    )
    if old not in original:
        return
    target.write_text(original.replace(old, new), encoding="utf-8")


def _install_hf_hub_compat_patch(vllm_bin: str) -> None:
    resolved_vllm_bin = _resolve_vllm_bin(vllm_bin)
    venv_bin = Path(resolved_vllm_bin).resolve().parent
    venv_root = venv_bin.parent
    candidates = sorted(venv_root.glob("lib/python*/site-packages/huggingface_hub/__init__.py"))
    if not candidates:
        return
    target = candidates[-1]
    original = target.read_text(encoding="utf-8")
    if "def is_offline_mode(" in original:
        return
    patch = (
        "\n\n"
        "# NanoHorizon compatibility shim for vLLM + transformers stacks that still\n"
        "# import `is_offline_mode` from the package root.\n"
        "def is_offline_mode() -> bool:\n"
        "    import os\n\n"
        "    value = str(\n"
        "        os.environ.get(\"HF_HUB_OFFLINE\")\n"
        "        or os.environ.get(\"TRANSFORMERS_OFFLINE\")\n"
        "        or \"\"\n"
        "    ).strip().lower()\n"
        "    return value in {\"1\", \"on\", \"true\", \"yes\"}\n"
    )
    target.write_text(original + patch, encoding="utf-8")


def _ensure_vllm_python_stack(vllm_bin: str) -> None:
    resolved_vllm_bin = _resolve_vllm_bin(vllm_bin)
    venv_bin = Path(resolved_vllm_bin).resolve().parent
    python_bin = venv_bin / "python"
    pip_bin = venv_bin / "pip"
    if not python_bin.is_file() or not pip_bin.is_file():
        return
    check_cmd = [
        str(python_bin),
        "-c",
        (
            "import huggingface_hub, transformers; "
            "from huggingface_hub import is_offline_mode; "
            "print(huggingface_hub.__version__); "
            "print(transformers.__version__)"
        ),
    ]
    result = subprocess.run(check_cmd, capture_output=True, text=True, check=False)
    if result.returncode == 0:
        return
    _install_hf_hub_compat_patch(resolved_vllm_bin)
    patched_result = subprocess.run(check_cmd, capture_output=True, text=True, check=False)
    if patched_result.returncode == 0:
        return
    error_text = "\n".join(
        part for part in (patched_result.stdout, patched_result.stderr, result.stdout, result.stderr) if part
    )
    requirement = ""
    if "huggingface-hub>=1.5.0,<2.0" in error_text:
        requirement = "huggingface_hub>=1.5.0,<2.0"
    elif "huggingface-hub>=0.34.0,<1.0" in error_text or "huggingface-hub<1.0" in error_text:
        requirement = "huggingface_hub>=0.34.0,<1.0"
    if not requirement:
        raise RuntimeError(f"unable to repair vLLM python stack automatically:\n{error_text.strip()}")
    subprocess.run(
        [
            str(pip_bin),
            "install",
            "--upgrade",
            requirement,
        ],
        check=True,
        text=True,
    )
    _install_hf_hub_compat_patch(resolved_vllm_bin)
    final_result = subprocess.run(check_cmd, capture_output=True, text=True, check=False)
    if final_result.returncode != 0:
        final_error = "\n".join(part for part in (final_result.stdout, final_result.stderr) if part).strip()
        raise RuntimeError(f"unable to repair vLLM python stack automatically:\n{final_error}")


def build_vllm_runtime_env(
    *,
    model_ref: str,
    enable_thinking: bool,
    config: LocalVLLMEvalConfig | None = None,
) -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("PYTHONPATH", _pythonpath_with_repo())
    env["PYTHONPATH"] = _pythonpath_with_repo()
    env["HF_HOME"] = str(
        (config.hf_home if config is not None else "") or env.get("HF_HOME") or "/root/.cache/huggingface"
    )
    env["HF_HUB_ENABLE_HF_TRANSFER"] = str(env.get("HF_HUB_ENABLE_HF_TRANSFER") or "1")
    env["TORCHINDUCTOR_CACHE_DIR"] = str(
        (config.torchinductor_cache_dir if config is not None else "")
        or env.get("TORCHINDUCTOR_CACHE_DIR")
        or "/root/.cache/vllm/torchinductor"
    )
    env["TRITON_CACHE_DIR"] = str(
        (config.triton_cache_dir if config is not None else "")
        or env.get("TRITON_CACHE_DIR")
        or "/root/.cache/vllm/triton"
    )
    if config is None or config.vllm_use_v1:
        env["VLLM_USE_V1"] = "1"
    if config is None or config.server_dev_mode:
        env["VLLM_SERVER_DEV_MODE"] = "1"
    if config is not None and (config.allow_runtime_lora_updates or config.lora_path):
        env["VLLM_ALLOW_RUNTIME_LORA_UPDATING"] = "1"
    if enable_thinking:
        cmd, env = enable_thinking_budget_support(
            cmd=["vllm", "serve", model_ref],
            env=env,
            model_ref=model_ref,
        )
        del cmd
    return env


def wait_for_local_vllm(
    *,
    port: int,
    timeout_seconds: float = 60 * 20,
    api_key: str = "",
) -> None:
    deadline = time.time() + timeout_seconds
    last_error: Exception | None = None
    headers: dict[str, str] = {}
    probe_path = "/health"
    if str(api_key).strip():
        headers["Authorization"] = f"Bearer {str(api_key).strip()}"
        probe_path = "/v1/models"
    while time.time() < deadline:
        try:
            with httpx.Client(timeout=5.0) as client:
                response = client.get(
                    f"http://127.0.0.1:{int(port)}{probe_path}",
                    headers=headers or None,
                )
                if response.status_code == 200:
                    return
                last_error = RuntimeError(f"{probe_path} returned HTTP {response.status_code}")
        except Exception as exc:
            last_error = exc
        time.sleep(1.0)
    raise RuntimeError(f"Timed out waiting for local vLLM health: {last_error!r}")


def _tail_log_text(path: str | Path | None, *, line_count: int = 80) -> str:
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
def local_vllm_server(
    *,
    config: LocalVLLMEvalConfig,
    log_path: str | Path | None = None,
) -> Iterator[dict[str, Any]]:
    env = build_vllm_runtime_env(
        model_ref=config.model,
        enable_thinking=config.enable_thinking,
        config=config,
    )
    resolved_vllm_bin = _resolve_vllm_bin(config.vllm_bin)
    effective_config = LocalVLLMEvalConfig(
        **{**config.__dict__, "vllm_bin": resolved_vllm_bin}
    )
    cmd = build_vllm_serve_command(effective_config)
    if config.enable_thinking:
        cmd, env = enable_thinking_budget_support(cmd=cmd, env=env, model_ref=config.model)
    log_file = None
    try:
        if log_path:
            target = Path(log_path).expanduser().resolve()
            target.parent.mkdir(parents=True, exist_ok=True)
            log_file = target.open("w", encoding="utf-8", buffering=1)
        _ensure_vllm_python_stack(resolved_vllm_bin)
        _install_hf_hub_compat_patch(resolved_vllm_bin)
        if config.lora_path:
            _install_lora_patch_into_venv(resolved_vllm_bin)
        print("Launching eval vLLM:", " ".join(shlex.quote(part) for part in cmd), flush=True)
        process = subprocess.Popen(
            cmd,
            env={**env, "PYTHONUNBUFFERED": "1"},
            stdin=subprocess.DEVNULL,
            stdout=log_file or sys.stdout,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            start_new_session=True,
        )
        deadline = time.time() + 60 * 20
        last_error: Exception | None = None
        while time.time() < deadline:
            if process.poll() is not None:
                log_tail = _tail_log_text(log_path)
                message = f"local vLLM exited before health check passed with code {process.returncode}"
                if log_tail:
                    raise RuntimeError(f"{message}\n----- local vLLM log tail -----\n{log_tail}")
                raise RuntimeError(message)
            try:
                wait_for_local_vllm(port=config.port, timeout_seconds=5.0, api_key=config.api_key)
                break
            except Exception as exc:
                last_error = exc
                time.sleep(1.0)
        else:
            log_tail = _tail_log_text(log_path)
            if log_tail:
                raise RuntimeError(
                    f"Timed out waiting for local vLLM health: {last_error!r}\n"
                    f"----- local vLLM log tail -----\n{log_tail}"
                )
            raise RuntimeError(f"Timed out waiting for local vLLM health: {last_error!r}")
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
