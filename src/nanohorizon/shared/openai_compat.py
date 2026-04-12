from __future__ import annotations

import json
import os
import re
from urllib.parse import urlparse

import httpx
from nanohorizon.craftax_core.metadata import DEFAULT_ACTION_NAMES, PRIMARY_TOOL_NAME

CRAFTAX_ACTIONS = set(DEFAULT_ACTION_NAMES)


def sanitize_craftax_actions(values: list[object]) -> list[str]:
    sanitized: list[str] = []
    for value in values:
        raw = str(value).strip().lower()
        if not raw:
            continue
        if raw in CRAFTAX_ACTIONS:
            sanitized.append(raw)
            continue
        for token in re.findall(r"[a-z_]+", raw):
            if token in CRAFTAX_ACTIONS:
                sanitized.append(token)
    return sanitized


def create_chat_completion(
    *,
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int = 400,
    temperature: float = 0.2,
    base_url: str | None = None,
    api_key: str | None = None,
    timeout_seconds: float | None = None,
    tools: list[dict] | None = None,
    tool_choice: dict | str | None = None,
    extra_body: dict | None = None,
) -> dict:
    resolved_api_key = str(api_key or os.getenv("OPENAI_API_KEY") or "").strip()
    resolved_base_url = str(
        base_url or os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1"
    ).rstrip("/")
    parsed = urlparse(resolved_base_url)
    is_local = parsed.hostname in {"127.0.0.1", "localhost"}
    resolved_timeout = float(
        timeout_seconds
        or os.getenv("NANOHORIZON_OPENAI_TIMEOUT_SECONDS")
        or (300.0 if is_local else 60.0)
    )
    if not resolved_api_key and not is_local:
        raise RuntimeError("OPENAI_API_KEY is required for non-local OpenAI-compatible completions")
    headers: dict[str, str] = {}
    if resolved_api_key:
        headers["Authorization"] = f"Bearer {resolved_api_key}"
    payload: dict = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if tools:
        payload["tools"] = tools
    if tool_choice is not None:
        payload["tool_choice"] = tool_choice
    if extra_body:
        payload.update(extra_body)
    with httpx.Client(timeout=httpx.Timeout(resolved_timeout, connect=30.0)) as client:
        response = client.post(
            f"{resolved_base_url}/chat/completions",
            headers=headers,
            json=payload,
        )
        if response.status_code >= 400:
            raise RuntimeError(
                f"chat completion failed status={response.status_code} body={response.text[:2000]}"
            )
        return response.json()


def chat_completion(
    *,
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int = 400,
    temperature: float = 0.2,
    base_url: str | None = None,
    api_key: str | None = None,
    timeout_seconds: float | None = None,
) -> str:
    payload = create_chat_completion(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        base_url=base_url,
        api_key=api_key,
        timeout_seconds=timeout_seconds,
    )
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise RuntimeError(f"unexpected completion payload: {json.dumps(payload)[:300]}")
    message = choices[0].get("message", {})
    return str(message.get("content") or "").strip()


def extract_craftax_actions(payload: dict, *, tool_name: str = PRIMARY_TOOL_NAME) -> list[str]:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return []
    message = choices[0].get("message", {})
    if not isinstance(message, dict):
        return []

    tool_calls = message.get("tool_calls")
    if isinstance(tool_calls, list):
        for tool_call in tool_calls:
            if not isinstance(tool_call, dict):
                continue
            function = tool_call.get("function", {})
            if not isinstance(function, dict):
                continue
            name = str(function.get("name") or "").strip()
            if name != (tool_name or PRIMARY_TOOL_NAME):
                continue
            arguments = function.get("arguments", "{}")
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except Exception:
                    arguments = {}
            if not isinstance(arguments, dict):
                continue
            for key in ("actions_list", "actions"):
                values = arguments.get(key)
                if isinstance(values, list):
                    parsed = sanitize_craftax_actions(values)
                    if parsed:
                        return parsed
    return []


def extract_qwen_tool_calls(payload: dict, *, tool_name: str = PRIMARY_TOOL_NAME) -> list[dict]:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return []
    message = choices[0].get("message", {})
    if not isinstance(message, dict):
        return []

    parsed_calls: list[dict] = []
    tool_calls = message.get("tool_calls")
    if isinstance(tool_calls, list):
        for tool_call in tool_calls:
            if not isinstance(tool_call, dict):
                continue
            function = tool_call.get("function", {})
            if not isinstance(function, dict):
                continue
            name = str(function.get("name") or "").strip()
            arguments = function.get("arguments", {})
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except Exception:
                    arguments = {}
            if name and isinstance(arguments, dict):
                if name != (tool_name or PRIMARY_TOOL_NAME):
                    continue
                parsed_calls.append({"name": name, "arguments": arguments})
        if parsed_calls:
            return parsed_calls

    content = message.get("content")
    if isinstance(content, str):
        for raw_block in re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>", content, flags=re.DOTALL):
            block = raw_block.strip()
            if not block:
                continue
            try:
                parsed = json.loads(block)
            except Exception:
                parsed = None
            if isinstance(parsed, dict):
                name = str(parsed.get("name") or "").strip()
                arguments = parsed.get("arguments", {})
                if name and isinstance(arguments, dict):
                    if name != (tool_name or PRIMARY_TOOL_NAME):
                        continue
                    parsed_calls.append({"name": name, "arguments": arguments})

    if parsed_calls:
        return parsed_calls
    return []


def extract_openai_tool_calls(payload: dict, *, tool_name: str = PRIMARY_TOOL_NAME) -> list[dict]:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return []
    message = choices[0].get("message", {})
    if not isinstance(message, dict):
        return []

    parsed_calls: list[dict] = []
    tool_calls = message.get("tool_calls")
    if isinstance(tool_calls, list):
        for tool_call in tool_calls:
            if not isinstance(tool_call, dict):
                continue
            function = tool_call.get("function", {})
            if not isinstance(function, dict):
                continue
            name = str(function.get("name") or "").strip()
            if name != (tool_name or PRIMARY_TOOL_NAME):
                continue
            arguments = function.get("arguments", {})
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except Exception:
                    arguments = {}
            if name and isinstance(arguments, dict):
                parsed_calls.append({"name": name, "arguments": arguments})
        if parsed_calls:
            return parsed_calls

    content = message.get("content")
    if isinstance(content, str):
        for raw_block in re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>", content, flags=re.DOTALL):
            block = raw_block.strip()
            if not block:
                continue
            try:
                parsed = json.loads(block)
            except Exception:
                parsed = None
            if not isinstance(parsed, dict):
                continue
            name = str(parsed.get("name") or "").strip()
            if name != (tool_name or PRIMARY_TOOL_NAME):
                continue
            arguments = parsed.get("arguments", {})
            if name and isinstance(arguments, dict):
                parsed_calls.append({"name": name, "arguments": arguments})
    return parsed_calls
