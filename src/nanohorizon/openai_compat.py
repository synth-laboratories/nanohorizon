from __future__ import annotations

import json
import os
from urllib.parse import urlparse
from typing import Any

import httpx


def chat_completion(
    *,
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int = 400,
    temperature: float = 0.2,
    base_url: str | None = None,
    api_key: str | None = None,
) -> str:
    resolved_api_key = str(api_key or os.getenv("OPENAI_API_KEY") or "").strip()
    resolved_base_url = str(base_url or os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
    parsed = urlparse(resolved_base_url)
    is_local = parsed.hostname in {"127.0.0.1", "localhost"}
    if not resolved_api_key and not is_local:
        raise RuntimeError("OPENAI_API_KEY is required for non-local OpenAI-compatible completions")
    headers: dict[str, str] = {}
    if resolved_api_key:
        headers["Authorization"] = f"Bearer {resolved_api_key}"
    with httpx.Client(timeout=60.0) as client:
        response = client.post(
            f"{resolved_base_url}/chat/completions",
            headers=headers,
            json={
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
        )
        response.raise_for_status()
        payload = response.json()
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise RuntimeError(f"unexpected completion payload: {json.dumps(payload)[:300]}")
    message = choices[0].get("message", {})
    return str(message.get("content") or "").strip()
