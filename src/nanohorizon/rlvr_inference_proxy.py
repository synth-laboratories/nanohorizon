from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException, Request

UPSTREAM_BASE_URL = str(os.getenv("NANOHORIZON_RLVR_PROXY_UPSTREAM_BASE_URL") or "").strip().rstrip("/")
PROXY_API_KEY = str(os.getenv("NANOHORIZON_RLVR_PROXY_API_KEY") or "").strip()
SERVED_MODEL_NAME = (
    str(os.getenv("NANOHORIZON_RLVR_PROXY_SERVED_MODEL_NAME") or "").strip() or "qwen35-4b-rlvr"
)
DEFAULT_REQUEST_MODEL = (
    str(os.getenv("NANOHORIZON_RLVR_PROXY_DEFAULT_REQUEST_MODEL") or "").strip() or SERVED_MODEL_NAME
)
STATE_PATH = Path(
    str(os.getenv("NANOHORIZON_RLVR_PROXY_STATE_PATH") or "/tmp/nanohorizon_rlvr_proxy_state.json")
).expanduser()

@asynccontextmanager
async def lifespan(_: FastAPI) -> Any:
    _write_state(_default_state())
    yield


app = FastAPI(title="nanohorizon-rlvr-inference-proxy", lifespan=lifespan)


def _default_state() -> dict[str, Any]:
    return {
        "served_model_name": SERVED_MODEL_NAME,
        "active_request_model": DEFAULT_REQUEST_MODEL,
        "active_lora_name": None,
        "active_policy_version": "bootstrap",
    }


def _write_state(payload: dict[str, Any]) -> dict[str, Any]:
    normalized = {
        "served_model_name": str(payload.get("served_model_name") or SERVED_MODEL_NAME),
        "active_request_model": str(payload.get("active_request_model") or DEFAULT_REQUEST_MODEL),
        "active_lora_name": (
            str(payload.get("active_lora_name")).strip() if payload.get("active_lora_name") is not None else None
        ),
        "active_policy_version": str(payload.get("active_policy_version") or "bootstrap"),
    }
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(normalized, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return normalized


def _read_state() -> dict[str, Any]:
    if not STATE_PATH.exists():
        return _write_state(_default_state())
    try:
        payload = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return _write_state(_default_state())
    if not isinstance(payload, dict):
        return _write_state(_default_state())
    return _write_state(payload)


def _require_auth(request: Request) -> None:
    if not PROXY_API_KEY:
        return
    raw = str(request.headers.get("authorization") or "").strip()
    expected = f"Bearer {PROXY_API_KEY}"
    if raw != expected:
        raise HTTPException(status_code=401, detail="invalid authorization")


async def _upstream_health() -> tuple[bool, str]:
    if not UPSTREAM_BASE_URL:
        return False, "missing upstream base url"
    health_base_url = UPSTREAM_BASE_URL.removesuffix("/v1")
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{health_base_url}/health")
        if response.status_code == 200:
            return True, "ok"
        return False, f"upstream /health returned HTTP {response.status_code}"
    except Exception as exc:
        return False, repr(exc)


@app.get("/health")
async def health() -> dict[str, Any]:
    ready, detail = await _upstream_health()
    return {
        "status": "ok" if ready else "starting",
        "upstream_ready": ready,
        "upstream_detail": detail,
        "upstream_base_url": UPSTREAM_BASE_URL,
        **_read_state(),
    }


@app.get("/admin/status")
async def admin_status(request: Request) -> dict[str, Any]:
    _require_auth(request)
    ready, detail = await _upstream_health()
    return {
        "upstream_ready": ready,
        "upstream_detail": detail,
        "upstream_base_url": UPSTREAM_BASE_URL,
        **_read_state(),
    }


@app.post("/admin/reset")
async def admin_reset(request: Request) -> dict[str, Any]:
    _require_auth(request)
    payload = await request.json() if request.headers.get("content-type", "").startswith("application/json") else {}
    policy_version = "bootstrap"
    if isinstance(payload, dict):
        policy_version = str(payload.get("policy_version") or "bootstrap")
    state = _write_state(
        {
            "served_model_name": SERVED_MODEL_NAME,
            "active_request_model": SERVED_MODEL_NAME,
            "active_lora_name": None,
            "active_policy_version": policy_version,
        }
    )
    return {"status": "ok", **state}


@app.post("/admin/load_adapter")
async def admin_load_adapter(request: Request) -> Any:
    _require_auth(request)
    payload = await request.json()
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="body must be a JSON object")
    lora_name = str(payload.get("lora_name") or "").strip()
    lora_path = str(payload.get("lora_path") or "").strip()
    policy_version = str(payload.get("policy_version") or lora_name or "policy").strip()
    if not lora_name or not lora_path:
        raise HTTPException(status_code=400, detail="lora_name and lora_path are required")

    headers: dict[str, str] = {}
    if PROXY_API_KEY:
        headers["Authorization"] = f"Bearer {PROXY_API_KEY}"
    body = {
        "lora_name": lora_name,
        "lora_path": lora_path,
        "load_inplace": True,
    }
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{UPSTREAM_BASE_URL}/load_lora_adapter",
                json=body,
                headers=headers,
            )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=repr(exc)) from exc
    if response.status_code >= 400:
        raise HTTPException(status_code=response.status_code, detail=response.text)
    state = _write_state(
        {
            "served_model_name": SERVED_MODEL_NAME,
            "active_request_model": lora_name,
            "active_lora_name": lora_name,
            "active_policy_version": policy_version,
        }
    )
    upstream_payload: Any
    try:
        upstream_payload = response.json()
    except Exception:
        upstream_payload = {"raw_body": response.text}
    return {"status": "ok", "upstream": upstream_payload, **state}


@app.post("/v1/chat/completions")
async def chat_completions(request: Request) -> Any:
    _require_auth(request)
    payload = await request.json()
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="body must be a JSON object")
    state = _read_state()
    active_request_model = str(state.get("active_request_model") or SERVED_MODEL_NAME)
    payload["model"] = active_request_model
    headers: dict[str, str] = {}
    if PROXY_API_KEY:
        headers["Authorization"] = f"Bearer {PROXY_API_KEY}"
    try:
        async with httpx.AsyncClient(timeout=300.0) as client:
            response = await client.post(
                f"{UPSTREAM_BASE_URL}/chat/completions",
                json=payload,
                headers=headers,
            )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=repr(exc)) from exc
    if response.status_code >= 400:
        raise HTTPException(status_code=response.status_code, detail=response.text)
    try:
        return response.json()
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"invalid upstream json: {exc!r}") from exc
