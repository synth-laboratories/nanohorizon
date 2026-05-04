#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib import request

DEFAULT_SHARED_CONTEXT_KEY = "runtime_resources.craftax"
DEFAULT_RESOURCE_ID = "craftax_rollout_http"


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_json_arg(raw: str) -> dict[str, Any]:
    value = raw.strip()
    if value.startswith("@"):
        value = Path(value[1:]).read_text(encoding="utf-8")
    payload = json.loads(value)
    if not isinstance(payload, dict):
        raise RuntimeError("expected JSON object")
    return payload


def _probe_health(base_url: str, timeout_seconds: float) -> dict[str, Any]:
    url = f"{base_url.rstrip('/')}/health"
    with request.urlopen(url, timeout=timeout_seconds) as response:
        body = response.read().decode("utf-8", errors="replace")
        return {
            "url": url,
            "status_code": int(response.status),
            "ok": 200 <= int(response.status) < 300,
            "body_preview": body[:1000],
        }


def _nanohorizon_import_proof() -> dict[str, Any]:
    modules = [
        "nanohorizon.shared.craftax_data",
        "nanohorizon.craftax_core.runner",
    ]
    results: dict[str, Any] = {}
    for module in modules:
        spec = importlib.util.find_spec(module)
        results[module] = {
            "importable": spec is not None,
            "origin": spec.origin if spec is not None else None,
        }
    return results


def _base_handle(*, mode: str) -> dict[str, Any]:
    return {
        "resource_id": DEFAULT_RESOURCE_ID,
        "ownership": "actor_managed",
        "mode": mode,
        "shared_context_key": DEFAULT_SHARED_CONTEXT_KEY,
        "created_at": _utc_now(),
        "created_for_run_id": os.getenv("SMR_RUN_ID") or os.getenv("SYNTH_RUN_ID"),
        "created_by_actor_id": os.getenv("SMR_ACTOR_ID") or os.getenv("SYNTH_ACTOR_ID"),
    }


def _safe_call(label: str, func: Any, *args: Any) -> Any:
    try:
        return func(*args)
    except Exception as exc:  # noqa: BLE001
        return {"error": repr(exc), "label": label}


def local_proof(args: argparse.Namespace) -> int:
    explicit_url = str(os.getenv("NANOHORIZON_CRAFTAX_CONTAINER_URL") or "").strip()
    allow_direct = str(os.getenv("NANOHORIZON_ALLOW_DIRECT_LOCAL") or "").strip() == "1"
    if explicit_url:
        health = _probe_health(explicit_url, args.timeout_seconds)
        handle = _base_handle(mode="local_http" if explicit_url.startswith("http") else "explicit_url")
        handle.update(
            {
                "url": explicit_url,
                "health": health,
                "ready": bool(health.get("ok")),
            }
        )
    elif allow_direct:
        import_proof = _nanohorizon_import_proof()
        handle = _base_handle(mode="local_direct")
        handle.update(
            {
                "url": "direct://local",
                "import_proof": import_proof,
                "ready": all(item["importable"] for item in import_proof.values()),
                "note": (
                    "Direct local mode is valid only for same-actor proof runs. "
                    "Use a Synth container-pool handle when another actor or reviewer must consume it."
                ),
            }
        )
    else:
        raise RuntimeError(
            "No actor-managed Craftax resource selected. Start a local HTTP service and set "
            "NANOHORIZON_CRAFTAX_CONTAINER_URL, or set NANOHORIZON_ALLOW_DIRECT_LOCAL=1 for "
            "same-actor local proof."
        )
    _write_json(Path(args.output), handle)
    print(json.dumps(handle, indent=2, sort_keys=True))
    return 0


def create_pool(args: argparse.Namespace) -> int:
    from synth_ai import SynthClient

    client = SynthClient(base_url=args.backend_url)
    pool_request = _read_json_arg(args.request_json)
    pool = client.pools.create(pool_request)
    pool_id = str(pool.get("id") or pool.get("pool_id") or "").strip()
    if not pool_id:
        raise RuntimeError(f"pool create response did not include id/pool_id: {pool}")
    handle = _base_handle(mode="synth_container_pool")
    handle.update(
        {
            "pool_id": pool_id,
            "pool": pool,
            "urls": _safe_call("get_urls", client.pools.get_urls, pool_id),
            "health": _safe_call("get_pool_container_health", client.pools.get_pool_container_health, pool_id),
            "info": _safe_call("get_pool_container_info", client.pools.get_pool_container_info, pool_id),
            "metadata": _safe_call("get_pool_container_metadata", client.pools.get_pool_container_metadata, pool_id),
            "ready": True,
        }
    )
    _write_json(Path(args.output), handle)
    print(json.dumps(handle, indent=2, sort_keys=True))
    return 0


def pool_readback(args: argparse.Namespace) -> int:
    from synth_ai import SynthClient

    client = SynthClient(base_url=args.backend_url)
    handle = _base_handle(mode="synth_container_pool")
    if args.task_id:
        health = _safe_call("get_task_container_health", client.pools.get_task_container_health, args.pool_id, args.task_id)
        info = _safe_call("get_task_container_info", client.pools.get_task_container_info, args.pool_id, args.task_id)
        metadata = _safe_call("get_task_container_metadata", client.pools.get_task_container_metadata, args.pool_id, args.task_id)
    else:
        health = _safe_call("get_pool_container_health", client.pools.get_pool_container_health, args.pool_id)
        info = _safe_call("get_pool_container_info", client.pools.get_pool_container_info, args.pool_id)
        metadata = _safe_call("get_pool_container_metadata", client.pools.get_pool_container_metadata, args.pool_id)
    handle.update(
        {
            "pool_id": args.pool_id,
            "task_id": args.task_id,
            "pool": _safe_call("get", client.pools.get, args.pool_id),
            "urls": _safe_call("get_urls", client.pools.get_urls, args.pool_id),
            "health": health,
            "info": info,
            "metadata": metadata,
            "ready": str(health.get("status") or "").lower() in {"healthy", "ready", "ok"},
        }
    )
    _write_json(Path(args.output), handle)
    print(json.dumps(handle, indent=2, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Manage actor-owned Craftax runtime resource evidence.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    local = subparsers.add_parser("local-proof")
    local.add_argument("--output", default="artifacts/container_proof.json")
    local.add_argument("--timeout-seconds", type=float, default=5.0)
    local.set_defaults(func=local_proof)

    create = subparsers.add_parser("create-pool")
    create.add_argument("--request-json", required=True, help="JSON object or @path")
    create.add_argument("--backend-url", default=None)
    create.add_argument("--output", default="artifacts/container_proof.json")
    create.set_defaults(func=create_pool)

    readback = subparsers.add_parser("pool-readback")
    readback.add_argument("--pool-id", required=True)
    readback.add_argument("--task-id", default=None)
    readback.add_argument("--backend-url", default=None)
    readback.add_argument("--output", default="artifacts/container_proof.json")
    readback.set_defaults(func=pool_readback)

    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())

