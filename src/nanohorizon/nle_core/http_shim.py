from __future__ import annotations

import os
import uuid
from typing import Any

from fastapi import FastAPI, HTTPException

from .metadata import DEFAULT_ACTION_NAMES, PRIMARY_TOOL_NAME
from .rollout import collect_rollouts, run_rollout_request


class _Store:
    def __init__(self) -> None:
        self.rollouts: dict[str, dict[str, Any]] = {}
        self.checkpoints: dict[str, dict[str, Any]] = {}


STORE = _Store()


def _nle_available() -> tuple[bool, str | None]:
    try:
        import nle  # noqa: F401
        import nle.nethack  # noqa: F401

        return True, None
    except Exception as exc:
        return False, str(exc)


def create_app() -> FastAPI:
    app = FastAPI(title="nanohorizon-nle-core")

    @app.get("/")
    def root() -> dict[str, Any]:
        return {"status": "ok", "service": "nle_core_http_shim"}

    @app.get("/health")
    def health() -> dict[str, Any]:
        available, error = _nle_available()
        return {
            "status": "ok" if available else "degraded",
            "service": "nle_core_http_shim",
            "upstream_ready": available,
            "upstream_error": error,
        }

    @app.get("/done")
    def done() -> dict[str, Any]:
        return {"ok": True, "service": "nle_core_http_shim"}

    @app.get("/info")
    def info() -> dict[str, Any]:
        return {
            "id": "nle_core_http_shim",
            "name": "NLE Core HTTP Shim",
            "description": "NetHack Learning Environment runtime with scout-score rollout support.",
        }

    @app.get("/task_info")
    def task_info() -> dict[str, Any]:
        return {
            "id": "nethack_scout",
            "name": "NetHack Scout",
            "description": "Primitive-action NLE task using scout score as the canonical reward.",
            "action_space": "tool_call",
            "environment_family": "nle",
            "task_id": "nethack_scout",
            "tool_name": PRIMARY_TOOL_NAME,
            "num_actions": len(DEFAULT_ACTION_NAMES),
            "actions": DEFAULT_ACTION_NAMES,
        }

    @app.post("/rollout")
    def rollout(request: dict[str, Any]) -> dict[str, Any]:
        payload = run_rollout_request(request)
        STORE.rollouts[str(payload["rollout_id"])] = payload
        return payload

    @app.post("/rollouts")
    def rollouts(request: list[dict[str, Any]] | dict[str, Any]) -> dict[str, Any]:
        requests = request if isinstance(request, list) else [request]
        results, summary = collect_rollouts(requests=requests)
        for payload in results:
            STORE.rollouts[str(payload["rollout_id"])] = payload
        return {"rollouts": results, "summary": summary}

    @app.get("/rollouts/{rollout_id}")
    def get_rollout(rollout_id: str) -> dict[str, Any]:
        payload = STORE.rollouts.get(rollout_id)
        if payload is None:
            raise HTTPException(status_code=404, detail="unknown rollout_id")
        return payload

    @app.post("/rollouts/{rollout_id}/checkpoints")
    def create_checkpoint(rollout_id: str, request: dict[str, Any]) -> dict[str, Any]:
        rollout_payload = STORE.rollouts.get(rollout_id)
        if rollout_payload is None:
            raise HTTPException(status_code=404, detail="unknown rollout_id")
        checkpoint_id = str(request.get("checkpoint_id") or f"checkpoint_{uuid.uuid4().hex}")
        STORE.checkpoints[checkpoint_id] = {
            "rollout_id": rollout_id,
            "metadata": dict(rollout_payload.get("metadata", {})),
        }
        return {"checkpoint_id": checkpoint_id}

    @app.get("/rollouts/{rollout_id}/checkpoints/{checkpoint_id}")
    def get_checkpoint(rollout_id: str, checkpoint_id: str) -> dict[str, Any]:
        payload = STORE.checkpoints.get(checkpoint_id)
        if payload is None or payload.get("rollout_id") != rollout_id:
            raise HTTPException(status_code=404, detail="unknown checkpoint_id")
        return {"checkpoint_id": checkpoint_id, "metadata": payload.get("metadata", {})}

    @app.post("/rollouts/{rollout_id}/resume")
    def resume_rollout(rollout_id: str, request: dict[str, Any]) -> dict[str, Any]:
        checkpoint_id = str(request.get("checkpoint_id") or "").strip()
        checkpoint = STORE.checkpoints.get(checkpoint_id)
        if checkpoint is None or checkpoint.get("rollout_id") != rollout_id:
            raise HTTPException(status_code=404, detail="unknown checkpoint_id")
        payload = STORE.rollouts.get(rollout_id)
        if payload is None:
            raise HTTPException(status_code=404, detail="unknown rollout_id")
        return payload

    @app.post("/rollouts/{rollout_id}/terminate")
    def terminate_rollout(rollout_id: str) -> dict[str, Any]:
        STORE.rollouts.pop(rollout_id, None)
        return {"status": "terminated", "rollout_id": rollout_id}

    return app


app = create_app()


def main() -> None:
    import uvicorn

    host = os.getenv("NANOHORIZON_NLE_BIND_HOST") or "127.0.0.1"
    port = int(os.getenv("NANOHORIZON_NLE_BIND_PORT") or "8913")
    workers = max(1, int(os.getenv("NANOHORIZON_NLE_UVICORN_WORKERS") or "1"))
    if workers > 1:
        uvicorn.run(
            "nanohorizon.nle_core.http_shim:app",
            host=str(host),
            port=port,
            log_level="info",
            workers=workers,
        )
        return
    uvicorn.run(app, host=str(host), port=port, log_level="info")


if __name__ == "__main__":
    main()

