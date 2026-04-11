from __future__ import annotations

import os
import uuid
from typing import Any

from fastapi import FastAPI, HTTPException


class _Store:
    def __init__(self) -> None:
        self.rollouts: dict[str, dict[str, Any]] = {}
        self.checkpoints: dict[str, bytes] = {}


STORE = _Store()


def ensure_texture_cache() -> dict[str, Any]:
    return {"status": "ok", "full": {"ready": True}, "classic": {"ready": True}}


def run_rollout_request(request: dict[str, Any]) -> dict[str, Any]:
    rollout_id = str(request.get("rollout_id") or request.get("trace_correlation_id") or uuid.uuid4())
    return {
        "rollout_id": rollout_id,
        "trace_correlation_id": request.get("trace_correlation_id", rollout_id),
        "trial_id": request.get("trial_id", rollout_id),
        "success_status": "success",
        "reward_info": {"outcome_reward": 0.0, "details": {"achievements": []}},
        "trace": {"inference": {"turns": []}},
        "metadata": {"llm_call_count": 0, "action_history": []},
        "artifact": [{"turns": []}],
    }


def create_app() -> FastAPI:
    app = FastAPI(title="nanohorizon-craftax-core")

    @app.get("/")
    def root() -> dict[str, Any]:
        return {"status": "ok", "service": "craftax_core_http_shim"}

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "service": "craftax_core_http_shim",
            "upstream_ready": True,
            "texture_cache": ensure_texture_cache(),
        }

    @app.get("/task_info")
    def task_info() -> dict[str, Any]:
        return {
            "id": "craftax_full",
            "name": "Craftax Full",
            "description": "Native full-Craftax long-horizon task.",
            "action_space": "tool_call",
            "env_kind": "full",
        }

    @app.post("/rollout")
    def rollout(request: dict[str, Any]) -> dict[str, Any]:
        payload = run_rollout_request(request)
        STORE.rollouts[str(payload["rollout_id"])] = payload
        return payload

    @app.post("/rollouts")
    def rollouts(request: list[dict[str, Any]] | dict[str, Any]) -> dict[str, Any]:
        requests = request if isinstance(request, list) else [request]
        results = [run_rollout_request(item) for item in requests]
        for payload in results:
            STORE.rollouts[str(payload["rollout_id"])] = payload
        return {"rollouts": results, "summary": {"requested_rollouts": len(results)}}

    @app.get("/rollouts/{rollout_id}")
    def get_rollout(rollout_id: str) -> dict[str, Any]:
        payload = STORE.rollouts.get(rollout_id)
        if payload is None:
            raise HTTPException(status_code=404, detail="unknown rollout_id")
        return payload

    @app.post("/rollouts/{rollout_id}/checkpoints")
    def create_checkpoint(rollout_id: str, request: dict[str, Any]) -> dict[str, Any]:
        rollout = STORE.rollouts.get(rollout_id)
        if rollout is None:
            raise HTTPException(status_code=404, detail="unknown rollout_id")
        checkpoint_id = str(request.get("checkpoint_id") or f"checkpoint_{uuid.uuid4().hex}")
        STORE.checkpoints[checkpoint_id] = str(rollout).encode("utf-8")
        return {"checkpoint_id": checkpoint_id}

    @app.get("/rollouts/{rollout_id}/checkpoints/{checkpoint_id}")
    def get_checkpoint(rollout_id: str, checkpoint_id: str) -> dict[str, Any]:
        del rollout_id
        payload = STORE.checkpoints.get(checkpoint_id)
        if payload is None:
            raise HTTPException(status_code=404, detail="unknown checkpoint_id")
        return {"checkpoint_id": checkpoint_id, "size_bytes": len(payload)}

    @app.post("/rollouts/{rollout_id}/resume")
    def resume_rollout(rollout_id: str, request: dict[str, Any]) -> dict[str, Any]:
        del rollout_id
        checkpoint_id = str(request.get("checkpoint_id") or "").strip()
        if not checkpoint_id or checkpoint_id not in STORE.checkpoints:
            raise HTTPException(status_code=404, detail="unknown checkpoint_id")
        return {"checkpoint_id": checkpoint_id, "resumed": True}

    @app.post("/rollouts/{rollout_id}/terminate")
    def terminate_rollout(rollout_id: str) -> dict[str, Any]:
        STORE.rollouts.pop(rollout_id, None)
        return {"status": "terminated", "rollout_id": rollout_id}

    return app


app = create_app()


def main() -> None:
    import uvicorn

    host = os.getenv("NANOHORIZON_CRAFTAX_BIND_HOST") or "127.0.0.1"
    port = int(os.getenv("NANOHORIZON_CRAFTAX_BIND_PORT") or "8903")
    uvicorn.run(app, host=str(host), port=port, log_level="info")


if __name__ == "__main__":
    main()
