from __future__ import annotations

import os
import uuid
from typing import Any

from fastapi import FastAPI, HTTPException

from .checkpoint import Checkpoint, CheckpointCodec
from .modalities import RenderMode
from .rollout import collect_rollouts, prewarm_one_step, run_rollout_request
from .texture_cache import ensure_texture_cache
from .upstream import make_runner


class _Store:
    def __init__(self) -> None:
        self.rollouts: dict[str, dict[str, Any]] = {}
        self.checkpoints: dict[str, bytes] = {}


STORE = _Store()
_RENDERER_PREWARMED = False


def _prewarm_full_renderer() -> dict[str, Any]:
    global _RENDERER_PREWARMED
    if _RENDERER_PREWARMED:
        return {"status": "ok", "prewarmed": True, "cached": True}
    if str(os.getenv("NANOHORIZON_CRAFTAX_PREWARM_RENDERER") or "1").strip().lower() in {
        "0",
        "false",
        "no",
    }:
        return {"status": "skipped", "prewarmed": False}
    runner = make_runner(kind="full", seed=0, render_mode=RenderMode.BOTH, block_pixel_size=16)
    output = runner.reset()
    runner.step(0)
    prewarm_one_step(env_kind="full", render_mode=RenderMode.TEXT)
    frame_shape = None
    if output.render.pixels is not None:
        frame_shape = list(output.render.pixels.shape)
    _RENDERER_PREWARMED = True
    return {"status": "ok", "prewarmed": True, "frame_shape": frame_shape}


def _ensure_runtime_ready() -> dict[str, Any]:
    texture = ensure_texture_cache()
    renderer = _prewarm_full_renderer()
    return {"texture_cache": texture, "renderer": renderer}


def create_app() -> FastAPI:
    app = FastAPI(title="nanohorizon-craftax-core")

    @app.get("/")
    def root() -> dict[str, Any]:
        return {"status": "ok", "service": "craftax_core_http_shim"}

    @app.get("/health")
    def health() -> dict[str, Any]:
        ready = _ensure_runtime_ready()
        return {
            "status": "ok",
            "service": "craftax_core_http_shim",
            "upstream_ready": True,
            **ready,
        }

    @app.get("/done")
    def done() -> dict[str, Any]:
        return {"ok": True, "service": "craftax_core_http_shim"}

    @app.get("/info")
    def info() -> dict[str, Any]:
        return {
            "id": "craftax_core_http_shim",
            "name": "Craftax Core HTTP Shim",
            "description": "Python Craftax runtime with deterministic rollout support.",
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
        rollout = STORE.rollouts.get(rollout_id)
        if rollout is None:
            raise HTTPException(status_code=404, detail="unknown rollout_id")
        checkpoint_id = str(request.get("checkpoint_id") or f"checkpoint_{uuid.uuid4().hex}")
        checkpoint = Checkpoint(
            version=1,
            seed=int(rollout.get("metadata", {}).get("seed", 0)),
            episode_index=0,
            step_index=int(len(rollout.get("metadata", {}).get("action_history", []))),
            next_rng=None,
            state=rollout,
            metadata={"source": "shim_rollout_snapshot"},
        )
        STORE.checkpoints[checkpoint_id] = CheckpointCodec.dumps(checkpoint)
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
        checkpoint = CheckpointCodec.loads(STORE.checkpoints[checkpoint_id])
        if not isinstance(checkpoint.state, dict):
            raise HTTPException(status_code=400, detail="checkpoint is not resumable")
        return checkpoint.state

    @app.post("/rollouts/{rollout_id}/terminate")
    def terminate_rollout(rollout_id: str) -> dict[str, Any]:
        STORE.rollouts.pop(rollout_id, None)
        return {"status": "terminated", "rollout_id": rollout_id}

    return app


app = create_app()


def main() -> None:
    import uvicorn

    runtime_report = _ensure_runtime_ready()
    print(f"Craftax runtime ready: {runtime_report}", flush=True)
    host = os.getenv("NANOHORIZON_CRAFTAX_BIND_HOST") or "127.0.0.1"
    port = int(os.getenv("NANOHORIZON_CRAFTAX_BIND_PORT") or "8903")
    workers = max(1, int(os.getenv("NANOHORIZON_CRAFTAX_UVICORN_WORKERS") or "1"))
    if workers > 1:
        uvicorn.run(
            "nanohorizon.craftax_core.http_shim:app",
            host=str(host),
            port=port,
            log_level="info",
            workers=workers,
        )
        return
    uvicorn.run(app, host=str(host), port=port, log_level="info")


if __name__ == "__main__":
    main()
