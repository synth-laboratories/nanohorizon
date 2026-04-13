"""Thin HTTP-facing shim for Craftax prompt shaping.

This file keeps the surface small and reviewable. It does not change the
underlying Craftax environment contract; it only shapes the model-facing
payload into semantic fields plus a bounded reward history.
"""

from __future__ import annotations

import os
import uuid
from typing import Any, Iterable, Mapping

from fastapi import FastAPI, HTTPException

from .checkpoint import Checkpoint, CheckpointCodec
from .metadata import PromptContext, RewardHistoryEntry, RewardHistoryWindow, StructuredObservation
from .rollout import collect_rollouts, run_rollout_request
from .texture_cache import ensure_texture_cache


class _Store:
    def __init__(self) -> None:
        self.rollouts: dict[str, dict[str, Any]] = {}
        self.checkpoints: dict[str, bytes] = {}


STORE = _Store()


def create_app() -> FastAPI:
    app = FastAPI(title="nanohorizon-craftax-core")

    @app.get("/")
    def root() -> dict[str, Any]:
        return {"status": "ok", "service": "craftax_core_http_shim"}

    @app.get("/health")
    def health() -> dict[str, Any]:
        texture = ensure_texture_cache()
        return {
            "status": "ok",
            "service": "craftax_core_http_shim",
            "upstream_ready": True,
            "texture_cache": texture,
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

    texture_report = ensure_texture_cache()
    print(f"Craftax texture cache ready: {texture_report}", flush=True)
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


def build_prompt_context(
    observation: Any,
    history: Iterable[Mapping[str, Any] | RewardHistoryEntry] = (),
    *,
    metadata: Mapping[str, Any] | None = None,
) -> PromptContext:
    structured = StructuredObservation.from_observation(observation)
    window = RewardHistoryWindow()
    for item in history:
        if isinstance(item, RewardHistoryEntry):
            entry = item
        else:
            entry = RewardHistoryEntry(
                action=item.get("action"),
                observation_summary=str(item.get("observation_summary", "")),
                reward_delta=float(item.get("reward_delta", 0.0)),
            )
        window.append(entry)
    return PromptContext(observation=structured, reward_history=window, metadata=dict(metadata or {}))


def render_prompt_turn(
    observation: Any,
    history: Iterable[Mapping[str, Any] | RewardHistoryEntry] = (),
    *,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a prompt-ready JSON payload with explicit semantic fields."""

    context = build_prompt_context(observation, history, metadata=metadata)
    return context.to_prompt_payload()


def summarize_history(history: RewardHistoryWindow | Iterable[RewardHistoryEntry]) -> list[dict[str, Any]]:
    if isinstance(history, RewardHistoryWindow):
        return history.to_prompt_payload()
    return [entry.to_prompt_payload() for entry in history]
