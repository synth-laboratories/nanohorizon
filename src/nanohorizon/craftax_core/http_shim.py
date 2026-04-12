"""Thin HTTP-facing shim for Craftax prompt shaping.

This file keeps the surface small and reviewable. It does not change the
underlying Craftax environment contract; it only shapes the model-facing
payload into semantic fields plus a bounded reward history.
"""

from __future__ import annotations

import os
from typing import Any, Iterable, Mapping

from fastapi import FastAPI, HTTPException, Request
import uvicorn

from .metadata import PromptContext, RewardHistoryEntry, RewardHistoryWindow, StructuredObservation


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


def run_rollout_request(request: Mapping[str, Any]) -> dict[str, Any]:
    from .rollout import run_rollout_request as _run_rollout_request

    return _run_rollout_request(request)


def create_app(*, env_kind: str = "full") -> FastAPI:
    app = FastAPI()

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"upstream_ready": True, "env_kind": env_kind}

    @app.get("/task_info")
    def task_info() -> dict[str, Any]:
        return {"env_kind": env_kind, "task_name": "craftax"}

    @app.post("/rollout")
    @app.post("/rollouts")
    async def rollout(request: Request) -> dict[str, Any]:
        payload = await request.json()
        if not isinstance(payload, dict):
            raise HTTPException(status_code=400, detail="rollout payload must be an object")
        return run_rollout_request(payload)

    return app


def main() -> None:
    host = str(os.getenv("NANOHORIZON_CRAFTAX_BIND_HOST", "127.0.0.1")).strip() or "127.0.0.1"
    port = int(os.getenv("NANOHORIZON_CRAFTAX_BIND_PORT", "8913"))
    workers = int(os.getenv("NANOHORIZON_CRAFTAX_UVICORN_WORKERS", "1"))
    env_kind = str(os.getenv("NANOHORIZON_CRAFTAX_ENV_KIND", "full")).strip() or "full"
    app = create_app(env_kind=env_kind)
    uvicorn.run(app, host=host, port=port, workers=max(1, workers), log_level="info")


if __name__ == "__main__":
    main()
