"""Thin HTTP-facing shim for Craftax prompt shaping.

This file keeps the surface small and reviewable. It does not change the
underlying Craftax environment contract; it only shapes the model-facing
payload into semantic fields plus a bounded reward history.
"""

from __future__ import annotations

import os
from typing import Any, Iterable, Mapping

from fastapi import FastAPI
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


def run_rollout_request(request: dict[str, Any]) -> dict[str, Any]:
    from .rollout import run_rollout_request as _run_rollout_request

    return _run_rollout_request(request)


def create_app() -> FastAPI:
    app = FastAPI(title="nanohorizon-craftax-core-http-shim")

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "upstream_ready": True,
            "service": "craftax_core_http_shim",
        }

    @app.get("/task_info")
    def task_info() -> dict[str, Any]:
        return {
            "env_kind": "full",
            "service": "craftax_core_http_shim",
        }

    @app.post("/rollout")
    def rollout(request: dict[str, Any]) -> dict[str, Any]:
        return run_rollout_request(request)

    @app.post("/rollouts")
    def rollouts(request: dict[str, Any]) -> dict[str, Any]:
        return run_rollout_request(request)

    return app


def main() -> None:
    host = str(os.getenv("NANOHORIZON_CRAFTAX_BIND_HOST") or "127.0.0.1")
    port = int(os.getenv("NANOHORIZON_CRAFTAX_BIND_PORT") or "8000")
    workers = int(os.getenv("NANOHORIZON_CRAFTAX_UVICORN_WORKERS") or "1")
    uvicorn.run(create_app(), host=host, port=port, workers=workers, log_level="info")


if __name__ == "__main__":
    main()
