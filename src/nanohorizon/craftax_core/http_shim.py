"""Thin HTTP-facing shim for Craftax prompt shaping.

This file keeps the surface small and reviewable. It does not change the
underlying Craftax environment contract; it only shapes the model-facing
payload into semantic fields plus a bounded reward history.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping

from fastapi import FastAPI

from .metadata import PromptContext, RewardHistoryEntry, RewardHistoryWindow, StructuredObservation


def run_rollout_request(request: Mapping[str, Any]) -> dict[str, Any]:
    from .rollout import run_rollout_request as _run_rollout_request

    return _run_rollout_request(request)


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


def create_app(*, env_kind: str = "full") -> FastAPI:
    app = FastAPI()

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "upstream_ready": True, "env_kind": env_kind}

    @app.get("/task_info")
    def task_info() -> dict[str, Any]:
        return {
            "task": "craftax",
            "env_kind": env_kind,
            "rollout_contract": "stable_rollout_http",
        }

    @app.post("/rollout")
    def rollout(request: dict[str, Any]) -> dict[str, Any]:
        return run_rollout_request(request)

    @app.post("/rollouts")
    def rollouts(request: dict[str, Any]) -> dict[str, Any]:
        return run_rollout_request(request)

    return app
