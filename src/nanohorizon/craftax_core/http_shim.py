"""Thin HTTP-facing shim for Craftax prompt shaping.

This file keeps the surface small and reviewable. It does not change the
underlying Craftax environment contract; it only shapes the model-facing
payload into semantic fields plus a bounded reward history.
"""

from __future__ import annotations

import os
from typing import Any, Iterable, Mapping

from fastapi import FastAPI

from .metadata import (
    PromptContext,
    RewardHistoryEntry,
    RewardHistoryWindow,
    StructuredObservation,
    craftax_achievement_context,
)

run_rollout_request: Any | None = None


def build_prompt_context(
    observation: Any,
    history: Iterable[Mapping[str, Any] | RewardHistoryEntry] = (),
    *,
    metadata: Mapping[str, Any] | None = None,
) -> PromptContext:
    structured = StructuredObservation.from_observation(observation)
    window = RewardHistoryWindow()
    metadata_dict = dict(metadata or {})
    unlocked_values = (
        metadata_dict.get("achievements_unlocked")
        or metadata_dict.get("unique_achievements")
        or metadata_dict.get("achievements")
        or []
    )
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
    achievement_context = craftax_achievement_context(unlocked_values)
    next_targets_values = metadata_dict.get("next_targets")
    if isinstance(next_targets_values, (str, bytes, bytearray)):
        next_targets_values = [next_targets_values]
    if next_targets_values:
        achievement_context["next_targets"] = [
            str(item).strip() for item in next_targets_values if str(item).strip()
        ]
    if metadata_dict.get("unique_achievement_count") is not None:
        try:
            achievement_context["unique_achievement_count"] = int(metadata_dict["unique_achievement_count"])
        except (TypeError, ValueError):
            pass
    if metadata_dict.get("achievement_score_rule"):
        achievement_context["achievement_score_rule"] = str(metadata_dict["achievement_score_rule"])
    return PromptContext(
        observation=structured,
        reward_history=window,
        achievements_unlocked=list(achievement_context["achievements_unlocked"]),
        next_targets=list(achievement_context["next_targets"]),
        unique_achievement_count=int(achievement_context["unique_achievement_count"]),
        achievement_score_rule=str(achievement_context["achievement_score_rule"]),
        metadata=metadata_dict,
    )


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


def create_app() -> FastAPI:
    app = FastAPI()

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "service": "craftax_core_http_shim",
            "upstream_ready": True,
        }

    @app.get("/task_info")
    def task_info() -> dict[str, Any]:
        return {
            "service": "craftax_core_http_shim",
            "env_kind": "full",
            "tool_name": "craftax_interact",
        }

    @app.post("/rollout")
    def rollout(request: dict[str, Any]) -> dict[str, Any]:
        global run_rollout_request
        if not callable(run_rollout_request):
            from .rollout import run_rollout_request as _run_rollout_request

            run_rollout_request = _run_rollout_request

        return run_rollout_request(request)

    return app


def main() -> int:
    import uvicorn

    host = str(os.getenv("NANOHORIZON_CRAFTAX_BIND_HOST") or "127.0.0.1")
    port = int(os.getenv("NANOHORIZON_CRAFTAX_BIND_PORT") or 8913)
    workers = int(os.getenv("NANOHORIZON_CRAFTAX_UVICORN_WORKERS") or 1)
    uvicorn.run(create_app(), host=host, port=port, workers=workers, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
