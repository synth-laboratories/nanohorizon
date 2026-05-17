"""Thin HTTP-facing shim for Craftax prompt shaping.

This file keeps the surface small and reviewable. It does not change the
underlying Craftax environment contract; it only shapes the model-facing
payload into semantic fields plus a bounded reward history.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping

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
