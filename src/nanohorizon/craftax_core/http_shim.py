"""Thin HTTP-facing shim for Craftax prompt shaping.

This file keeps the surface small and reviewable. It does not change the
underlying Craftax environment contract; it only shapes the model-facing
payload into semantic fields plus a bounded reward history.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping

from fastapi import FastAPI

from .metadata import (
    PRIMARY_TOOL_NAME,
    PromptContext,
    RewardHistoryEntry,
    RewardHistoryWindow,
    StructuredObservation,
)


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


def render_prompt_turn_text(
    observation: Any,
    history: Iterable[Mapping[str, Any] | RewardHistoryEntry] = (),
    *,
    metadata: Mapping[str, Any] | None = None,
    target_action_batch_size: int = 8,
) -> str:
    """Return the prompt as a compact natural-language turn description."""

    context = build_prompt_context(observation, history, metadata=metadata)
    payload = context.to_prompt_payload()
    observation_text = str(observation).strip() or "No text renderer available."
    lines = [
        "Current Craftax long-horizon observation:",
        observation_text,
    ]

    structured_summary = str(payload.get("structured_observation_summary") or "").strip()
    if structured_summary:
        lines.extend(["Structured observation summary:", structured_summary])

    reward_history = payload.get("reward_history")
    if isinstance(reward_history, list) and reward_history:
        lines.append("Recent reward history:")
        for entry in reward_history:
            if not isinstance(entry, dict):
                continue
            lines.append(
                "- action={action} reward={reward_delta:+.2f} sign={sign} obs={observation_summary}".format(
                    action=str(entry.get("action") or "").strip() or "noop",
                    reward_delta=float(entry.get("reward_delta", 0.0)),
                    sign=str(entry.get("sign") or "neutral"),
                    observation_summary=str(entry.get("observation_summary") or "").strip() or "none",
                )
            )

        summary = payload.get("reward_history_summary")
        if isinstance(summary, dict):
            lines.append(
                "History summary: "
                f"net_reward={float(summary.get('net_reward', 0.0)):+.2f}, "
                f"positive_steps={int(summary.get('positive_steps', 0))}, "
                f"negative_steps={int(summary.get('negative_steps', 0))}, "
                f"repeat_action_streak={int(summary.get('repeat_action_streak', 0))}, "
                f"trailing_negative_streak={int(summary.get('trailing_negative_streak', 0))}"
            )

        advice = str(payload.get("reward_history_advice") or "").strip()
        if advice:
            lines.append(f"Trajectory advice: {advice}")

    lines.extend(
        [
            "",
            f"Plan a short useful macro-action. Use the {PRIMARY_TOOL_NAME} tool exactly once.",
            f"Return exactly {max(1, int(target_action_batch_size))} actions unless the environment is already done.",
            "Use only valid full-Craftax actions.",
            "Do not return JSON or plain text actions.",
        ]
    )
    return "\n".join(lines)


def summarize_history(history: RewardHistoryWindow | Iterable[RewardHistoryEntry]) -> list[dict[str, Any]]:
    if isinstance(history, RewardHistoryWindow):
        return history.to_prompt_payload()
    return [entry.to_prompt_payload() for entry in history]


def run_rollout_request(request: dict[str, Any]) -> dict[str, Any]:
    from .rollout import run_rollout_request as _run_rollout_request

    return _run_rollout_request(request)


def create_app() -> FastAPI:
    app = FastAPI()

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "upstream_ready": True}

    @app.get("/task_info")
    def task_info() -> dict[str, Any]:
        return {
            "env_kind": "full",
            "primary_tool_name": PRIMARY_TOOL_NAME,
        }

    @app.post("/rollout")
    def rollout(request: dict[str, Any]) -> dict[str, Any]:
        return run_rollout_request(request)

    return app
