"""Shared Craftax response builders.

The shim stays intentionally narrow: it exposes the same candidate metadata for
health, task-info, and rollout surfaces so the pipeline uses one source of truth
for the TODO scratchpad.
"""

from __future__ import annotations

from typing import Any, Mapping

from .metadata import CraftaxCandidateMetadata, default_metadata


def _metadata(metadata: CraftaxCandidateMetadata | None) -> CraftaxCandidateMetadata:
    return metadata or default_metadata()


def build_health_payload(
    metadata: CraftaxCandidateMetadata | None = None,
) -> dict[str, Any]:
    resolved = _metadata(metadata)
    return {
        "status": "ok",
        "candidate_label": resolved.candidate_label,
        "primary_strategy": resolved.primary_strategy,
        "todo_count": len(resolved.todo_items),
    }


def build_task_info(
    metadata: CraftaxCandidateMetadata | None = None,
) -> dict[str, Any]:
    resolved = _metadata(metadata)
    return {
        "candidate_label": resolved.candidate_label,
        "primary_strategy": resolved.primary_strategy,
        "objective": resolved.objective,
        "shared_harness_surfaces": list(resolved.shared_harness_surfaces),
        "todo_items": [item.to_dict() for item in resolved.todo_items],
        "todo_block": resolved.todo_block(),
    }


def build_rollout_payload(
    metadata: CraftaxCandidateMetadata | None = None,
    *,
    result: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    resolved = _metadata(metadata)
    return {
        "candidate_label": resolved.candidate_label,
        "primary_strategy": resolved.primary_strategy,
        "status": "ready",
        "result": dict(result or {}),
    }


def craftax_http_response(
    path: str,
    metadata: CraftaxCandidateMetadata | None = None,
) -> dict[str, Any]:
    if path == "/health":
        return build_health_payload(metadata)
    if path == "/task_info":
        return build_task_info(metadata)
    if path in {"/rollouts", "/rollout"}:
        return build_rollout_payload(metadata)
    return {
        "status": "not_found",
        "path": path,
        "available_paths": ["/health", "/task_info", "/rollouts", "/rollout"],
    }

