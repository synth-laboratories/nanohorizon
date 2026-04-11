"""HTTP-shaped helpers for the Server Push E2E candidate."""

from __future__ import annotations

from .metadata import CraftaxCandidateMetadata, build_server_push_e2e_metadata


def build_task_info(
    metadata: CraftaxCandidateMetadata | None = None,
) -> dict[str, object]:
    candidate = metadata or build_server_push_e2e_metadata()
    return {
        "health": "ok",
        "task_info": candidate.to_task_info(),
    }


def build_health_payload() -> dict[str, str]:
    return {"health": "ok"}

