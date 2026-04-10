from __future__ import annotations

from typing import Any, Mapping

from .metadata import CraftaxCandidateMetadata, WorkingMemoryBuffer, normalize_resource_state


def build_craftax_rollout_context(
    *,
    metadata: CraftaxCandidateMetadata,
    memory: WorkingMemoryBuffer,
    observation: str,
    subgoal: str,
    resource_state: Mapping[str, Any],
    action_plan: str = "",
    outcome: str = "",
) -> dict[str, Any]:
    """Package a compact harness context for an HTTP rollout or prompt payload."""
    record = memory.push(
        subgoal=subgoal,
        resource_state=resource_state,
        action_plan=action_plan,
        outcome=outcome,
        observation=observation,
    )
    return {
        "candidate_label": metadata.candidate_label,
        "strategy": metadata.strategy,
        "prompt_summary": metadata.prompt_summary,
        "observation": observation,
        "latest_record": record.to_dict(),
        "working_memory": memory.snapshot(),
        "working_memory_text": memory.render(),
        "normalized_resource_state": normalize_resource_state(resource_state),
    }

