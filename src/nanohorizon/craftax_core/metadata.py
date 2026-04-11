"""Metadata for the NanoHorizon Craftax candidate.

This module keeps the TODO scratchpad compact and explicit so the pipeline
surface can reuse the same candidate context across health, task-info, and
rollout responses.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any


@dataclass(frozen=True, slots=True)
class CraftaxTodoItem:
    """Single scratchpad item for the candidate pipeline."""

    title: str
    rationale: str
    status: str = "pending"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class CraftaxCandidateMetadata:
    """Stable metadata bundle for the candidate task."""

    candidate_label: str = "Pipeline Fix E2E"
    primary_strategy: str = "Todo Tool"
    objective: str = (
        "Build a strong NanoHorizon leaderboard candidate inside the NanoHorizon repo."
    )
    shared_harness_surfaces: tuple[str, ...] = (
        "docs/task-craftax.md",
        "src/nanohorizon/craftax_core/http_shim.py",
        "src/nanohorizon/craftax_core/runner.py",
        "src/nanohorizon/craftax_core/metadata.py",
        "scripts/run_craftax_model_eval.sh",
    )
    todo_items: tuple[CraftaxTodoItem, ...] = (
        CraftaxTodoItem(
            title="Keep shared harness surfaces stable",
            rationale="Avoid unnecessary churn in the benchmark contract and review surface.",
        ),
        CraftaxTodoItem(
            title="Use verifier feedback before readiness",
            rationale="Treat validation as a gating step, not a postscript.",
        ),
        CraftaxTodoItem(
            title="Leave a reviewable commit and real PR",
            rationale="Preserve a durable handoff that downstream reviewers can inspect.",
        ),
    )
    notes: str = (
        "Compact scratchpad for a pipeline-focused Craftax candidate. "
        "The TODO list is intentionally short so it stays legible in every surface."
    )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["todo_items"] = [item.to_dict() for item in self.todo_items]
        payload["shared_harness_surfaces"] = list(self.shared_harness_surfaces)
        return payload

    def todo_block(self) -> str:
        lines = ["TODO"]
        for index, item in enumerate(self.todo_items, start=1):
            lines.append(f"{index}. {item.title} - {item.rationale}")
        return "\n".join(lines)


def default_metadata() -> CraftaxCandidateMetadata:
    return CraftaxCandidateMetadata()

