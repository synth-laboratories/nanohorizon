"""HTTP-shaped shim for the Craftax rollout contract."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Mapping

from .metadata import CRAFTAX_SURFACES, TODO_TOOL_STRATEGY, build_default_todo_items


@dataclass(frozen=True)
class CraftaxHTTPShim:
    """Tiny in-process stand-in for the Craftax HTTP contract.

    The shim intentionally keeps the public surface close to the rollout
    contract described in the task instructions: ``/health``, ``/task_info``,
    ``/rollouts``, and the compatibility alias ``/rollout``.
    """

    candidate_label: str = "Daytona E2E Run 3"

    def health(self) -> Dict[str, Any]:
        return {"ok": True, "service": "nanohorizon-craftax", "candidate_label": self.candidate_label}

    def task_info(self) -> Dict[str, Any]:
        return {
            "candidate_label": self.candidate_label,
            "strategy": TODO_TOOL_STRATEGY,
            "stable_surfaces": [asdict(surface) for surface in CRAFTAX_SURFACES],
            "todo_items": [asdict(item) for item in build_default_todo_items()],
        }

    def rollouts(self, payload: Mapping[str, Any] | None = None) -> Dict[str, Any]:
        return {"accepted": True, "route": "/rollouts", "payload": dict(payload or {})}

    def rollout(self, payload: Mapping[str, Any] | None = None) -> Dict[str, Any]:
        return self.rollouts(payload)

    def info(self) -> Dict[str, Any]:
        return {"candidate_label": self.candidate_label, "strategy": TODO_TOOL_STRATEGY}
