from __future__ import annotations

from .http_shim import build_craftax_rollout_context
from .metadata import (
    CraftaxCandidateMetadata,
    CraftaxStepRecord,
    WorkingMemoryBuffer,
    normalize_resource_state,
)

__all__ = [
    "CraftaxCandidateMetadata",
    "CraftaxHarnessRunner",
    "CraftaxStepInput",
    "CraftaxStepRecord",
    "WorkingMemoryBuffer",
    "build_craftax_rollout_context",
    "normalize_resource_state",
]


def __getattr__(name: str):
    if name in {"CraftaxHarnessRunner", "CraftaxStepInput"}:
        from .runner import CraftaxHarnessRunner, CraftaxStepInput

        return {
            "CraftaxHarnessRunner": CraftaxHarnessRunner,
            "CraftaxStepInput": CraftaxStepInput,
        }[name]
    raise AttributeError(name)

