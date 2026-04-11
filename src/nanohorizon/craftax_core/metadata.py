"""Metadata for the Craftax leaderboard candidate."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict

CANDIDATE_LABEL = "Video Validation Run"
CRAFTAX_SCRATCHPAD_LIMIT = 6
CRAFTAX_METADATA_VERSION = "1.0"
SCRATCHPAD_PATH = Path(".craftax_todo.json")
PRESERVED_HARNESS_SURFACES = (
    "docs/task-craftax.md",
    "src/nanohorizon/craftax_core/http_shim.py",
    "src/nanohorizon/craftax_core/runner.py",
    "src/nanohorizon/craftax_core/metadata.py",
    "scripts/run_craftax_model_eval.sh",
)


@dataclass(frozen=True, slots=True)
class CraftaxRunMetadata:
    """Stable metadata surfaced by the runner and evaluation script."""

    candidate_label: str = CANDIDATE_LABEL
    metadata_version: str = CRAFTAX_METADATA_VERSION
    scratchpad_limit: int = CRAFTAX_SCRATCHPAD_LIMIT
    harness_surfaces: tuple[str, ...] = PRESERVED_HARNESS_SURFACES

    @property
    def preserved_harness_surfaces(self) -> tuple[str, ...]:
        return self.harness_surfaces

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["harness_surfaces"] = list(self.harness_surfaces)
        payload["preserved_harness_surfaces"] = list(self.harness_surfaces)
        return payload


def build_candidate_metadata() -> CraftaxRunMetadata:
    return CraftaxRunMetadata()
