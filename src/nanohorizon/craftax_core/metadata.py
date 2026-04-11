"""Metadata for the Craftax video-validation candidate."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

CANDIDATE_LABEL = "Final E2E With Video"
EXPERIMENT_ID = "craftax_full_e2e_with_video"
CRAFTAX_METADATA_VERSION = "1.0"
SCRATCHPAD_PATH = Path("experiments") / EXPERIMENT_ID / "todo.json"
PRESERVED_HARNESS_SURFACES = (
    "docs/task-craftax.md",
    "src/nanohorizon/craftax_core/http_shim.py",
    "src/nanohorizon/craftax_core/runner.py",
    "src/nanohorizon/craftax_core/metadata.py",
    "scripts/run_craftax_model_eval.sh",
)


@dataclass(frozen=True, slots=True)
class CraftaxRunMetadata:
    candidate_label: str = CANDIDATE_LABEL
    experiment_id: str = EXPERIMENT_ID
    metadata_version: str = CRAFTAX_METADATA_VERSION
    scratchpad_limit: int = 3
    harness_surfaces: tuple[str, ...] = PRESERVED_HARNESS_SURFACES
    scratchpad_path: Path = SCRATCHPAD_PATH

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["harness_surfaces"] = list(self.harness_surfaces)
        payload["preserved_harness_surfaces"] = list(self.harness_surfaces)
        payload["scratchpad_path"] = str(self.scratchpad_path)
        return payload


def build_candidate_metadata() -> CraftaxRunMetadata:
    return CraftaxRunMetadata()


def build_candidate_manifest() -> dict[str, Any]:
    metadata = build_candidate_metadata().to_dict()
    return {
        "label": metadata["candidate_label"],
        "experiment_id": metadata["experiment_id"],
        "metadata_version": metadata["metadata_version"],
        "output_root": str(SCRATCHPAD_PATH.parent),
        "strategy": "Todo Tool",
        "track_name": "craftax",
        "verification_modes": ["metadata_roundtrip_smoke", "scratchpad_render_smoke"],
    }


def build_candidate_prompt() -> str:
    return (
        "You are a Craftax policy agent. "
        "Keep a tiny private todo list with exactly three items before each tool call. "
        "The three items must track (1) the most urgent danger or blocker, "
        "(2) the next tile, object, or resource target, and "
        "(3) the fallback action that breaks a loop if progress stalls. "
        "Refresh completed todo items every turn. "
        "If the policy repeats the same movement pattern without progress or new information, "
        "replace the stale target item instead of continuing the loop. "
        "Do not reveal the todo list or scratchpad in the final answer."
    )
