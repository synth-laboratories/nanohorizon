#!/usr/bin/env python3
"""Verifier for the Craftax video-validation scaffold."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from nanohorizon.craftax_core.metadata import (  # noqa: E402
    CANDIDATE_LABEL,
    PRESERVED_HARNESS_SURFACES,
    SCRATCHPAD_PATH,
    build_candidate_metadata,
)
from nanohorizon.craftax_core.runner import build_runner_summary  # noqa: E402


def main() -> int:
    metadata = build_candidate_metadata()
    assert metadata.candidate_label == CANDIDATE_LABEL
    assert metadata.harness_surfaces == PRESERVED_HARNESS_SURFACES
    assert SCRATCHPAD_PATH.exists(), f"missing scratchpad: {SCRATCHPAD_PATH}"
    summary = build_runner_summary()
    assert summary["scratchpad_present"], "scratchpad content was not detected"
    assert summary["payload"]["candidate"]["candidate_label"] == CANDIDATE_LABEL
    for surface in PRESERVED_HARNESS_SURFACES:
        assert Path(surface).exists(), f"missing preserved surface: {surface}"
    print("verification: passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

