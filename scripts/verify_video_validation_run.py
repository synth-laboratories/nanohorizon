#!/usr/bin/env python3
"""Verifier for the Video Validation Run scaffold."""

from __future__ import annotations

from pathlib import Path

from nanohorizon.craftax_core.metadata import (
    CANDIDATE_LABEL,
    PRESERVED_HARNESS_SURFACES,
    SCRATCHPAD_PATH,
    build_candidate_metadata,
)
from nanohorizon.craftax_core.runner import build_runner_summary


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

