from __future__ import annotations

import json
import tempfile
from pathlib import Path
import unittest

from nanohorizon.craftax_core.metadata import (
    CANDIDATE_LABEL,
    PRIMARY_STRATEGY,
    build_candidate_manifest,
    build_candidate_prompt,
)
from nanohorizon.craftax_core.runner import main


class CandidateSmokeTest(unittest.TestCase):
    def test_manifest_contains_candidate_metadata(self) -> None:
        manifest = build_candidate_manifest()
        self.assertEqual(manifest["label"], CANDIDATE_LABEL)
        self.assertEqual(manifest["strategy"], PRIMARY_STRATEGY)
        self.assertGreaterEqual(len(manifest["todo_items"]), 3)

    def test_prompt_contains_todo_scratchpad(self) -> None:
        prompt = build_candidate_prompt()
        self.assertIn("todo scratchpad", prompt)
        self.assertIn("Stop only after the verifier pass is documented.", prompt)

    def test_runner_writes_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "manifest.json"
            exit_code = main(["--write", str(path)])
            self.assertEqual(exit_code, 0)
            written = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(written["label"], CANDIDATE_LABEL)
