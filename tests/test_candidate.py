from __future__ import annotations

import contextlib
import io
import json
import tempfile
from pathlib import Path
import unittest

from nanohorizon.baselines.prompt_opt import candidate_config
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
        self.assertEqual(manifest["track_name"], candidate_config()["track"])

    def test_prompt_contains_todo_scratchpad(self) -> None:
        prompt = build_candidate_prompt()
        self.assertIn("todo scratchpad", prompt)
        self.assertIn("Stop only after the verifier pass is documented.", prompt)

    def test_runner_writes_summary_and_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            output_dir = Path(tmpdir) / "artifacts"
            manifest_path = output_dir / "candidate_manifest.json"
            prompt_path = output_dir / "candidate_prompt.txt"
            config_path.write_text(json.dumps(candidate_config()), encoding="utf-8")
            with contextlib.redirect_stdout(io.StringIO()):
                exit_code = main(
                    [
                        "--config",
                        str(config_path),
                        "--output-dir",
                        str(output_dir),
                        "--write",
                        str(manifest_path),
                        "--prompt-out",
                        str(prompt_path),
                    ]
                )
            self.assertEqual(exit_code, 0)
            summary = json.loads((output_dir / "smoke_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["candidate_manifest"]["label"], CANDIDATE_LABEL)
            self.assertEqual(summary["loaded_config"]["candidate"]["label"], CANDIDATE_LABEL)
            self.assertEqual(manifest_path.read_text(encoding="utf-8").strip(), json.dumps(build_candidate_manifest(), indent=2, sort_keys=True))
            self.assertEqual(prompt_path.read_text(encoding="utf-8"), build_candidate_prompt() + "\n")
