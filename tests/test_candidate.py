from __future__ import annotations

import json
import tempfile
from pathlib import Path
import unittest

from nanohorizon.baselines.prompt_opt import (
    FULL_AUTO_E2E_SYSTEM_PROMPT,
    TRACK_NAME,
    candidate_config,
    load_config,
    todo_scratchpad_directive,
)
from nanohorizon.craftax_core.metadata import (
    CANDIDATE_LABEL,
    OUTPUT_ROOT,
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
        self.assertEqual(manifest["output_root"], OUTPUT_ROOT)
        self.assertEqual(
            manifest["verification_modes"],
            ["config_roundtrip_smoke", "prompt_render_smoke"],
        )
        self.assertEqual(manifest["track_name"], TRACK_NAME)

    def test_prompt_contains_candidate_prompt_text(self) -> None:
        prompt = build_candidate_prompt()
        self.assertIn(FULL_AUTO_E2E_SYSTEM_PROMPT, prompt)
        self.assertIn("Todo contract:", prompt)
        self.assertIn(todo_scratchpad_directive(), prompt)

    def test_candidate_config_round_trip(self) -> None:
        self.assertEqual(
            load_config("configs/craftax_prompt_opt_qwen35_4b_full_auto_e2e.yaml"),
            candidate_config(),
        )

    def test_runner_writes_summary_and_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "artifacts"
            manifest_path = output_dir / "candidate_manifest.json"
            prompt_path = output_dir / "candidate_prompt.txt"
            exit_code = main(
                [
                    "--config",
                    "configs/craftax_prompt_opt_qwen35_4b_full_auto_e2e.yaml",
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
            self.assertEqual(summary["loaded_config"], candidate_config())
            self.assertEqual(
                manifest_path.read_text(encoding="utf-8"),
                json.dumps(build_candidate_manifest(), indent=2, sort_keys=True) + "\n",
            )
            self.assertEqual(
                prompt_path.read_text(encoding="utf-8"),
                build_candidate_prompt() + "\n",
            )
