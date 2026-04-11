from __future__ import annotations

import tempfile
from pathlib import Path
import unittest

from nanohorizon.craftax_core.http_shim import CraftaxHTTPShim
from nanohorizon.craftax_core.runner import CraftaxRunner, main, render_todo_board


class CraftaxCoreTests(unittest.TestCase):
    def test_todo_board_mentions_key_items(self) -> None:
        board = render_todo_board()
        self.assertIn("Todo Board", board)
        self.assertIn("scratchpad", board)
        self.assertIn("verify", board)

    def test_http_shim_exposes_rollout_alias(self) -> None:
        shim = CraftaxHTTPShim()
        self.assertEqual(shim.rollout({"hello": "world"}), shim.rollouts({"hello": "world"}))
        self.assertTrue(shim.health()["ok"])

    def test_runner_summary_includes_surfaces(self) -> None:
        runner = CraftaxRunner()
        summary = runner.summary()
        self.assertEqual(summary["candidate_label"], "Daytona E2E Run 3")
        self.assertIn("stable_surfaces", summary)
        self.assertGreaterEqual(len(summary["stable_surfaces"]), 3)

    def test_runner_main_writes_smoke_payload(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "smoke.json"
            exit_code = main(["--smoke", "--json", "--output", str(output_path)])
            self.assertEqual(exit_code, 0)
            payload = output_path.read_text(encoding="utf-8")
            self.assertIn('"smoke": true', payload)
            self.assertIn('"candidate_label": "Daytona E2E Run 3"', payload)


if __name__ == "__main__":
    unittest.main()
