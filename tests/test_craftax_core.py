from __future__ import annotations

import unittest

from nanohorizon.craftax_core.http_shim import CraftaxHTTPShim
from nanohorizon.craftax_core.runner import CraftaxRunner, render_todo_board


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


if __name__ == "__main__":
    unittest.main()

