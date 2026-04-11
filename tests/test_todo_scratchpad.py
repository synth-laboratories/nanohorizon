from __future__ import annotations

import unittest

from nanohorizon.craftax_core import TodoScratchpad, compact_todo_block


class TodoScratchpadMarkdownTest(unittest.TestCase):
    def test_render_marks_done_and_truncates(self) -> None:
        scratchpad = TodoScratchpad(title="Craftax todo")
        scratchpad.add("check harness surfaces")
        scratchpad.add("run verifier")
        scratchpad.add("capture notes")
        self.assertTrue(scratchpad.mark_done("run verifier"))

        self.assertEqual(
            scratchpad.render(max_items=2),
            "\n".join(
                [
                    "## Craftax todo",
                    "- [ ] check harness surfaces",
                    "- [x] run verifier",
                    "- [...] 1 more",
                ]
            ),
        )

    def test_compact_block_rejects_empty_items(self) -> None:
        scratchpad = TodoScratchpad(title="Craftax todo")

        with self.assertRaises(ValueError) as ctx:
            scratchpad.add("   ")

        self.assertIn("must not be empty", str(ctx.exception))

    def test_compact_todo_block_renders_from_strings(self) -> None:
        self.assertEqual(
            compact_todo_block("Craftax todo", ["first", "second"]),
            "\n".join(["## Craftax todo", "- [ ] first", "- [ ] second"]),
        )


if __name__ == "__main__":
    unittest.main()
