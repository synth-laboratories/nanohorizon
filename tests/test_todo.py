from __future__ import annotations

import unittest

from nanohorizon.craftax_core import TodoItem, TodoScratchpad, compact_todo_block


class TodoModuleCompatibilityTest(unittest.TestCase):
    def test_public_module_exports_scratchpad_primitives(self) -> None:
        item = TodoItem("verify candidate")
        scratchpad = TodoScratchpad(title="Craftax todo")
        scratchpad.add(item.text)

        self.assertEqual(item.render(), "- [ ] verify candidate")
        self.assertEqual(
            compact_todo_block("Craftax todo", [item.text]),
            "\n".join(["## Craftax todo", "- [ ] verify candidate"]),
        )


if __name__ == "__main__":
    unittest.main()
