from __future__ import annotations

import unittest

from nanohorizon.craftax_core.todo_tool import TodoBoard, TodoItem


class TodoToolTests(unittest.TestCase):
    def test_invalid_status_and_metadata_are_coerced(self) -> None:
        item = TodoItem.from_dict(
            {
                "title": "sanitize me",
                "status": "not-a-real-state",
                "metadata": "ignore-me",
            }
        )

        self.assertEqual(item.title, "sanitize me")
        self.assertEqual(item.status, "todo")
        self.assertEqual(item.metadata, {})

    def test_round_trip_preserves_item_state(self) -> None:
        board = TodoBoard(board_id="shared-history")
        first = board.add_item(
            "draft candidate",
            owner="agent",
            metadata={"phase": "plan"},
        )
        second = board.add_item("verify candidate", status="doing")
        board.mark_done(first.item_id or "")

        restored = TodoBoard.from_dict(board.to_dict())

        self.assertEqual(restored.board_id, "shared-history")
        self.assertEqual([item.title for item in restored.items], ["draft candidate", "verify candidate"])
        self.assertEqual(restored.items[0].status, "done")
        self.assertEqual(restored.items[1].status, "doing")
        self.assertEqual(restored.items[0].owner, "agent")
        self.assertEqual(restored.items[0].metadata, {"phase": "plan"})
        self.assertIsNotNone(restored.next_item())
        self.assertEqual(restored.open_items()[0].title, "verify candidate")
        self.assertIn("project_todo", board.to_dict())

    def test_from_dict_accepts_project_todo_alias(self) -> None:
        restored = TodoBoard.from_dict(
            {
                "board_id": "run-2",
                "project_todo": [
                    {
                        "title": "draft candidate",
                        "status": "doing",
                        "owner": "agent",
                    }
                ],
            }
        )

        self.assertEqual(restored.board_id, "run-2")
        self.assertEqual(len(restored.items), 1)
        self.assertEqual(restored.items[0].title, "draft candidate")
        self.assertEqual(restored.items[0].status, "doing")
        self.assertEqual(restored.items[0].owner, "agent")

    def test_from_titles_creates_ordered_board(self) -> None:
        board = TodoBoard.from_titles(["read docs", "patch code", "verify"], board_id="run-1")

        self.assertEqual(board.board_id, "run-1")
        self.assertEqual([item.title for item in board.items], ["read docs", "patch code", "verify"])
        self.assertEqual(board.next_item().title, "read docs")

    def test_unknown_item_id_raises(self) -> None:
        board = TodoBoard()

        with self.assertRaises(KeyError):
            board.mark_done("missing")


if __name__ == "__main__":
    unittest.main()
