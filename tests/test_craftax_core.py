import unittest

from nanohorizon.craftax_core.metadata import WorkingMemoryBuffer


class WorkingMemoryBufferTests(unittest.TestCase):
    def test_caps_entries(self) -> None:
        buffer = WorkingMemoryBuffer(capacity=2)
        buffer.push(subgoal="first", resource_state={"wood": 1})
        buffer.push(subgoal="second", resource_state={"wood": 2})
        buffer.push(subgoal="third", resource_state={"wood": 3})

        snapshot = buffer.snapshot()
        self.assertEqual([item["subgoal"] for item in snapshot], ["second", "third"])

    def test_renders_compact_prompt(self) -> None:
        buffer = WorkingMemoryBuffer(capacity=2)
        buffer.push(
            subgoal="gather food",
            resource_state={"food": 2, "wood": 1},
            action_plan="keep it small",
        )

        rendered = buffer.render()
        self.assertIn("Working memory", rendered)
        self.assertIn("subgoal=gather food", rendered)
        self.assertIn("resources={food=2, wood=1}", rendered)

