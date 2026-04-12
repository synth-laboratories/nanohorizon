import unittest

from nanohorizon.craftax_core.http_shim import render_prompt_turn
from nanohorizon.craftax_core.metadata import REWARD_WINDOW_SIZE
from nanohorizon.craftax_core.runner import summarize_reward_history


class CraftaxInterfaceTests(unittest.TestCase):
    def test_render_prompt_turn_uses_structured_fields(self) -> None:
        payload = render_prompt_turn(
            {
                "health": 10,
                "food": 2,
                "energy": 8,
                "nearby_entities": ["tree", "wolf"],
                "inventory": {"wood": 4},
                "noise": "extra",
            },
            history=[
                {"action": "move_north", "observation_summary": "safe", "reward_delta": 1.0},
                {"action": "mine", "observation_summary": "spent energy", "reward_delta": -0.5},
            ],
            metadata={
                "achievements_unlocked": [
                    "collect_wood",
                    "collect_sapling",
                    "collect_drink",
                    "eat_cow",
                    "eat_plant",
                ]
            },
        )

        self.assertEqual(payload["structured_observation"]["health"], 10)
        self.assertEqual(payload["structured_observation"]["extras"], {"noise": "extra"})
        self.assertEqual(payload["reward_history_labels"], ["+", "-"])
        self.assertEqual(payload["reward_history"][0]["sign"], "+")
        self.assertEqual(payload["achievements_unlocked"], [
            "collect_wood",
            "eat_cow",
            "collect_sapling",
            "collect_drink",
            "eat_plant",
        ])
        self.assertEqual(
            payload["next_targets"],
            ["place_table", "make_wood_pickaxe", "make_wood_sword", "place_plant"],
        )
        self.assertEqual(payload["unique_achievement_count"], 5)
        self.assertIn("zero reward", payload["achievement_score_rule"])

    def test_reward_history_is_bounded_to_five_steps(self) -> None:
        history = [
            {"action": f"a{i}", "observation_summary": f"s{i}", "reward_delta": float(i - 2)}
            for i in range(7)
        ]
        summarized = summarize_reward_history(history)

        self.assertEqual(len(summarized), REWARD_WINDOW_SIZE)
        self.assertEqual([entry["action"] for entry in summarized], ["a2", "a3", "a4", "a5", "a6"])

    def test_array_like_observations_are_labeled(self) -> None:
        class FakeArray:
            def __init__(self, values):
                self._values = values

            def tolist(self):
                return self._values

        payload = render_prompt_turn(FakeArray([11, 4, 9, "orc", {"stick": 1}, "tail"]))

        self.assertEqual(payload["structured_observation"]["health"], 11)
        self.assertEqual(
            payload["structured_observation"]["extras"]["remaining_features"],
            {"feature_5": "tail"},
        )
