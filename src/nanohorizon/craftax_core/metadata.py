"""Stable prompt metadata for Craftax interface shaping.

The goal is to keep the harness surfaces narrow while making the prompt input
more semantically legible for downstream policy code:

* Craftax observations are converted into labeled fields.
* Reward context is tracked as a rolling five-step window.
* Each history element keeps the action, a brief observation summary, and the
  signed reward delta so the model sees recent trajectory, not only the present
  frame.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence

OBSERVATION_FIELD_ORDER = (
    "health",
    "food",
    "energy",
    "nearby_entities",
    "inventory",
)

REWARD_WINDOW_SIZE = 5


def _sign_label(reward_delta: float) -> str:
    if reward_delta > 0:
        return "+"
    if reward_delta < 0:
        return "-"
    return "neutral"


def _shorten_text(value: Any, limit: int = 140) -> str:
    text = repr(value) if not isinstance(value, str) else value
    if len(text) <= limit:
        return text
    return f"{text[: limit - 3]}..."


@dataclass(slots=True)
class StructuredObservation:
    """Semantic view of a Craftax observation."""

    health: Any = None
    food: Any = None
    energy: Any = None
    nearby_entities: Any = None
    inventory: Any = None
    extras: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_observation(cls, observation: Any) -> "StructuredObservation":
        if isinstance(observation, Mapping):
            known = {field: observation.get(field) for field in OBSERVATION_FIELD_ORDER}
            extras = {
                key: value
                for key, value in observation.items()
                if key not in OBSERVATION_FIELD_ORDER
            }
            return cls(**known, extras=extras)

        if hasattr(observation, "tolist") and not isinstance(observation, (str, bytes, bytearray)):
            try:
                observation = observation.tolist()
            except Exception:
                pass

        if isinstance(observation, Sequence) and not isinstance(observation, (str, bytes, bytearray)):
            values = list(observation)
            known = {
                field: values[idx] if idx < len(values) else None
                for idx, field in enumerate(OBSERVATION_FIELD_ORDER)
            }
            extras = {
                "remaining_features": {
                    f"feature_{idx}": value
                    for idx, value in enumerate(values[len(OBSERVATION_FIELD_ORDER) :], start=len(OBSERVATION_FIELD_ORDER))
                }
            }
            return cls(**known, extras=extras)

        return cls(extras={"raw_observation": observation})

    def to_prompt_payload(self) -> dict[str, Any]:
        payload = {field: getattr(self, field) for field in OBSERVATION_FIELD_ORDER}
        payload["extras"] = self.extras
        return payload

    def brief_summary(self) -> str:
        parts = []
        for field in OBSERVATION_FIELD_ORDER:
            value = getattr(self, field)
            if value is not None:
                parts.append(f"{field}={_shorten_text(value)}")
        if self.extras:
            parts.append(f"extras={_shorten_text(self.extras)}")
        return ", ".join(parts) if parts else "no structured observation available"


@dataclass(slots=True)
class RewardHistoryEntry:
    """One reward-tagged trajectory step."""

    action: Any
    observation_summary: str
    reward_delta: float

    @property
    def sign_label(self) -> str:
        return _sign_label(self.reward_delta)

    def to_prompt_payload(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "observation_summary": self.observation_summary,
            "reward_delta": self.reward_delta,
            "sign": self.sign_label,
        }


@dataclass(slots=True)
class RewardHistoryWindow:
    """Rolling window of the most recent reward-tagged steps."""

    entries: list[RewardHistoryEntry] = field(default_factory=list)

    def append(self, entry: RewardHistoryEntry) -> None:
        self.entries.append(entry)
        if len(self.entries) > REWARD_WINDOW_SIZE:
            self.entries = self.entries[-REWARD_WINDOW_SIZE:]

    def extend(self, entries: Sequence[RewardHistoryEntry]) -> None:
        for entry in entries:
            self.append(entry)

    def to_prompt_payload(self) -> list[dict[str, Any]]:
        return [entry.to_prompt_payload() for entry in self.entries]

    def labels(self) -> list[str]:
        return [entry.sign_label for entry in self.entries]


@dataclass(slots=True)
class PromptContext:
    """Prompt-ready Craftax interface payload."""

    observation: StructuredObservation
    reward_history: RewardHistoryWindow = field(default_factory=RewardHistoryWindow)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_prompt_payload(self) -> dict[str, Any]:
        return {
            "structured_observation": self.observation.to_prompt_payload(),
            "structured_observation_summary": self.observation.brief_summary(),
            "reward_history": self.reward_history.to_prompt_payload(),
            "reward_history_labels": self.reward_history.labels(),
            "metadata": dict(self.metadata),
        }

    def to_jsonable(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Legacy constants — preserved for backward compatibility with rollout.py,
# eval_model.py, and other modules that import these names directly from
# nanohorizon.craftax_core.metadata.
# ---------------------------------------------------------------------------

PRIMARY_TOOL_NAME = "craftax_interact"

FULL_ACTIONS = {
    "noop": 0,
    "move_left": 1,
    "move_right": 2,
    "move_up": 3,
    "move_down": 4,
    "do": 5,
    "sleep": 6,
    "place_stone": 7,
    "place_table": 8,
    "place_furnace": 9,
    "place_plant": 10,
    "make_wood_pickaxe": 11,
    "make_stone_pickaxe": 12,
    "make_iron_pickaxe": 13,
    "make_wood_sword": 14,
    "make_stone_sword": 15,
    "make_iron_sword": 16,
    "rest": 17,
    "descend": 18,
    "ascend": 19,
    "make_diamond_pickaxe": 20,
    "make_diamond_sword": 21,
    "make_iron_armour": 22,
    "make_diamond_armour": 23,
    "shoot_arrow": 24,
    "make_arrow": 25,
    "cast_fireball": 26,
    "cast_iceball": 27,
    "place_torch": 28,
    "drink_potion_red": 29,
    "drink_potion_green": 30,
    "drink_potion_blue": 31,
    "drink_potion_pink": 32,
    "drink_potion_cyan": 33,
    "drink_potion_yellow": 34,
    "read_book": 35,
    "enchant_sword": 36,
    "enchant_armour": 37,
    "make_torch": 38,
    "level_up_dexterity": 39,
    "level_up_strength": 40,
    "level_up_intelligence": 41,
    "enchant_bow": 42,
}

FULL_ACHIEVEMENTS = {
    0: "collect_wood",
    1: "place_table",
    2: "eat_cow",
    3: "collect_sapling",
    4: "collect_drink",
    5: "make_wood_pickaxe",
    6: "make_wood_sword",
    7: "place_plant",
    8: "defeat_zombie",
    9: "collect_stone",
    10: "place_stone",
    11: "eat_plant",
    12: "defeat_skeleton",
    13: "make_stone_pickaxe",
    14: "make_stone_sword",
    15: "wake_up",
    16: "place_furnace",
    17: "collect_coal",
    18: "collect_iron",
    19: "collect_diamond",
    20: "make_iron_pickaxe",
    21: "make_iron_sword",
    22: "make_arrow",
    23: "make_torch",
    24: "place_torch",
    25: "make_diamond_sword",
    26: "make_iron_armour",
    27: "make_diamond_armour",
    28: "enter_gnomish_mines",
    29: "enter_dungeon",
    30: "enter_sewers",
    31: "enter_vault",
    32: "enter_troll_mines",
    33: "enter_fire_realm",
    34: "enter_ice_realm",
    35: "enter_graveyard",
    36: "defeat_gnome_warrior",
    37: "defeat_gnome_archer",
    38: "defeat_orc_soldier",
    39: "defeat_orc_mage",
    40: "defeat_lizard",
    41: "defeat_kobold",
    42: "defeat_troll",
    43: "defeat_deep_thing",
    44: "defeat_pigman",
    45: "defeat_fire_elemental",
    46: "defeat_frost_troll",
    47: "defeat_ice_elemental",
    48: "damage_necromancer",
    49: "defeat_necromancer",
    50: "eat_bat",
    51: "eat_snail",
    52: "find_bow",
    53: "fire_bow",
    54: "collect_sapphire",
    55: "learn_fireball",
    56: "cast_fireball",
    57: "learn_iceball",
    58: "cast_iceball",
    59: "collect_ruby",
    60: "make_diamond_pickaxe",
    61: "open_chest",
    62: "drink_potion",
    63: "enchant_sword",
    64: "enchant_armour",
    65: "defeat_knight",
    66: "defeat_archer",
}

DEFAULT_ACTION_NAMES = list(FULL_ACTIONS.keys())
DEFAULT_ACHIEVEMENT_NAMES = list(FULL_ACHIEVEMENTS.values())
