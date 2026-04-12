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


def _coerce_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize_text_fragments(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, Mapping):
        fragments: list[str] = []
        for key, item in value.items():
            fragments.append(f"{key}={_shorten_text(item, 80)}")
        return fragments
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return [str(item) for item in value]
    return [str(value)]


def _contains_text(text: str, needles: Sequence[str]) -> bool:
    lowered = text.lower()
    return any(needle in lowered for needle in needles)


def _inventory_positive_keys(inventory: Any) -> list[str]:
    keys: list[str] = []
    if isinstance(inventory, Mapping):
        for key, value in inventory.items():
            if isinstance(value, (int, float)) and float(value) > 0.0:
                keys.append(str(key))
    return keys


def _recent_negative_streak(labels: Sequence[str]) -> int:
    streak = 0
    for label in reversed(labels):
        if label == "-":
            streak += 1
            continue
        break
    return streak


@dataclass(slots=True)
class DecisionBrief:
    """Compact policy hint derived from the current Craftax context."""

    mode: str
    primary_focus: str
    fallback_action: str
    loop_risk: str
    priority_targets: list[str] = field(default_factory=list)
    signals: list[str] = field(default_factory=list)

    def to_prompt_payload(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "primary_focus": self.primary_focus,
            "fallback_action": self.fallback_action,
            "loop_risk": self.loop_risk,
            "priority_targets": list(self.priority_targets),
            "signals": list(self.signals),
        }

    def brief_summary(self) -> str:
        targets = ", ".join(self.priority_targets) if self.priority_targets else "none"
        signals = ", ".join(self.signals) if self.signals else "none"
        return (
            f"mode={self.mode}; focus={self.primary_focus}; fallback={self.fallback_action}; "
            f"loop_risk={self.loop_risk}; targets={targets}; signals={signals}"
        )


def derive_decision_brief(
    observation: StructuredObservation,
    reward_history: RewardHistoryWindow,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> DecisionBrief:
    metadata = dict(metadata or {})
    fragments = []
    fragments.extend(_normalize_text_fragments(observation.brief_summary()))
    fragments.extend(_normalize_text_fragments(observation.extras.get("raw_observation")))
    fragments.extend(_normalize_text_fragments(metadata.get("observation_text")))
    fragments.extend(_normalize_text_fragments(metadata.get("observation_summary")))
    combined = " ".join(fragment for fragment in fragments if fragment).lower()

    if isinstance(observation.inventory, Mapping):
        inventory_keys = _inventory_positive_keys(observation.inventory)
    else:
        inventory_keys = []

    recent_actions = metadata.get("recent_actions")
    if isinstance(recent_actions, Sequence) and not isinstance(recent_actions, (str, bytes, bytearray)):
        recent_action_list = [str(item).strip().lower() for item in recent_actions if str(item).strip()]
    else:
        recent_action_list = []

    signals: list[str] = []
    if combined:
        for token, tag in (
            ("night", "night"),
            ("daylight", "daylight"),
            ("low energy", "low_energy"),
            ("low health", "low_health"),
            ("tree", "tree"),
            ("sapling", "sapling"),
            ("stone", "stone"),
            ("wood", "wood"),
            ("table", "table"),
            ("furnace", "furnace"),
            ("enemy", "enemy"),
            ("wolf", "wolf"),
            ("zombie", "zombie"),
            ("skeleton", "skeleton"),
        ):
            if token in combined:
                signals.append(tag)
    for key in inventory_keys[:4]:
        signals.append(f"inv:{key}")
    signals = list(dict.fromkeys(signals))

    health = _coerce_float(observation.health)
    food = _coerce_float(observation.food)
    energy = _coerce_float(observation.energy)
    low_resource = any(value is not None and value <= 3.0 for value in (health, energy)) or (
        food is not None and food <= 1.0
    )
    low_resource = low_resource or _contains_text(
        combined,
        (
            "health low",
            "low health",
            "energy low",
            "low energy",
            "food low",
            "low food",
        ),
    )
    threat_signal = any(tag in {"enemy", "wolf", "zombie", "skeleton"} for tag in signals)
    resource_signal = any(tag in {"tree", "sapling", "stone", "wood"} for tag in signals)
    crafting_signal = any(tag in {"table", "furnace"} for tag in signals)
    inventory_signal = any(key in {"wood", "stone", "coal", "iron", "diamond"} for key in inventory_keys)
    negative_streak = _recent_negative_streak(reward_history.labels())
    repeated_actions = bool(
        len(recent_action_list) >= 3 and len(set(recent_action_list[-3:])) == 1
    )
    loop_risk = "high" if repeated_actions or negative_streak >= 2 else "moderate" if negative_streak == 1 else "low"

    priority_targets: list[str] = []
    mode = "explore"
    if low_resource or (combined and "sleep" in combined and (health is not None and health <= 4.0)):
        mode = "stabilize"
        priority_targets.extend(["recover health or energy", "avoid risky crafting"])
    elif threat_signal and not resource_signal:
        mode = "stabilize"
        priority_targets.extend(["survive the threat", "move to safer ground"])
    elif resource_signal:
        mode = "gather"
        if "tree" in signals:
            priority_targets.append("tree")
        if "sapling" in signals:
            priority_targets.append("sapling")
        if "stone" in signals:
            priority_targets.append("stone")
        if "wood" in signals:
            priority_targets.append("wood")
    elif crafting_signal or inventory_signal:
        mode = "craft"
        if "table" in signals or "wood" in inventory_keys:
            priority_targets.append("place_table")
        if "furnace" in signals or "stone" in inventory_keys:
            priority_targets.append("make_pickaxe")
    else:
        priority_targets.extend(["scan for a gatherable target", "break any movement loop"])

    if mode == "stabilize":
        primary_focus = "protect survival before spending actions on crafting or long movement"
        fallback_action = "switch direction or rest only if the local state clearly needs recovery"
    elif mode == "gather":
        best_target = priority_targets[0] if priority_targets else "the nearest useful resource"
        primary_focus = f"move toward {best_target} and use do when adjacent"
        fallback_action = "change axis or choose a different movement direction if blocked"
    elif mode == "craft":
        primary_focus = "convert already gathered materials into the simplest progress-making craft"
        fallback_action = "return to gathering if the materials are not sufficient yet"
    else:
        primary_focus = "move to fresh information and avoid repeating the same local movement"
        fallback_action = "switch direction to break the loop and re-scan the area"

    if loop_risk == "high":
        fallback_action = "change movement pattern immediately and avoid repeating the last action batch"

    return DecisionBrief(
        mode=mode,
        primary_focus=primary_focus,
        fallback_action=fallback_action,
        loop_risk=loop_risk,
        priority_targets=priority_targets[:3],
        signals=signals[:6],
    )


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
        decision_brief = derive_decision_brief(
            self.observation,
            self.reward_history,
            metadata=self.metadata,
        )
        return {
            "structured_observation": self.observation.to_prompt_payload(),
            "structured_observation_summary": self.observation.brief_summary(),
            "reward_history": self.reward_history.to_prompt_payload(),
            "reward_history_labels": self.reward_history.labels(),
            "decision_brief": decision_brief.to_prompt_payload(),
            "decision_brief_summary": decision_brief.brief_summary(),
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
