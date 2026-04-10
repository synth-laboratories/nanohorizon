from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

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


def compact_state_summary(text: str, *, max_lines: int = 3, max_chars: int = 320) -> str:
    """Extract a compact resource/state summary from Craftax renderer text."""
    stripped_lines = [line.strip() for line in str(text).splitlines() if line.strip()]
    interesting_prefixes = (
        "inventory:",
        "achievements:",
        "player_position:",
        "health:",
        "resources:",
        "stats:",
        "position:",
    )
    selected = [
        line
        for line in stripped_lines
        if line.lower().startswith(interesting_prefixes)
    ]
    if not selected:
        selected = stripped_lines
    summary = " | ".join(selected[:max_lines])
    return summary[:max_chars].strip()


@dataclass(frozen=True)
class CraftaxWorkingMemoryEntry:
    turn_index: int
    plan: str
    actions: tuple[str, ...]
    observation: str
    state_summary: str
    reward: float = 0.0
    achievements: tuple[str, ...] = ()
    note: str = ""

    def render(self) -> str:
        pieces = [
            f"turn={self.turn_index}",
            f"plan={self.plan}",
        ]
        if self.actions:
            pieces.append(f"actions={', '.join(self.actions)}")
        if self.state_summary:
            pieces.append(f"state={self.state_summary}")
        if self.achievements:
            pieces.append(f"achievements={', '.join(self.achievements)}")
        if self.reward:
            pieces.append(f"reward={self.reward:.2f}")
        if self.note:
            pieces.append(f"note={self.note}")
        return " | ".join(pieces)


class WorkingMemoryBuffer:
    def __init__(self, capacity: int = 4) -> None:
        if capacity < 1:
            raise ValueError("capacity must be at least 1")
        self.capacity = int(capacity)
        self._entries: deque[CraftaxWorkingMemoryEntry] = deque()

    def push(
        self,
        *,
        turn_index: int,
        plan: str,
        actions: Sequence[str],
        observation: str,
        state_summary: str,
        reward: float = 0.0,
        achievements: Sequence[str] = (),
        note: str = "",
    ) -> CraftaxWorkingMemoryEntry:
        entry = CraftaxWorkingMemoryEntry(
            turn_index=int(turn_index),
            plan=str(plan).strip() or "unspecified",
            actions=tuple(str(action).strip() for action in actions if str(action).strip()),
            observation=str(observation).strip(),
            state_summary=compact_state_summary(state_summary),
            reward=float(reward),
            achievements=tuple(str(item).strip() for item in achievements if str(item).strip()),
            note=str(note).strip(),
        )
        self._entries.append(entry)
        while len(self._entries) > self.capacity:
            self._entries.popleft()
        return entry

    def latest(self) -> CraftaxWorkingMemoryEntry | None:
        return self._entries[-1] if self._entries else None

    def snapshot(self) -> list[dict[str, Any]]:
        return [asdict(entry) for entry in self._entries]

    def render(self) -> str:
        if not self._entries:
            return ""
        lines = ["Working memory (recent turns):"]
        lines.extend(f"- {entry.render()}" for entry in self._entries)
        return "\n".join(lines)
