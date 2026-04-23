from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

PRIMARY_TOOL_NAME = "nle_interact"


@dataclass(frozen=True)
class NLEAction:
    name: str
    value: int


_FALLBACK_ACTIONS: tuple[NLEAction, ...] = (
    NLEAction("more", 19),
    NLEAction("north", 107),
    NLEAction("east", 108),
    NLEAction("south", 106),
    NLEAction("west", 104),
    NLEAction("north_east", 117),
    NLEAction("south_east", 110),
    NLEAction("south_west", 98),
    NLEAction("north_west", 121),
    NLEAction("search", 115),
    NLEAction("kick", 4),
    NLEAction("eat", 101),
    NLEAction("pickup", 44),
    NLEAction("drop", 100),
    NLEAction("inventory", 105),
    NLEAction("wait", 46),
    NLEAction("escape", 27),
)


def _normalize_action_name(raw: str, *, used: set[str]) -> str:
    name = re.sub(r"[^a-zA-Z0-9]+", "_", raw).strip("_").lower()
    if not name:
        name = "action"
    base = name
    suffix = 2
    while name in used:
        name = f"{base}_{suffix}"
        suffix += 1
    used.add(name)
    return name


def _action_label(action: Any) -> str:
    name = getattr(action, "name", None)
    if name:
        return str(name)
    return str(action)


def _action_value(action: Any) -> int:
    try:
        return int(action)
    except Exception:
        value = getattr(action, "value", action)
        return int(value)


def action_catalog() -> list[NLEAction]:
    try:
        import nle.nethack as nethack

        raw_actions = list(nethack.ACTIONS)
    except Exception:
        return list(_FALLBACK_ACTIONS)

    used: set[str] = set()
    catalog: list[NLEAction] = []
    for action in raw_actions:
        label = _action_label(action)
        name = _normalize_action_name(label, used=used)
        catalog.append(NLEAction(name=name, value=_action_value(action)))
    return catalog


DEFAULT_ACTIONS = action_catalog()
DEFAULT_ACTION_NAMES = [action.name for action in DEFAULT_ACTIONS]
ACTION_NAME_TO_VALUE = {action.name: action.value for action in DEFAULT_ACTIONS}


def safe_fallback_action_name() -> str:
    for candidate in ("more", "miscaction_more", "wait", "escape"):
        if candidate in ACTION_NAME_TO_VALUE:
            return candidate
    return DEFAULT_ACTION_NAMES[0]
