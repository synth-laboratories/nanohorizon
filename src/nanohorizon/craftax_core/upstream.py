from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from .metadata import DEFAULT_ACHIEVEMENT_NAMES, FULL_ACHIEVEMENTS
from .modalities import CallableRenderer, RenderMode
from .runner import DeterministicCraftaxRunner

EnvKind = Literal["full", "classic"]


def _raise_missing(extra: str) -> None:
    raise ImportError("craftax imports were not available. Install the classic dependency group. " + extra)


def _normalize_action_names(raw: dict[str, int]) -> dict[str, int]:
    normalized: dict[str, int] = {}
    for name, index in raw.items():
        key = str(name).strip().lower()
        if key in {"left", "right", "up", "down"}:
            key = f"move_{key}"
        normalized[key] = int(index)
    return normalized


@dataclass(frozen=True)
class CraftaxRendererConfig:
    kind: EnvKind = "full"
    verbose: bool = True
    formatting: Literal["md", "xml"] = "md"
    map_format: Literal["full", "compact"] = "full"


def _fallback_state_text(state: Any) -> str:
    lines = ["Craftax state summary"]
    inventory = getattr(state, "inventory", None)
    if inventory is not None:
        names = [name for name in dir(inventory) if not name.startswith("_")]
        inventory_items = []
        for name in names:
            value = getattr(inventory, name)
            if isinstance(value, (int, float, bool, np.integer, np.floating)):
                inventory_items.append(f"{name}={value}")
        if inventory_items:
            lines.append("inventory: " + ", ".join(sorted(inventory_items)))
    achievements = getattr(state, "achievements", None)
    if achievements is not None:
        unlocked = []
        for index, raw_value in enumerate(np.asarray(achievements).tolist()):
            if float(raw_value) > 0:
                unlocked.append(FULL_ACHIEVEMENTS.get(index, f"achievement_{index}"))
        lines.append("achievements: " + (", ".join(unlocked) if unlocked else "none"))
    player = getattr(state, "player_position", None)
    if player is not None:
        lines.append(f"player_position: {player}")
    return "\n".join(lines)


def _fallback_state_view(state: Any) -> dict[str, Any]:
    inventory = getattr(state, "inventory", None)
    inventory_items: dict[str, Any] = {}
    if inventory is not None:
        for name in dir(inventory):
            if name.startswith("_"):
                continue
            value = getattr(inventory, name)
            if isinstance(value, (int, float, bool, np.integer, np.floating)):
                inventory_items[name] = value
    achievements = achievement_names_from_state(state)
    player_position = getattr(state, "player_position", None)
    return {
        "summary": _fallback_state_text(state),
        "inventory": inventory_items,
        "achievements": achievements,
        "player_position": player_position,
    }


class CraftaxRendererFactory:
    def __init__(self, config: CraftaxRendererConfig) -> None:
        self.config = config

    def build(self) -> CallableRenderer:
        kind = self.config.kind
        try:
            if kind == "classic":
                from craftax.craftax.constants import BLOCK_PIXEL_SIZE_HUMAN
                from craftax.craftax_classic.renderer import render_craftax_pixels
            else:
                from craftax.craftax.constants import BLOCK_PIXEL_SIZE_HUMAN
                from craftax.craftax.renderer import render_craftax_pixels
        except Exception as exc:  # pragma: no cover - optional dependency
            _raise_missing(str(exc))

        text_fn = _fallback_state_text
        structured_fn = _fallback_state_view
        try:
            if kind == "classic":
                from craftaxlm.classic.state import render_craftax_classic_text_custom

                def _text_fn(state: Any) -> str:
                    try:
                        view = render_craftax_classic_text_custom(state)
                        return view.render_to_text_simple(
                            verbose=self.config.verbose,
                            formatting=self.config.formatting,
                            map_format=self.config.map_format,
                        )
                    except Exception:
                        return _fallback_state_text(state)

                text_fn = _text_fn
            else:
                from craftaxlm.full.state import render_craftax_text_custom

                def _text_fn(state: Any) -> str:
                    try:
                        view = render_craftax_text_custom(state)
                        return view.render_to_text_simple(
                            verbose=self.config.verbose,
                            formatting=self.config.formatting,
                            map_format=self.config.map_format,
                        )
                    except Exception:
                        return _fallback_state_text(state)

                text_fn = _text_fn
        except Exception:
            pass

        def render_pixels_fn(state: Any) -> np.ndarray:
            frame = render_craftax_pixels(state, BLOCK_PIXEL_SIZE_HUMAN)
            return np.asarray(frame, dtype=np.uint8)

        return CallableRenderer(
            text_fn=text_fn,
            pixels_fn=render_pixels_fn,
            structured_fn=structured_fn,
        )


def make_symbolic_env(kind: EnvKind = "full") -> Any:
    try:
        from craftax.craftax_env import make_craftax_env_from_name
    except Exception as exc:  # pragma: no cover - optional dependency
        _raise_missing(str(exc))
    env_name = "Craftax-Symbolic-v1" if kind == "full" else "Craftax-Classic-Symbolic-v1"
    return make_craftax_env_from_name(env_name, auto_reset=False)


def make_runner(
    *,
    kind: EnvKind = "full",
    seed: int = 0,
    params: Any = None,
    render_mode: RenderMode = RenderMode.TEXT,
    verbose: bool = True,
    formatting: Literal["md", "xml"] = "md",
    map_format: Literal["full", "compact"] = "full",
) -> DeterministicCraftaxRunner:
    renderer = CraftaxRendererFactory(
        CraftaxRendererConfig(
            kind=kind,
            verbose=verbose,
            formatting=formatting,
            map_format=map_format,
        )
    ).build()
    return DeterministicCraftaxRunner(
        env=lambda: make_symbolic_env(kind),
        renderer=renderer,
        seed=seed,
        params=params,
        render_mode=render_mode,
    )


def action_name_to_index() -> dict[str, int]:
    return _normalize_action_names(
        {
            **{
                "noop": 0,
                "left": 1,
                "right": 2,
                "up": 3,
                "down": 4,
            },
            **{name: index for name, index in {
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
            }.items()},
        }
    )


def achievement_names_from_state(state: Any) -> list[str]:
    raw = getattr(state, "achievements", None)
    if raw is None:
        return []
    unlocked: list[str] = []
    for index, value in enumerate(np.asarray(raw).tolist()):
        if float(value) > 0:
            unlocked.append(FULL_ACHIEVEMENTS.get(index, f"achievement_{index}"))
    return unlocked
