from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np


def glyph_cmap_off() -> int:
    try:
        import nle.nethack as nethack

        return int(nethack.GLYPH_CMAP_OFF)
    except Exception:
        return 2359


@dataclass
class ScoutTracker:
    cmap_off: int = field(default_factory=glyph_cmap_off)
    explored_by_level: dict[tuple[int, int], int] = field(default_factory=dict)
    total: float = 0.0

    def reset(self) -> None:
        self.explored_by_level = {}
        self.total = 0.0

    def reward(self, observation: Mapping[str, Any]) -> float:
        glyphs = observation.get("glyphs")
        blstats = observation.get("blstats")
        if glyphs is None or blstats is None:
            return 0.0
        glyph_array = np.asarray(glyphs)
        stats = np.asarray(blstats).reshape(-1)
        dnum = int(stats[23]) if len(stats) > 23 else 0
        dlevel = int(stats[24]) if len(stats) > 24 else int(stats[12]) if len(stats) > 12 else 0
        key = (dnum, dlevel)
        explored = int(np.sum(glyph_array != self.cmap_off))
        previous = self.explored_by_level.get(key, 0)
        reward = float(max(0, explored - previous))
        self.explored_by_level[key] = max(previous, explored)
        self.total += reward
        return reward
