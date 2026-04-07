from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable

import numpy as np


class RenderMode(str, Enum):
    NONE = "none"
    TEXT = "text"
    PIXELS = "pixels"
    BOTH = "both"

    @property
    def wants_text(self) -> bool:
        return self in {RenderMode.TEXT, RenderMode.BOTH}

    @property
    def wants_pixels(self) -> bool:
        return self in {RenderMode.PIXELS, RenderMode.BOTH}


@dataclass(frozen=True)
class RenderBundle:
    mode: RenderMode
    text: str | None = None
    pixels: np.ndarray | None = None
    state_view: Any = None


@dataclass(frozen=True)
class CallableRenderer:
    text_fn: Callable[[Any], str] | None = None
    pixels_fn: Callable[[Any], np.ndarray] | None = None
    structured_fn: Callable[[Any], Any] | None = None

    def render(self, state: Any, mode: RenderMode) -> RenderBundle:
        text = self.text_fn(state) if self.text_fn and mode.wants_text else None
        pixels = self.pixels_fn(state) if self.pixels_fn and mode.wants_pixels else None
        state_view = self.structured_fn(state) if self.structured_fn else None
        return RenderBundle(mode=mode, text=text, pixels=pixels, state_view=state_view)

