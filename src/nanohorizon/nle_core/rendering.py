from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont
except Exception:  # pragma: no cover - optional dependency
    Image = None
    ImageDraw = None
    ImageFont = None


_COLOR_TABLE = {
    0: (0, 0, 0),
    1: (170, 0, 0),
    2: (0, 170, 0),
    3: (170, 85, 0),
    4: (0, 0, 170),
    5: (170, 0, 170),
    6: (0, 170, 170),
    7: (210, 210, 210),
    8: (90, 90, 90),
    9: (255, 85, 85),
    10: (85, 255, 85),
    11: (255, 255, 85),
    12: (85, 85, 255),
    13: (255, 85, 255),
    14: (85, 255, 255),
    15: (255, 255, 255),
}


def decode_bytes(value: Any) -> str:
    array = np.asarray(value)
    if array.dtype.kind in {"U", "S"}:
        return "".join(str(item) for item in array.tolist()).rstrip("\x00 ")
    chars = []
    for raw in array.reshape(-1).tolist():
        try:
            code = int(raw)
        except Exception:
            continue
        if code == 0:
            continue
        chars.append(chr(code))
    return "".join(chars).rstrip()


def terminal_lines(tty_chars: Any) -> list[str]:
    chars = np.asarray(tty_chars)
    if chars.ndim != 2:
        return []
    return [decode_bytes(row) for row in chars]


def render_text(observation: Mapping[str, Any], *, info: Mapping[str, Any] | None = None) -> str:
    parts: list[str] = []
    message = observation.get("message")
    if message is not None:
        parts.append(f"message: {decode_bytes(message)}")

    blstats = observation.get("blstats")
    if blstats is not None:
        stats = np.asarray(blstats).reshape(-1)
        labels = [
            ("x", 0),
            ("y", 1),
            ("score", 9),
            ("hp", 10),
            ("max_hp", 11),
            ("depth", 12),
            ("gold", 13),
            ("energy", 14),
            ("max_energy", 15),
            ("ac", 16),
            ("xl", 18),
            ("time", 20),
            ("dnum", 23),
            ("dlevel", 24),
        ]
        rendered = []
        for label, index in labels:
            if index < len(stats):
                rendered.append(f"{label}={int(stats[index])}")
        if rendered:
            parts.append("blstats: " + " ".join(rendered))

    if info:
        end_status = info.get("end_status")
        if end_status is not None:
            parts.append(f"end_status: {end_status}")

    lines = terminal_lines(observation.get("tty_chars")) if observation.get("tty_chars") is not None else []
    if lines:
        parts.append("terminal:\n" + "\n".join(lines).rstrip())
    return "\n".join(part for part in parts if part).strip()


def render_pixels(observation: Mapping[str, Any]) -> np.ndarray | None:
    tty_chars = observation.get("tty_chars")
    if tty_chars is None:
        return None
    chars = np.asarray(tty_chars)
    if chars.ndim != 2:
        return None

    colors = observation.get("tty_colors")
    color_array = np.asarray(colors) if colors is not None else np.full(chars.shape, 15, dtype=np.uint8)
    if color_array.shape != chars.shape:
        color_array = np.full(chars.shape, 15, dtype=np.uint8)

    if Image is None or ImageDraw is None or ImageFont is None:
        h, w = chars.shape
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        for color, rgb in _COLOR_TABLE.items():
            frame[color_array == color] = rgb
        return frame

    font = ImageFont.load_default()
    cell_w, cell_h = 8, 12
    height, width = chars.shape
    image = Image.new("RGB", (width * cell_w, height * cell_h), (0, 0, 0))
    draw = ImageDraw.Draw(image)
    for y in range(height):
        for x in range(width):
            code = int(chars[y, x])
            if code == 0:
                continue
            char = chr(code)
            color = _COLOR_TABLE.get(int(color_array[y, x]) % 16, (255, 255, 255))
            draw.text((x * cell_w, y * cell_h), char, fill=color, font=font)
    return np.asarray(image, dtype=np.uint8)
