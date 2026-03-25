from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from nanohorizon.shared.common import ensure_dir

try:
    from PIL import Image
except Exception:  # pragma: no cover - optional dependency
    Image = None


@dataclass(frozen=True)
class MediaArtifacts:
    frames_dir: str | None = None
    gif_path: str | None = None
    mp4_path: str | None = None


def _to_uint8_frame(frame: np.ndarray) -> np.ndarray:
    array = np.asarray(frame)
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    return array


def save_png_frames(*, frames: list[np.ndarray], output_dir: str | Path) -> str | None:
    if not frames or Image is None:
        return None
    target = ensure_dir(output_dir)
    for index, frame in enumerate(frames):
        Image.fromarray(_to_uint8_frame(frame)).save(target / f"frame_{index:04d}.png")
    return str(target)


def save_gif(*, frames: list[np.ndarray], output_path: str | Path, fps: int = 6) -> str | None:
    if not frames or Image is None:
        return None
    duration_ms = max(1, int(1000 / max(1, int(fps))))
    target = Path(output_path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    pil_frames = [Image.fromarray(_to_uint8_frame(frame)) for frame in frames]
    pil_frames[0].save(
        target,
        save_all=True,
        append_images=pil_frames[1:],
        duration=duration_ms,
        loop=0,
    )
    return str(target)


def save_mp4(*, frames: list[np.ndarray], output_path: str | Path, fps: int = 6) -> str | None:
    if not frames:
        return None
    try:
        import imageio.v3 as iio
    except Exception:  # pragma: no cover - optional dependency
        return None
    target = Path(output_path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(target, [_to_uint8_frame(frame) for frame in frames], fps=max(1, int(fps)))
    return str(target)


def persist_media(
    *,
    frames: list[np.ndarray],
    output_dir: str | Path,
    fps: int = 6,
    write_mp4: bool = True,
) -> MediaArtifacts:
    target = ensure_dir(output_dir)
    frames_dir = save_png_frames(frames=frames, output_dir=target / "frames")
    gif_path = save_gif(frames=frames, output_path=target / "rollout.gif", fps=fps)
    mp4_path = save_mp4(frames=frames, output_path=target / "rollout.mp4", fps=fps) if write_mp4 else None
    return MediaArtifacts(frames_dir=frames_dir, gif_path=gif_path, mp4_path=mp4_path)

