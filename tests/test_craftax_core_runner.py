from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np

from nanohorizon.craftax_core.checkpoint import CheckpointCodec, state_digest
from nanohorizon.craftax_core.media import persist_media
from nanohorizon.craftax_core.modalities import RenderMode
from nanohorizon.craftax_core.texture_cache import ensure_texture_cache
from tests._craftax_fakes import make_test_runner


def test_runner_checkpoint_restore_and_rewind(monkeypatch, tmp_path):
    runner = make_test_runner(monkeypatch, render_mode=RenderMode.BOTH)

    reset_output = runner.reset()
    assert reset_output.render.text == "position=1|ticks=[1]"

    first = runner.step(1)
    checkpoint = runner.checkpoint(label="after_first")
    second = runner.step(1)
    assert second.render.text != first.render.text

    restored = runner.restore(checkpoint)
    assert restored.render.text == first.render.text
    rewound = runner.rewind_episode()
    assert rewound.render.text == "position=1|ticks=[1]"

    path = tmp_path / "checkpoint.bin"
    CheckpointCodec.save(checkpoint, path)
    loaded = CheckpointCodec.load(path)
    assert loaded.label == "after_first"
    assert state_digest({"position": loaded.state.position}) == state_digest({"position": first.info["position"]})


def test_texture_cache_is_idempotent():
    first = ensure_texture_cache()
    second = ensure_texture_cache()
    assert first["status"] == "ok"
    assert second["status"] == "ok"


def test_texture_cache_syncs_with_shared_root(monkeypatch, tmp_path):
    shared_root = tmp_path / "shared"
    full_target = tmp_path / "pkg" / "texture_cache.pbz2"
    classic_target = tmp_path / "pkg" / "texture_cache_classic.pbz2"
    full_target.parent.mkdir(parents=True, exist_ok=True)
    full_target.write_bytes(b"full-cache")
    classic_target.write_bytes(b"classic-cache")

    craftax_pkg = types.ModuleType("craftax")
    craftax_full_pkg = types.ModuleType("craftax.craftax")
    craftax_classic_pkg = types.ModuleType("craftax.craftax_classic")
    full_constants = types.ModuleType("craftax.craftax.constants")
    classic_constants = types.ModuleType("craftax.craftax_classic.constants")
    full_constants.TEXTURE_CACHE_FILE = str(full_target)
    classic_constants.TEXTURE_CACHE_FILE = str(classic_target)

    monkeypatch.setenv("NANOHORIZON_CRAFTAX_CACHE_DIR", str(shared_root))
    monkeypatch.setitem(sys.modules, "craftax", craftax_pkg)
    monkeypatch.setitem(sys.modules, "craftax.craftax", craftax_full_pkg)
    monkeypatch.setitem(sys.modules, "craftax.craftax.constants", full_constants)
    monkeypatch.setitem(sys.modules, "craftax.craftax_classic", craftax_classic_pkg)
    monkeypatch.setitem(sys.modules, "craftax.craftax_classic.constants", classic_constants)

    initial = ensure_texture_cache()
    assert Path(initial["full"]["shared_texture_cache_file"]).exists()
    assert Path(initial["classic"]["shared_texture_cache_file"]).exists()

    full_target.unlink()
    classic_target.unlink()

    restored = ensure_texture_cache()
    assert full_target.read_bytes() == b"full-cache"
    assert classic_target.read_bytes() == b"classic-cache"
    assert restored["full"]["restored_from_shared_cache"] is True
    assert restored["classic"]["restored_from_shared_cache"] is True


def test_media_persist_smoke(tmp_path):
    frames = [np.zeros((8, 8, 3), dtype=np.uint8), np.full((8, 8, 3), 255, dtype=np.uint8)]
    artifacts = persist_media(frames=frames, output_dir=tmp_path / "media", fps=4, write_mp4=False)
    assert artifacts.frames_dir
    assert artifacts.gif_path
