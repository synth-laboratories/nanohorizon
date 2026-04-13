from __future__ import annotations

import sys
import types
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from nanohorizon.craftax_core.checkpoint import CheckpointCodec, state_digest
from nanohorizon.craftax_core.media import persist_media
from nanohorizon.craftax_core.modalities import RenderMode
from nanohorizon.craftax_core.texture_cache import ensure_texture_cache
from tests._craftax_fakes import make_test_runner


class _MutableFakeRandom:
    @staticmethod
    def PRNGKey(seed: int) -> tuple[int, int]:
        return (int(seed), 0)

    @staticmethod
    def split(key: tuple[int, int]) -> tuple[tuple[int, int], tuple[int, int]]:
        seed, counter = key
        return (seed, counter + 1), (seed, counter + 2)


class _MutableFakeTreeUtil:
    @staticmethod
    def tree_flatten(tree):
        if isinstance(tree, dict):
            keys = sorted(tree.keys())
            return [tree[key] for key in keys], tuple(keys)
        return [tree], type(tree).__name__


class _MutableFakeJax:
    random = _MutableFakeRandom()
    tree_util = _MutableFakeTreeUtil()
    Array = tuple


@dataclass
class _MutableState:
    position: int
    ticks: list[int]


class _InPlaceEnv:
    default_params = {"terminal_position": 99}

    def reset(self, key, params=None):
        del params
        return None, _MutableState(position=0, ticks=[int(key[1])])

    def step(self, key, state, action: int, params=None):
        del params
        state.position += int(action) + int(key[1])
        state.ticks.append(int(key[1]))
        return None, state, float(state.position), False, {"position": state.position, "tick": int(key[1])}


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


def test_checkpoint_snapshots_mutable_runtime_state_by_default(monkeypatch):
    import nanohorizon.craftax_core.checkpoint as checkpoint_module
    import nanohorizon.craftax_core.runner as runner_module

    fake_jax = _MutableFakeJax()
    monkeypatch.setattr(runner_module, "jax", fake_jax)
    monkeypatch.setattr(checkpoint_module, "jax", fake_jax)

    results = []
    for seed in range(4):
        runner = runner_module.DeterministicCraftaxRunner(
            env=_InPlaceEnv(),
            renderer=runner_module.CallableRenderer(text_fn=lambda state: f"{state.position}:{state.ticks}"),
            seed=seed,
        )
        runner.reset()
        first = runner.step(1)
        checkpoint = runner.checkpoint(label="probe")
        runner.step(2)
        restored = runner.restore(checkpoint)
        results.append((seed, first.render.text, restored.render.text, restored.info))

    assert all(restored_text == first_text for _, first_text, restored_text, _ in results)
    assert all(restored_info == {"position": 4, "tick": 3} for _, _, _, restored_info in results)


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


def test_texture_cache_restores_shared_files_before_craftax_import(monkeypatch, tmp_path):
    package_root = tmp_path / "fake_site"
    full_assets = package_root / "craftax" / "craftax" / "assets"
    classic_assets = package_root / "craftax" / "craftax_classic" / "assets"
    full_assets.mkdir(parents=True, exist_ok=True)
    classic_assets.mkdir(parents=True, exist_ok=True)

    for init_file in (
        package_root / "craftax" / "__init__.py",
        package_root / "craftax" / "craftax" / "__init__.py",
        package_root / "craftax" / "craftax_classic" / "__init__.py",
    ):
        init_file.write_text("", encoding="utf-8")

    full_target = full_assets / "texture_cache.pbz2"
    classic_target = classic_assets / "texture_cache_classic.pbz2"
    (package_root / "craftax" / "craftax" / "constants.py").write_text(
        f'TEXTURE_CACHE_FILE = r"{full_target}"\n',
        encoding="utf-8",
    )
    (package_root / "craftax" / "craftax_classic" / "constants.py").write_text(
        f'TEXTURE_CACHE_FILE = r"{classic_target}"\n',
        encoding="utf-8",
    )

    shared_root = tmp_path / "shared"
    shared_full = shared_root / "full" / full_target.name
    shared_classic = shared_root / "classic" / classic_target.name
    shared_full.parent.mkdir(parents=True, exist_ok=True)
    shared_classic.parent.mkdir(parents=True, exist_ok=True)
    shared_full.write_bytes(b"shared-full-cache")
    shared_classic.write_bytes(b"shared-classic-cache")

    monkeypatch.setenv("NANOHORIZON_CRAFTAX_CACHE_DIR", str(shared_root))
    monkeypatch.syspath_prepend(str(package_root))
    for module_name in list(sys.modules):
        if module_name == "craftax" or module_name.startswith("craftax."):
            monkeypatch.delitem(sys.modules, module_name, raising=False)

    restored = ensure_texture_cache()

    assert full_target.read_bytes() == b"shared-full-cache"
    assert classic_target.read_bytes() == b"shared-classic-cache"
    assert restored["full"]["restored_from_shared_cache"] is True
    assert restored["classic"]["restored_from_shared_cache"] is True


def test_media_persist_smoke(tmp_path):
    frames = [np.zeros((8, 8, 3), dtype=np.uint8), np.full((8, 8, 3), 255, dtype=np.uint8)]
    artifacts = persist_media(frames=frames, output_dir=tmp_path / "media", fps=4, write_mp4=False)
    assert artifacts.frames_dir
    assert artifacts.gif_path
