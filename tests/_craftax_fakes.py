from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from nanohorizon.craftax_core.modalities import CallableRenderer, RenderMode
from nanohorizon.craftax_core.runner import DeterministicCraftaxRunner


class FakeRandom:
    @staticmethod
    def PRNGKey(seed: int) -> tuple[int, int]:
        return (int(seed), 0)

    @staticmethod
    def split(key: tuple[int, int]) -> tuple[tuple[int, int], tuple[int, int]]:
        seed, counter = key
        return (seed, counter + 1), (seed, counter + 2)


class FakeTreeUtil:
    @staticmethod
    def tree_flatten(tree):
        if isinstance(tree, dict):
            keys = sorted(tree.keys())
            return [tree[key] for key in keys], tuple(keys)
        return [tree], type(tree).__name__


class FakeJax:
    random = FakeRandom()
    tree_util = FakeTreeUtil()
    Array = tuple


@dataclass(frozen=True)
class DummyState:
    position: int
    rng_ticks: tuple[int, ...]
    achievements: tuple[str, ...] = ()


class KeyedDummyEnv:
    default_params = {"terminal_position": 64}

    def reset(self, key, params=None):
        del params
        tick = int(key[1]) + int(int(key[0]) % 2 == 0)
        return None, DummyState(position=tick, rng_ticks=(tick,))

    def step(self, key, state, action: int, params=None):
        limit = int((params or self.default_params).get("terminal_position", 64))
        tick = int(key[1]) + int(int(key[0]) % 2 == 0)
        next_position = int(state.position) + int(action) + tick
        achievements = list(state.achievements)
        if next_position >= 5 and "collect_wood" not in achievements:
            achievements.append("collect_wood")
        if next_position >= 10 and "collect_sapling" not in achievements:
            achievements.append("collect_sapling")
        next_state = DummyState(
            position=next_position,
            rng_ticks=tuple([*state.rng_ticks, tick]),
            achievements=tuple(achievements),
        )
        reward = float(next_position)
        done = next_position >= limit
        info = {"position": next_position, "rng_tick": tick}
        return None, next_state, reward, done, info


def build_renderer() -> CallableRenderer:
    return CallableRenderer(
        text_fn=lambda state: f"position={state.position}|ticks={list(state.rng_ticks)}",
        pixels_fn=lambda state: np.full((4, 4, 3), state.position % 255, dtype=np.uint8),
        structured_fn=lambda state: {
            "position": state.position,
            "rng_ticks": list(state.rng_ticks),
            "achievements": list(state.achievements),
        },
    )


def make_test_runner(
    monkeypatch,
    *,
    seed: int = 7,
    render_mode: RenderMode = RenderMode.BOTH,
):
    import nanohorizon.craftax_core.checkpoint as checkpoint_module
    import nanohorizon.craftax_core.runner as runner_module

    fake_jax = FakeJax()
    monkeypatch.setattr(runner_module, "jax", fake_jax)
    monkeypatch.setattr(checkpoint_module, "jax", fake_jax)
    return DeterministicCraftaxRunner(
        env=KeyedDummyEnv(),
        renderer=build_renderer(),
        seed=seed,
        render_mode=render_mode,
    )
