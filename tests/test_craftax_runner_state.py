from __future__ import annotations

from dataclasses import dataclass

import nanohorizon.craftax_core.checkpoint as checkpoint_module
import nanohorizon.craftax_core.runner as runner_module


class FakeRandom:
    @staticmethod
    def PRNGKey(seed: int):
        return (int(seed), 0)

    @staticmethod
    def split(key):
        seed, counter = key
        return (seed, counter + 1), (seed, counter + 2)


class FakeTreeUtil:
    @staticmethod
    def tree_flatten(tree):
        return [tree], type(tree).__name__


class FakeJax:
    random = FakeRandom()
    tree_util = FakeTreeUtil()
    Array = tuple


@dataclass
class MutableState:
    position: int = 0
    history: list[int] | None = None

    def __post_init__(self) -> None:
        if self.history is None:
            self.history = []


class MutableEnv:
    default_params = {"terminal_position": 10}

    def reset(self, key, params=None):
        del key, params
        return None, MutableState(position=0, history=[0])

    def step(self, key, state, action: int, params=None):
        del key, params
        state.position += int(action) + 1
        state.history.append(state.position)
        done = state.position >= 10
        return None, state, float(state.position), done, {"position": state.position}


class Renderer:
    def render(self, state, mode):
        del mode

        class RenderBundle:
            text = f"position={state.position}"
            pixels = None
            structured = {"position": state.position, "history": list(state.history)}

        return RenderBundle()


def test_episode_start_rewind_isolated_from_in_place_state_mutation(monkeypatch) -> None:
    fake_jax = FakeJax()
    monkeypatch.setattr(runner_module, "jax", fake_jax)
    monkeypatch.setattr(checkpoint_module, "jax", fake_jax)

    runner = runner_module.DeterministicCraftaxRunner(
        env=MutableEnv(),
        renderer=Renderer(),
        seed=0,
    )

    reset_output = runner.reset()
    baseline = reset_output.render.structured

    runner.step(1)
    runner.step(2)

    rewind_output = runner.rewind_episode()

    assert rewind_output.render.structured == baseline
    assert runner.episode_start.state.position == 0
    assert runner.episode_start.state.history == [0]
