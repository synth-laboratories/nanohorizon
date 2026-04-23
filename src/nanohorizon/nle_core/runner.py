from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from nanohorizon.craftax_core.modalities import RenderBundle, RenderMode

from .metadata import ACTION_NAME_TO_VALUE
from .rendering import render_pixels, render_text
from .scout import ScoutTracker

OBSERVATION_KEYS = ("message", "blstats", "glyphs", "tty_chars", "tty_colors", "tty_cursor")


@dataclass(frozen=True)
class NLEStepOutput:
    render: RenderBundle
    reward: float
    done: bool
    info: Mapping[str, Any]
    step_index: int
    episode_index: int
    action: int | None = None


def make_nle_env(
    *,
    env_id: str = "NetHackChallenge-v0",
    seed: int | None = None,
    max_episode_steps: int | None = None,
    savedir: str | None = None,
) -> Any:
    del seed
    actions = tuple(ACTION_NAME_TO_VALUE.values())
    base_kwargs: dict[str, Any] = {
        "observation_keys": OBSERVATION_KEYS,
        "actions": actions,
        "allow_all_yn_questions": True,
        "allow_all_modes": True,
    }
    if max_episode_steps is not None:
        base_kwargs["max_episode_steps"] = int(max_episode_steps)
    if savedir is not None:
        base_kwargs["savedir"] = savedir

    if env_id in {"NLE", "NLE-v0", "NetHackChallenge-v0"}:
        try:
            from nle.env import base
        except Exception as exc:  # pragma: no cover - optional dependency
            raise ImportError("nle is required for the NLE runtime") from exc
        return base.NLE(**base_kwargs)

    try:
        import gymnasium as gym
    except Exception as exc:  # pragma: no cover - optional dependency
        raise ImportError("gymnasium is required for the NLE runtime") from exc

    kwargs: dict[str, Any] = {
        "observation_keys": OBSERVATION_KEYS,
        "actions": actions,
        "allow_all_yn_questions": True,
    }
    if max_episode_steps is not None:
        kwargs["max_episode_steps"] = int(max_episode_steps)
    if savedir is not None:
        kwargs["savedir"] = savedir
    try:
        return gym.make(env_id, **kwargs)
    except TypeError as exc:
        if "actions" not in str(exc):
            raise
        try:
            from nle.env import base
        except Exception as import_exc:  # pragma: no cover - optional dependency
            raise ImportError("nle is required for the NLE runtime") from import_exc
        return base.NLE(**base_kwargs)


class DeterministicNLERunner:
    def __init__(
        self,
        *,
        env: Any | None = None,
        env_id: str = "NetHackChallenge-v0",
        seed: int = 0,
        max_episode_steps: int | None = None,
        render_mode: RenderMode = RenderMode.TEXT,
        savedir: str | None = None,
    ) -> None:
        self.seed = int(seed)
        self.render_mode = render_mode
        self.env = env if env is not None else make_nle_env(
            env_id=env_id,
            seed=self.seed,
            max_episode_steps=max_episode_steps,
            savedir=savedir,
        )
        self.scout = ScoutTracker()
        self._episode_index = -1
        self._step_index = 0
        self.action_history: list[int] = []
        self.last_observation: Mapping[str, Any] | None = None
        self.last_info: Mapping[str, Any] = {}

    def _render(self, observation: Mapping[str, Any], info: Mapping[str, Any] | None = None) -> RenderBundle:
        text = render_text(observation, info=info) if self.render_mode.wants_text else None
        pixels = render_pixels(observation) if self.render_mode.wants_pixels else None
        return RenderBundle(mode=self.render_mode, text=text, pixels=pixels, state_view=observation)

    def reset(self) -> NLEStepOutput:
        self.scout.reset()
        self._episode_index += 1
        self._step_index = 0
        self.action_history = []
        try:
            obs, info = self.env.reset(seed=self.seed + self._episode_index)
        except TypeError:
            obs, info = self.env.reset(), {}
            if isinstance(obs, tuple) and len(obs) == 2:
                obs, info = obs
        if info is None:
            info = {}
        self.last_observation = obs
        self.last_info = info
        return NLEStepOutput(
            render=self._render(obs, info),
            reward=0.0,
            done=False,
            info=info,
            step_index=self._step_index,
            episode_index=self._episode_index,
            action=None,
        )

    def step(self, action_index: int) -> NLEStepOutput:
        if self.last_observation is None:
            raise RuntimeError("call reset() before step()")
        result = self.env.step(int(action_index))
        if len(result) == 5:
            obs, native_reward, terminated, truncated, info = result
            done = bool(terminated or truncated)
        else:
            obs, native_reward, done, info = result
        if info is None:
            info = {}
        scout_reward = self.scout.reward(obs)
        self._step_index += 1
        self.action_history.append(int(action_index))
        self.last_observation = obs
        self.last_info = {
            **dict(info),
            "native_reward": float(native_reward),
            "scout_reward": float(scout_reward),
            "scout_score": float(self.scout.total),
        }
        return NLEStepOutput(
            render=self._render(obs, self.last_info),
            reward=float(scout_reward),
            done=bool(done),
            info=self.last_info,
            step_index=self._step_index,
            episode_index=self._episode_index,
            action=int(action_index),
        )

    def step_many(self, actions: list[int]) -> list[NLEStepOutput]:
        outputs: list[NLEStepOutput] = []
        for action in actions:
            output = self.step(int(action))
            outputs.append(output)
            if output.done:
                break
        return outputs


def make_runner(
    *,
    seed: int = 0,
    render_mode: RenderMode = RenderMode.TEXT,
    env_id: str = "NetHackChallenge-v0",
    max_episode_steps: int | None = None,
    savedir: str | None = None,
) -> DeterministicNLERunner:
    return DeterministicNLERunner(
        seed=seed,
        render_mode=render_mode,
        env_id=env_id,
        max_episode_steps=max_episode_steps,
        savedir=savedir,
    )


def native_score_from_observation(observation: Mapping[str, Any] | None) -> float:
    if observation is None:
        return 0.0
    blstats = observation.get("blstats")
    if blstats is None:
        return 0.0
    stats = np.asarray(blstats).reshape(-1)
    return float(stats[9]) if len(stats) > 9 else 0.0
