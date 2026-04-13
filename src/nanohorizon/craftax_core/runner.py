"""CLI and helpers for the Craftax interface-shaped candidate."""

from __future__ import annotations

import copy
import json
import sys
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, Protocol

from .metadata import PromptContext, RewardHistoryEntry, RewardHistoryWindow, StructuredObservation
from .checkpoint import Checkpoint
from .modalities import CallableRenderer, RenderBundle, RenderMode

try:
    import jax
except Exception:  # pragma: no cover - optional dependency
    jax = None


# ---------------------------------------------------------------------------
# Interface v1 helpers
# ---------------------------------------------------------------------------


def summarize_reward_history(
    history: Iterable[Mapping[str, Any] | RewardHistoryEntry],
) -> list[dict[str, Any]]:
    window = RewardHistoryWindow()
    for item in history:
        if isinstance(item, RewardHistoryEntry):
            window.append(item)
        else:
            window.append(
                RewardHistoryEntry(
                    action=item.get("action"),
                    observation_summary=str(item.get("observation_summary", "")),
                    reward_delta=float(item.get("reward_delta", 0.0)),
                )
            )
    return window.to_prompt_payload()


def build_prompt_context(
    observation: Any,
    history: Iterable[Mapping[str, Any] | RewardHistoryEntry] = (),
    *,
    metadata: Mapping[str, Any] | None = None,
) -> PromptContext:
    return PromptContext(
        observation=StructuredObservation.from_observation(observation),
        reward_history=RewardHistoryWindow(
            [entry if isinstance(entry, RewardHistoryEntry) else RewardHistoryEntry(
                action=entry.get("action"),
                observation_summary=str(entry.get("observation_summary", "")),
                reward_delta=float(entry.get("reward_delta", 0.0)),
            ) for entry in history]
        ),
        metadata=dict(metadata or {}),
    )


def _load_payload_from_stdin() -> dict[str, Any]:
    raw = sys.stdin.read().strip()
    if not raw:
        return {}
    return json.loads(raw)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] in {"-h", "--help"}:
        print(
            "Usage: python -m nanohorizon.craftax_core.runner < input.json\n"
            "Reads an observation/history payload from stdin and emits a prompt-ready JSON object."
        )
        return 0

    payload = _load_payload_from_stdin()
    from .http_shim import render_prompt_turn

    prompt = render_prompt_turn(
        payload.get("observation"),
        payload.get("history", ()),
        metadata=payload.get("metadata"),
    )
    json.dump(prompt, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# ---------------------------------------------------------------------------
# DeterministicCraftaxRunner — preserved for upstream.py compatibility
# ---------------------------------------------------------------------------


class EnvLike(Protocol):
    default_params: Any

    def reset(self, key: Any, params: Any = None): ...

    def step(self, key: Any, state: Any, action: int, params: Any = None): ...


@dataclass(frozen=True)
class StepOutput:
    render: RenderBundle
    reward: float
    done: bool
    info: Mapping[str, Any]
    step_index: int
    episode_index: int
    action: int | None = None


class DeterministicCraftaxRunner:
    def __init__(
        self,
        *,
        env: EnvLike | Callable[[], EnvLike],
        renderer: CallableRenderer,
        seed: int = 0,
        params: Any = None,
        render_mode: RenderMode = RenderMode.TEXT,
    ) -> None:
        if jax is None:
            raise RuntimeError("jax is required for DeterministicCraftaxRunner")
        self._env_or_factory = env
        self._renderer = renderer
        self.render_mode = render_mode
        self.seed = int(seed)
        self._root_rng = jax.random.PRNGKey(self.seed)
        self._next_rng = self._root_rng
        self._episode_index = -1
        self._step_index = 0
        self.action_history: list[int] = []
        self.env = env() if callable(env) else env
        self.params = self.env.default_params if params is None else params
        self.state: Any = None
        self.last_info: Mapping[str, Any] = {}
        self.episode_start: Checkpoint | None = None

    def _split(self) -> Any:
        key, next_key = jax.random.split(self._next_rng)
        self._next_rng = next_key
        return key

    def reset(self) -> StepOutput:
        reset_key = self._split()
        _obs, self.state = self.env.reset(reset_key, self.params)
        self._episode_index += 1
        self._step_index = 0
        self.action_history = []
        self.last_info = {}
        self.episode_start = self.checkpoint(label="episode_start")
        return StepOutput(
            render=self._renderer.render(self.state, self.render_mode),
            reward=0.0,
            done=False,
            info=self.last_info,
            step_index=self._step_index,
            episode_index=self._episode_index,
            action=None,
        )

    def step(self, action: int) -> StepOutput:
        if self.state is None:
            raise RuntimeError("call reset() before step()")
        step_key = self._split()
        _obs, state, reward, done, info = self.env.step(step_key, self.state, int(action), self.params)
        self.state = state
        self.last_info = info if isinstance(info, Mapping) else {}
        self._step_index += 1
        self.action_history.append(int(action))
        return StepOutput(
            render=self._renderer.render(self.state, self.render_mode),
            reward=float(reward),
            done=bool(done),
            info=self.last_info,
            step_index=self._step_index,
            episode_index=self._episode_index,
            action=int(action),
        )

    def step_many(self, actions: list[int]) -> list[StepOutput]:
        outputs: list[StepOutput] = []
        for action in actions:
            output = self.step(int(action))
            outputs.append(output)
            if output.done:
                break
        return outputs

    def checkpoint(
        self,
        *,
        label: str | None = None,
        copy_state: bool = False,
        metadata: Mapping[str, Any] | None = None,
    ) -> Checkpoint:
        if self.state is None:
            raise RuntimeError("cannot checkpoint before reset")
        stored_state = copy.deepcopy(self.state) if copy_state else self.state
        stored_rng = copy.deepcopy(self._next_rng) if copy_state else self._next_rng
        stored_params = copy.deepcopy(self.params) if copy_state else self.params
        return Checkpoint(
            version=1,
            seed=self.seed,
            episode_index=self._episode_index,
            step_index=self._step_index,
            next_rng=stored_rng,
            state=stored_state,
            params=stored_params,
            action_history=tuple(self.action_history),
            label=label,
            metadata=dict(metadata or {}),
        )

    def restore(self, checkpoint: Checkpoint) -> StepOutput:
        self.state = checkpoint.state
        self.params = checkpoint.params
        self._next_rng = checkpoint.next_rng
        self._episode_index = checkpoint.episode_index
        self._step_index = checkpoint.step_index
        self.action_history = list(checkpoint.action_history)
        return StepOutput(
            render=self._renderer.render(self.state, self.render_mode),
            reward=0.0,
            done=False,
            info=self.last_info,
            step_index=self._step_index,
            episode_index=self._episode_index,
            action=None,
        )

    def rewind_episode(self) -> StepOutput:
        if self.episode_start is None:
            raise RuntimeError("no episode start checkpoint available")
        return self.restore(self.episode_start)
