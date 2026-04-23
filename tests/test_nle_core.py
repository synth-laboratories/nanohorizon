from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from fastapi.testclient import TestClient

from nanohorizon.craftax_core.modalities import RenderBundle, RenderMode
from nanohorizon.nle_core.metadata import (
    ACTION_NAME_TO_VALUE,
    DEFAULT_ACTION_NAMES,
    PRIMARY_TOOL_NAME,
)
from nanohorizon.nle_core.rendering import render_pixels, render_text
from nanohorizon.nle_core.rollout import run_rollout_request, sanitize_nle_actions
from nanohorizon.nle_core.scout import ScoutTracker


def _obs(*, visible: int, dnum: int = 0, dlevel: int = 1, cmap_off: int = 999) -> dict[str, np.ndarray]:
    glyphs = np.full((4, 4), cmap_off, dtype=np.int32)
    glyphs.reshape(-1)[:visible] = np.arange(visible)
    blstats = np.zeros(27, dtype=np.int32)
    blstats[23] = dnum
    blstats[24] = dlevel
    return {"glyphs": glyphs, "blstats": blstats}


def test_scout_reward_increases_without_double_counting_same_level():
    tracker = ScoutTracker(cmap_off=999)

    assert tracker.reward(_obs(visible=3, cmap_off=999)) == 3.0
    assert tracker.reward(_obs(visible=3, cmap_off=999)) == 0.0
    assert tracker.reward(_obs(visible=5, cmap_off=999)) == 2.0
    assert tracker.total == 5.0


def test_scout_reward_tracks_levels_independently():
    tracker = ScoutTracker(cmap_off=999)

    assert tracker.reward(_obs(visible=4, dnum=0, dlevel=1, cmap_off=999)) == 4.0
    assert tracker.reward(_obs(visible=2, dnum=0, dlevel=2, cmap_off=999)) == 2.0
    assert tracker.reward(_obs(visible=5, dnum=0, dlevel=1, cmap_off=999)) == 1.0
    assert tracker.total == 7.0


def test_nle_action_catalog_and_sanitizer_expose_tool_actions():
    assert PRIMARY_TOOL_NAME == "nle_interact"
    assert DEFAULT_ACTION_NAMES
    first = DEFAULT_ACTION_NAMES[0]
    assert first in ACTION_NAME_TO_VALUE
    assert sanitize_nle_actions([first, f"try {first} now", "not_an_action"]) == [first]


def test_nle_text_and_pixel_rendering_from_terminal_observation():
    tty = np.array([[ord("@"), ord("."), ord(" ")], [ord("H"), ord("P"), ord(":")]], dtype=np.uint8)
    obs = {
        "message": np.array([ord("H"), ord("i")], dtype=np.uint8),
        "blstats": np.zeros(27, dtype=np.int32),
        "tty_chars": tty,
        "tty_colors": np.ones_like(tty, dtype=np.uint8),
    }

    text = render_text(obs)
    pixels = render_pixels(obs)

    assert "message: Hi" in text
    assert "terminal:" in text
    assert "@." in text
    assert pixels is not None
    assert pixels.ndim == 3
    assert pixels.shape[-1] == 3


def test_nle_http_shim_health_task_info_and_mock_rollout(monkeypatch):
    import nanohorizon.nle_core.http_shim as shim

    monkeypatch.setattr(shim, "_nle_available", lambda: (True, None))
    monkeypatch.setattr(
        shim,
        "run_rollout_request",
        lambda request: {
            "rollout_id": "rollout_test",
            "trace_correlation_id": request["trace_correlation_id"],
            "success_status": "success",
            "reward_info": {
                "outcome_reward": 2.0,
                "outcome_objectives": {"scout_score": 2.0},
                "details": {"scout_score": 2.0},
            },
            "trace": {"inference": {"turns": []}},
            "metadata": {"environment_family": "nle", "task_id": "nethack_scout", "scout_score": 2.0},
            "artifact": [{"turns": []}],
        },
    )
    app = shim.create_app()
    client = TestClient(app)

    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["upstream_ready"] is True

    task_info = client.get("/task_info")
    assert task_info.status_code == 200
    assert task_info.json()["environment_family"] == "nle"
    assert task_info.json()["tool_name"] == "nle_interact"

    rollout = client.post(
        "/rollout",
        json={
            "trace_correlation_id": "trace_1",
            "env": {"seed": 1, "config": {"max_steps": 1}},
            "policy": {"config": {"inference_url": "http://example.test", "model": "demo", "api_key": ""}},
        },
    )
    assert rollout.status_code == 200
    assert rollout.json()["metadata"]["task_id"] == "nethack_scout"


def test_nle_rollout_repairs_invalid_actions_and_reports_scout(monkeypatch):
    import nanohorizon.nle_core.rollout as rollout_module

    valid_action = DEFAULT_ACTION_NAMES[0]
    valid_value = ACTION_NAME_TO_VALUE[valid_action]

    class FakeRunner:
        def __init__(self) -> None:
            self.last_observation = {"blstats": np.zeros(27, dtype=np.int32)}
            self.action_history: list[int] = []

        def reset(self):
            return SimpleNamespace(
                done=False,
                render=RenderBundle(mode=RenderMode.TEXT, text="nle obs 0", pixels=None),
                reward=0.0,
                info={},
                step_index=0,
                episode_index=0,
            )

        def step_many(self, actions):
            outputs = []
            for action in actions:
                self.action_history.append(int(action))
                outputs.append(
                    SimpleNamespace(
                        done=True,
                        reward=3.0 if int(action) == valid_value else 0.0,
                        render=RenderBundle(mode=RenderMode.TEXT, text="nle obs 1", pixels=None),
                    )
                )
            return outputs

    payloads = [
        {"choices": [{"message": {"content": "bad", "tool_calls": []}}]},
        {
            "choices": [
                {
                    "message": {
                        "content": "",
                        "tool_calls": [
                            {
                                "function": {
                                    "name": PRIMARY_TOOL_NAME,
                                    "arguments": {"actions_list": [valid_action]},
                                }
                            }
                        ],
                    }
                }
            ]
        },
    ]

    monkeypatch.setattr(rollout_module, "make_runner", lambda **_: FakeRunner())
    monkeypatch.setattr(rollout_module, "_chat_completion", lambda **_: payloads.pop(0))

    result = run_rollout_request(
        {
            "trace_correlation_id": "nle_trace",
            "env": {"seed": 1, "config": {"max_steps": 1}},
            "policy": {
                "config": {
                    "inference_url": "http://example.test",
                    "model": "demo",
                    "api_key": "",
                    "target_action_batch_size": 1,
                    "min_action_batch_size": 1,
                }
            },
        }
    )

    assert result["reward_info"]["outcome_reward"] == 3.0
    assert result["reward_info"]["outcome_objectives"]["scout_score"] == 3.0
    assert result["reward_info"]["details"]["scout_score"] == 3.0
    assert result["metadata"]["environment_family"] == "nle"
    assert result["metadata"]["task_id"] == "nethack_scout"
    assert result["metadata"]["action_history"] == [valid_value]
    assert result["trace"]["inference"]["turns"][0]["invalid_parse"] is True

