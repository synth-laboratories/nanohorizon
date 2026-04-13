from __future__ import annotations

from types import SimpleNamespace

from nanohorizon.craftax_core.checkpoint import CheckpointCodec, state_digest
from nanohorizon.craftax_core.metadata import PRIMARY_TOOL_NAME
from nanohorizon.craftax_core.modalities import RenderBundle, RenderMode
from nanohorizon.craftax_core.rollout import ACTION_NAME_TO_INDEX, run_rollout_request
from nanohorizon.baselines.offline_sft import build_openai_sft_rows_from_rollouts
from nanohorizon.shared.craftax_data import summarize_achievement_frequencies
from nanohorizon.shared.openai_compat import extract_craftax_actions, sanitize_craftax_actions
from tests._craftax_fakes import make_test_runner


def _state_signature(runner) -> dict[str, object]:
    return {
        "position": runner.state.position,
        "rng_ticks": runner.state.rng_ticks,
        "achievements": runner.state.achievements,
        "action_history": tuple(runner.action_history),
    }


def test_rng_seed_and_action_sequence_are_deterministic(monkeypatch):
    runner_a = make_test_runner(monkeypatch, seed=11, render_mode=RenderMode.BOTH)
    runner_b = make_test_runner(monkeypatch, seed=11, render_mode=RenderMode.BOTH)
    runner_c = make_test_runner(monkeypatch, seed=12, render_mode=RenderMode.BOTH)

    runner_a.reset()
    runner_b.reset()
    runner_c.reset()

    runner_a.step_many([1, 2, 3])
    runner_b.step_many([1, 2, 3])
    runner_c.step_many([1, 2, 3])

    assert _state_signature(runner_a) == _state_signature(runner_b)
    assert state_digest(_state_signature(runner_a)) == state_digest(_state_signature(runner_b))
    assert state_digest(_state_signature(runner_a)) != state_digest(_state_signature(runner_c))


def test_checkpoint_restore_replays_identically_with_rng_sensitive_env(monkeypatch):
    runner = make_test_runner(monkeypatch, seed=5, render_mode=RenderMode.BOTH)
    runner.reset()
    runner.step(1)
    checkpoint = runner.checkpoint(label="midpoint", copy_state=True, metadata={"phase": "mid"})

    original_outputs = runner.step_many([2, 1, 2])
    original_signature = _state_signature(runner)

    runner.restore(checkpoint)
    replayed_outputs = runner.step_many([2, 1, 2])
    replay_signature = _state_signature(runner)

    assert [item.render.text for item in original_outputs] == [item.render.text for item in replayed_outputs]
    assert original_signature == replay_signature
    assert checkpoint.metadata["phase"] == "mid"


def test_render_modes_project_the_same_state(monkeypatch):
    outputs = {}
    for mode in (RenderMode.NONE, RenderMode.TEXT, RenderMode.PIXELS, RenderMode.BOTH):
        runner = make_test_runner(monkeypatch, seed=9, render_mode=mode)
        outputs[mode] = runner.reset().render

    assert outputs[RenderMode.NONE].text is None
    assert outputs[RenderMode.NONE].pixels is None
    assert outputs[RenderMode.TEXT].text == "position=1|ticks=[1]"
    assert outputs[RenderMode.TEXT].pixels is None
    assert outputs[RenderMode.PIXELS].text is None
    assert outputs[RenderMode.PIXELS].pixels is not None
    assert outputs[RenderMode.BOTH].text == "position=1|ticks=[1]"
    assert outputs[RenderMode.BOTH].pixels is not None
    assert outputs[RenderMode.TEXT].state_view == outputs[RenderMode.BOTH].state_view
    assert outputs[RenderMode.PIXELS].state_view == outputs[RenderMode.BOTH].state_view


def test_checkpoint_codec_supports_all_compressions(monkeypatch):
    runner = make_test_runner(monkeypatch, seed=4, render_mode=RenderMode.TEXT)
    runner.reset()
    runner.step_many([1, 2])
    checkpoint = runner.checkpoint(label="compressed", copy_state=True, metadata={"kind": "unit"})

    for compression in ("none", "gzip", "lzma"):
        encoded = CheckpointCodec.dumps(checkpoint, compression=compression)
        decoded = CheckpointCodec.loads(encoded, compression=compression)
        assert decoded.label == "compressed"
        assert decoded.metadata["kind"] == "unit"
        assert decoded.action_history == checkpoint.action_history
        assert decoded.next_rng == checkpoint.next_rng
        assert state_digest({"position": decoded.state.position}) == state_digest({"position": checkpoint.state.position})


def test_sft_rows_keep_tool_only_turns_with_reasoning():
    rows = build_openai_sft_rows_from_rollouts(
        [
            {
                "rollout_id": "rollout_tool_only",
                "trace_correlation_id": "trace_tool_only",
                "success_status": "success",
                "reward_info": {
                    "outcome_reward": 1.0,
                    "outcome_objectives": {"unique_achievements": 1.0},
                    "details": {"achievements": ["collect_wood"]},
                },
                "trace": {
                    "inference": {
                        "turns": [
                            {
                                "turn_index": 0,
                                "prompt_messages": [{"role": "user", "content": "obs"}],
                                "assistant_text": "",
                                "reasoning_text": "take a valid tool action",
                                "actions": ["move_right", "do"],
                                "decision_reward": 1.0,
                                "return_to_go": 1.0,
                                "trainable": True,
                                "invalid_parse": False,
                            }
                        ]
                    }
                },
                "artifact": [],
                "metadata": {"achievements": ["collect_wood"]},
            }
        ],
        reward_threshold=0.0,
    )

    assert len(rows) == 1
    assistant_message = rows[0]["messages"][-1]
    assert assistant_message["content"] == ""
    assert assistant_message["reasoning_content"] == "take a valid tool action"
    assert assistant_message["tool_calls"][0]["function"]["arguments"]["actions_list"] == [
        "move_right",
        "do",
    ]


def test_run_rollout_request_consumes_every_model_action(monkeypatch):
    import nanohorizon.craftax_core.rollout as rollout_module

    class FakeRunner:
        def __init__(self) -> None:
            self.state = {"steps": 0}
            self.action_history: list[int] = []

        def reset(self):
            return SimpleNamespace(
                done=False,
                render=RenderBundle(mode=RenderMode.TEXT, text="obs 0", pixels=None),
                reward=0.0,
                info={},
                step_index=0,
                episode_index=0,
            )

        def step_many(self, actions):
            outputs = []
            for action in actions:
                self.action_history.append(int(action))
                self.state["steps"] += 1
                outputs.append(
                    SimpleNamespace(
                        done=False,
                        reward=1.0 if int(action) == ACTION_NAME_TO_INDEX["do"] else 0.0,
                        render=RenderBundle(
                            mode=RenderMode.TEXT,
                            text=f"obs {self.state['steps']}",
                            pixels=None,
                        ),
                    )
                )
            return outputs

    fake_runner = FakeRunner()
    call_messages: list[list[dict[str, object]]] = []
    payloads = [
        {
            "choices": [
                {
                    "message": {
                        "content": "",
                        "tool_calls": [
                            {
                                "function": {
                                    "name": PRIMARY_TOOL_NAME,
                                    "arguments": {"actions_list": ["move_right", "move_up", "do", "move_left"]},
                                }
                            }
                        ],
                    }
                }
            ]
        },
        {
            "choices": [
                {
                    "message": {
                        "content": "",
                        "tool_calls": [
                            {
                                "function": {
                                    "name": PRIMARY_TOOL_NAME,
                                    "arguments": {"actions_list": ["move_down", "move_right", "move_up", "do", "move_left"]},
                                }
                            }
                        ],
                    }
                }
            ]
        },
    ]

    monkeypatch.setattr(rollout_module, "make_runner", lambda **_: fake_runner)

    def fake_chat_completion(**kwargs):  # type: ignore[no-untyped-def]
        call_messages.append(list(kwargs["messages"]))
        return payloads.pop(0)

    monkeypatch.setattr(rollout_module, "_chat_completion", fake_chat_completion)
    monkeypatch.setattr(
        rollout_module,
        "achievement_names_from_state",
        lambda state: ["collect_wood"] if int(state["steps"]) >= 5 else [],
    )

    result = run_rollout_request(
        {
            "trace_correlation_id": "uses_all_actions",
            "env": {"seed": 1, "config": {"env_kind": "full", "max_steps": 2, "episode_max_steps": 2}},
            "policy": {
                "config": {
                    "inference_url": "http://example.test",
                    "model": "demo",
                    "api_key": "",
                    "system_prompt": "demo",
                    "target_action_batch_size": 8,
                    "min_action_batch_size": 4,
                }
            },
        }
    )

    turns = result["trace"]["inference"]["turns"]
    proposed_actions = sum(len(turn["actions"]) for turn in turns)
    assert result["metadata"]["llm_call_count"] == 2
    assert len(result["metadata"]["action_history"]) == proposed_actions
    assert result["metadata"]["achievements"] == ["collect_wood"]
    assert turns[0]["invalid_parse"] is False
    assert turns[1]["invalid_parse"] is False
    assert "Recent rollout evidence:" in str(call_messages[0][1]["content"])
    assert "recent_actions=move_right, move_up, do, move_left" in str(call_messages[1][1]["content"])


def test_run_rollout_request_repairs_short_action_batches(monkeypatch):
    import nanohorizon.craftax_core.rollout as rollout_module

    class FakeRunner:
        def __init__(self) -> None:
            self.state = {"steps": 0}
            self.action_history: list[int] = []

        def reset(self):
            return SimpleNamespace(
                done=False,
                render=RenderBundle(mode=RenderMode.TEXT, text="obs 0", pixels=None),
                reward=0.0,
                info={},
                step_index=0,
                episode_index=0,
            )

        def step_many(self, actions):
            outputs = []
            for action in actions:
                self.action_history.append(int(action))
                self.state["steps"] += 1
                outputs.append(
                    SimpleNamespace(
                        done=False,
                        reward=0.0,
                        render=RenderBundle(mode=RenderMode.TEXT, text=f"obs {self.state['steps']}", pixels=None),
                    )
                )
            return outputs

    fake_runner = FakeRunner()
    payloads = [
        {
            "choices": [
                {
                    "message": {
                        "content": "",
                        "tool_calls": [
                            {
                                "function": {
                                    "name": PRIMARY_TOOL_NAME,
                                    "arguments": {"actions_list": ["move_right"]},
                                }
                            }
                        ],
                    }
                }
            ]
        },
        {
            "choices": [
                {
                    "message": {
                        "content": "",
                        "tool_calls": [
                            {
                                "function": {
                                    "name": PRIMARY_TOOL_NAME,
                                    "arguments": {
                                        "actions_list": ["move_right", "move_up", "move_left", "move_down"]
                                    },
                                }
                            }
                        ],
                    }
                }
            ]
        },
    ]

    monkeypatch.setattr(rollout_module, "make_runner", lambda **_: fake_runner)
    monkeypatch.setattr(rollout_module, "_chat_completion", lambda **_: payloads.pop(0))
    monkeypatch.setattr(rollout_module, "achievement_names_from_state", lambda state: [])

    result = run_rollout_request(
        {
            "trace_correlation_id": "repair_actions",
            "env": {"seed": 1, "config": {"env_kind": "full", "max_steps": 1, "episode_max_steps": 1}},
            "policy": {
                "config": {
                    "inference_url": "http://example.test",
                    "model": "demo",
                    "api_key": "",
                    "system_prompt": "demo",
                    "target_action_batch_size": 4,
                    "min_action_batch_size": 4,
                }
            },
        }
    )

    turn = result["trace"]["inference"]["turns"][0]
    assert result["metadata"]["llm_call_count"] == 2
    assert turn["invalid_parse"] is True
    assert len(turn["actions"]) == 4
    assert len(result["metadata"]["action_history"]) == 4


def test_openai_compat_action_extraction_and_frequency_summary():
    payload = {
        "choices": [
            {
                "message": {
                    "tool_calls": [
                        {
                            "function": {
                                "name": PRIMARY_TOOL_NAME,
                                "arguments": {
                                    "actions_list": ["MOVE_RIGHT", "move_right", "invalid", "do", "move_up"]
                                },
                            }
                        }
                    ]
                }
            }
        ]
    }
    assert sanitize_craftax_actions(["MOVE_RIGHT", "move_right", "bad", "do"]) == ["move_right", "do"]
    assert extract_craftax_actions(payload) == ["move_right", "do", "move_up"]

    summary = summarize_achievement_frequencies(
        [
            {"success_status": "success", "reward_info": {}, "trace": {}, "metadata": {"achievements": ["collect_wood"]}},
            {"success_status": "success", "reward_info": {}, "trace": {}, "metadata": {"achievements": ["collect_sapling"]}},
        ],
        achievement_names=["collect_wood", "collect_sapling", "wake_up"],
        denominator=4,
    )
    assert summary["collect_wood"] == {"count": 1, "frequency": 0.25}
    assert summary["collect_sapling"] == {"count": 1, "frequency": 0.25}
    assert summary["wake_up"] == {"count": 0, "frequency": 0.0}
