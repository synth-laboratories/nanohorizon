from __future__ import annotations

from types import SimpleNamespace

from nanohorizon.craftax_core.modalities import RenderBundle, RenderMode
from nanohorizon.craftax_core import rollout as rollout_module


def test_run_rollout_includes_structured_state_view_in_user_prompt(monkeypatch) -> None:
    captured_messages: list[list[dict[str, object]]] = []

    class FakeRunner:
        def __init__(self) -> None:
            self.state = {"achievements": []}
            self.action_history: list[int] = []

        def reset(self):
            return SimpleNamespace(
                done=False,
                reward=0.0,
                info={},
                step_index=0,
                episode_index=0,
                render=RenderBundle(
                    mode=RenderMode.TEXT,
                    text="text observation only",
                    pixels=None,
                    state_view={"health": 10, "nearby_entities": ["tree"], "inventory": {"wood": 0}},
                ),
            )

        def step_many(self, actions):
            self.action_history.extend(actions)
            self.state = {"achievements": ["collect_wood"]}
            return [
                SimpleNamespace(
                    done=True,
                    reward=1.0,
                    render=RenderBundle(mode=RenderMode.TEXT, text="done", pixels=None),
                )
            ]

    def fake_chat_completion(**kwargs):  # type: ignore[no-untyped-def]
        captured_messages.append(list(kwargs["messages"]))
        return {
            "choices": [
                {
                    "message": {
                        "content": "",
                        "tool_calls": [
                            {
                                "function": {
                                    "name": rollout_module.PRIMARY_TOOL_NAME,
                                    "arguments": {"actions_list": ["do"]},
                                }
                            }
                        ],
                    }
                }
            ]
        }

    monkeypatch.setattr(rollout_module, "make_runner", lambda **_: FakeRunner())
    monkeypatch.setattr(rollout_module, "_chat_completion", fake_chat_completion)
    monkeypatch.setattr(rollout_module, "achievement_names_from_state", lambda state: list(state.get("achievements", [])))

    result = rollout_module.run_rollout(
        inference_url="http://example.test/v1/chat/completions",
        model="demo",
        api_key="",
        seed=0,
        max_steps=1,
        trace_correlation_id="trace",
        system_prompt="system",
        target_action_batch_size=1,
        min_action_batch_size=1,
        request_logprobs=False,
    )

    assert result["success_status"] == "success"
    assert captured_messages
    user_message = captured_messages[0][1]["content"]
    assert "Structured state view" in user_message
    assert '"nearby_entities":["tree"]' in user_message
    assert "Use the structured state view first" in user_message
