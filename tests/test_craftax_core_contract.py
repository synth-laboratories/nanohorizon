from __future__ import annotations

from fastapi.testclient import TestClient

from nanohorizon.craftax_core.http_shim import create_app
from nanohorizon.craftax_core.metadata import DEFAULT_ACTION_NAMES, PRIMARY_TOOL_NAME
from nanohorizon.craftax_core.rollout import (
    _chat_completion,
    _extract_reasoning_text,
    _observation_prompt,
    _roadmap_next_targets,
    run_rollout,
)
from nanohorizon.shared.eval_model import _default_system_prompt, evaluate_model
from nanohorizon.shared.openai_compat import extract_craftax_actions


def test_action_catalog_is_full_craftax():
    assert "make_diamond_pickaxe" in DEFAULT_ACTION_NAMES
    assert "cast_fireball" in DEFAULT_ACTION_NAMES
    assert "drink_potion_red" in DEFAULT_ACTION_NAMES


def test_tool_parsing_accepts_craftax_tool_name():
    payload = {
        "choices": [
            {
                "message": {
                    "tool_calls": [
                        {
                            "function": {
                                "name": PRIMARY_TOOL_NAME,
                                "arguments": {"actions_list": ["make_diamond_pickaxe", "cast_fireball"]},
                            }
                        }
                    ]
                }
            }
        ]
    }
    assert extract_craftax_actions(payload, tool_name=PRIMARY_TOOL_NAME) == [
        "make_diamond_pickaxe",
        "cast_fireball",
    ]


def test_reasoning_extraction_accepts_qwen_reasoning_field():
    assert _extract_reasoning_text({"content": None, "reasoning": "think first"}) == "think first"


def test_default_system_prompt_mentions_achievement_roadmap():
    prompt = _default_system_prompt(thinking_budget_tokens=256)
    assert "Achievement roadmap:" in prompt
    assert "Zero-reward-for-repeat rule" in prompt
    assert "collect_wood -> collect_sapling -> place_table" in prompt


def test_roadmap_next_targets_progresses_with_unlocked_achievements():
    assert _roadmap_next_targets(set()) == ["collect_wood", "collect_sapling", "place_table"]
    assert _roadmap_next_targets({"collect_wood", "collect_sapling", "place_table"})[0] == "make_wood_pickaxe"


def test_observation_prompt_includes_progress_fields():
    prompt = _observation_prompt(
        observation_text="obs",
        target_action_batch_size=4,
        achievements_unlocked=["collect_wood"],
        next_targets=["collect_sapling", "place_table"],
    )
    assert "Achievements unlocked: collect_wood" in prompt
    assert "Next targets: collect_sapling, place_table" in prompt
    assert "obs" in prompt


def test_remote_rollout_requests_strip_vllm_only_request_overrides(monkeypatch):
    captured: dict[str, object] = {}

    class FakeResponse:
        status_code = 200

        @staticmethod
        def json() -> dict[str, object]:
            return {"choices": [{"message": {"tool_calls": []}}]}

    class FakeClient:
        def __init__(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            del args, kwargs

        def __enter__(self):  # type: ignore[no-untyped-def]
            return self

        def __exit__(self, exc_type, exc, tb):  # type: ignore[no-untyped-def]
            del exc_type, exc, tb
            return False

        def post(self, url, headers=None, json=None):  # type: ignore[no-untyped-def]
            captured["url"] = url
            captured["headers"] = headers or {}
            captured["json"] = json or {}
            return FakeResponse()

    import nanohorizon.craftax_core.rollout as rollout

    monkeypatch.setattr(rollout.httpx, "Client", FakeClient)

    _chat_completion(
        inference_url="https://example.modal.run/v1/chat/completions",
        model="demo",
        api_key="",
        messages=[{"role": "user", "content": "hello"}],
        temperature=0.0,
        max_tokens=32,
        enable_thinking=True,
        thinking_budget_tokens=256,
        timeout_s=5,
        request_logprobs=False,
    )

    request_payload = captured["json"]
    assert isinstance(request_payload, dict)
    assert "chat_template_kwargs" not in request_payload
    assert "vllm_xargs" not in request_payload
    assert request_payload["tool_choice"] == "auto"



def test_http_shim_health_and_task_info(monkeypatch):
    import nanohorizon.craftax_core.http_shim as shim

    monkeypatch.setattr(
        shim,
        "run_rollout_request",
        lambda request: {
            "rollout_id": "rollout_test",
            "trace_correlation_id": request["trace_correlation_id"],
            "success_status": "success",
            "reward_info": {"outcome_reward": 1.0, "details": {"achievements": ["collect_wood"]}},
            "trace": {"inference": {"turns": []}},
            "metadata": {"llm_call_count": 1, "achievements": ["collect_wood"]},
            "artifact": [{"turns": []}],
        },
    )
    app = create_app()
    client = TestClient(app)

    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["upstream_ready"] is True

    task_info = client.get("/task_info")
    assert task_info.status_code == 200
    assert task_info.json()["env_kind"] == "full"

    rollout = client.post(
        "/rollout",
        json={
            "trace_correlation_id": "trace_1",
            "env": {"seed": 1, "config": {"max_steps": 1}},
            "policy": {"config": {"inference_url": "http://example.test", "model": "demo", "api_key": ""}},
        },
    )
    assert rollout.status_code == 200
    assert rollout.json()["success_status"] == "success"


def test_rollout_repair_prompt_avoids_replaying_assistant_tool_calls(monkeypatch):
    import nanohorizon.craftax_core.rollout as rollout

    class FakeRender:
        text = "obs"
        pixels = None

    class FakeStep:
        def __init__(self, *, done: bool, reward: float = 0.0):
            self.done = done
            self.reward = reward
            self.render = FakeRender()

    class FakeRunner:
        def __init__(self):
            self.state = object()
            self.action_history: list[int] = []

        def reset(self):
            return FakeStep(done=False)

        def step_many(self, actions):
            self.action_history.extend(actions)
            return [FakeStep(done=True)]

    call_messages: list[list[dict[str, object]]] = []
    payloads = iter(
        [
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
                                        "arguments": {"actions_list": ["move_right"]},
                                    }
                                }
                            ],
                        }
                    }
                ]
            },
        ]
    )

    def fake_chat_completion(**kwargs):  # type: ignore[no-untyped-def]
        call_messages.append(list(kwargs["messages"]))
        return next(payloads)

    monkeypatch.setattr(rollout, "make_runner", lambda **kwargs: FakeRunner())
    monkeypatch.setattr(rollout, "achievement_names_from_state", lambda state: [])
    monkeypatch.setattr(rollout, "_chat_completion", fake_chat_completion)

    result = run_rollout(
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
    assert len(call_messages) == 2
    repair_messages = call_messages[1]
    assert not any(message.get("role") == "assistant" for message in repair_messages)


def test_eval_model_default_prompt_mentions_achievement_roadmap(monkeypatch, tmp_path):
    captured: dict[str, object] = {}

    async def fake_collect_rollouts_concurrently(**kwargs):  # type: ignore[no-untyped-def]
        captured["system_prompt"] = kwargs["system_prompt"]
        return [
            {
                "success_status": "success",
                "reward_info": {
                    "outcome_reward": 1.0,
                    "outcome_objectives": {"unique_achievements": 1.0},
                    "details": {"achievements": ["collect_wood"]},
                },
                "trace": {"inference": {"turns": []}},
                "metadata": {"achievements": ["collect_wood"]},
                "artifact": [{"turns": []}],
            }
        ]

    monkeypatch.setattr("nanohorizon.shared.eval_model.collect_rollouts_concurrently", fake_collect_rollouts_concurrently)

    result = evaluate_model(
        base_model="demo",
        output_dir=tmp_path,
        container_url="direct://local",
        seed_start=10000,
        num_rollouts=1,
        max_steps=1,
        max_concurrent_rollouts=1,
        max_length=16,
        max_new_tokens=1,
        inference_url="http://example.test/v1/chat/completions",
        request_model="demo",
        request_timeout_seconds=1.0,
    )

    system_prompt = str(captured["system_prompt"])
    assert "Achievement roadmap" in system_prompt
    assert "collect_wood -> collect_sapling -> place_table -> make_wood_pickaxe" in system_prompt
    assert "Zero-reward-for-repeat rule" in system_prompt
    assert result["mean_outcome_reward"] == 1.0
