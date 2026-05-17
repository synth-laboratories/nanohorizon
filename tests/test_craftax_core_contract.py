from __future__ import annotations

from fastapi.testclient import TestClient

from nanohorizon.craftax_core.http_shim import create_app
from nanohorizon.craftax_core.metadata import DEFAULT_ACTION_NAMES, PRIMARY_TOOL_NAME
from nanohorizon.craftax_core.rollout import _chat_completion, _extract_reasoning_text, run_rollout
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
