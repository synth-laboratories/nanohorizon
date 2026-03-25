from __future__ import annotations

from fastapi.testclient import TestClient

from nanohorizon.craftax_core.http_shim import create_app
from nanohorizon.craftax_core.metadata import DEFAULT_ACTION_NAMES, PRIMARY_TOOL_NAME
from nanohorizon.craftax_core.rollout import _extract_reasoning_text
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
