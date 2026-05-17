from __future__ import annotations

import json
from pathlib import Path

from nanohorizon.shared import eval_model as eval_module


def test_evaluate_model_uses_short_action_batches(monkeypatch, tmp_path):
    captured: dict[str, object] = {}

    async def fake_collect_rollouts_concurrently(**kwargs):  # type: ignore[no-untyped-def]
        captured.update(kwargs)
        seeds = list(kwargs["seeds"])
        return [
            {
                "rollout_id": f"rollout_{seed}",
                "trace_correlation_id": f"trace_{seed}",
                "success_status": "success",
                "reward_info": {
                    "outcome_reward": 2.0,
                    "outcome_objectives": {
                        "unique_achievements": 2.0,
                        "reward": 2.0,
                    },
                    "details": {
                        "achievements": ["collect_wood"],
                        "llm_call_count": 1,
                    },
                },
                "metadata": {
                    "seed": seed,
                    "llm_call_count": 1,
                    "achievements": ["collect_wood"],
                },
                "trace": {"inference": {"turns": []}},
                "artifact": [{"turns": []}],
            }
            for seed in seeds
        ]

    monkeypatch.setattr(eval_module, "collect_rollouts_concurrently", fake_collect_rollouts_concurrently)

    summary = eval_module.evaluate_model(
        base_model="Qwen/Qwen3.5-4B",
        output_dir=tmp_path / "out",
        container_url="direct://local",
        seed_start=17,
        num_rollouts=3,
        max_steps=2,
        max_concurrent_rollouts=2,
        max_length=4096,
        max_new_tokens=32,
        inference_url="https://example.invalid/v1/chat/completions",
        inference_api_key="dummy",
        request_model="demo",
    )

    assert captured["target_action_batch_size"] == 4
    assert captured["min_action_batch_size"] == 3
    assert "3-4 valid full-Craftax actions" in str(captured["system_prompt"])
    assert summary["requested_num_eval_rollouts"] == 3
    assert summary["num_eval_rollouts"] == 3
    assert summary["mean_outcome_reward"] == 2.0
    assert json.loads((Path(tmp_path / "out" / "eval_summary.json")).read_text(encoding="utf-8"))["mean_outcome_reward"] == 2.0
