from __future__ import annotations

def test_eval_model_uses_native_env_reward_total_when_achievement_fields_are_missing(
    monkeypatch,
    tmp_path,
):
    import nanohorizon.shared.eval_model as eval_model

    rollouts = [
        {
            "rollout_id": f"rollout_{idx}",
            "success_status": "success",
            "reward_info": {
                "outcome_objectives": {},
                "details": {"native_env_reward_total": float(idx + 1)},
            },
            "trace": {"inference": {"turns": []}},
            "metadata": {"llm_call_count": idx + 2},
        }
        for idx in range(3)
    ]

    monkeypatch.setattr(eval_model, "release_cuda_memory", lambda: None)

    async def fake_collect_rollouts_concurrently(**_):  # type: ignore[no-untyped-def]
        return rollouts

    monkeypatch.setattr(eval_model, "collect_rollouts_concurrently", fake_collect_rollouts_concurrently)

    summary = eval_model.evaluate_model(
        base_model="Qwen/Qwen3.5-4B",
        output_dir=tmp_path / "eval",
        container_url="direct://local",
        seed_start=7,
        num_rollouts=3,
        max_steps=1,
        max_concurrent_rollouts=1,
        max_length=256,
        max_new_tokens=16,
        inference_url="http://example.invalid/v1/chat/completions",
        request_model="demo",
    )

    assert summary["num_eval_rollouts"] == 3
    assert summary["mean_outcome_reward"] == 2.0
    assert summary["mean_outcome_reward_over_requested_rollouts"] == 2.0
    assert summary["details"][0]["outcome_reward"] == 1.0
    assert summary["details"][2]["outcome_reward"] == 3.0


def test_mirrored_rollout_helpers_share_the_same_fallback() -> None:
    rollout = {
        "reward_info": {
            "outcome_objectives": {},
            "details": {"native_env_reward_total": 4.5},
        }
    }

    from nanohorizon.baselines.offline_sft import rollout_outcome_reward as offline_reward
    from nanohorizon.baselines.prompt_opt import rollout_outcome_reward as prompt_reward
    from nanohorizon.baselines.rlvr import rollout_outcome_reward as rlvr_reward
    from nanohorizon.shared.craftax_data import rollout_outcome_reward as shared_reward

    assert shared_reward(rollout) == 4.5
    assert offline_reward(rollout) == 4.5
    assert prompt_reward(rollout) == 4.5
    assert rlvr_reward(rollout) == 4.5
