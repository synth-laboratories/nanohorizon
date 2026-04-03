from __future__ import annotations

import asyncio

import nanohorizon.shared.craftax_data as craftax_data
from nanohorizon.baselines.cpt_data import (
    BudgetState,
    append_rows_until_token_budget,
    build_seed_schedule,
    project_rollout_to_text_row,
    restore_rows_with_budget,
)


class FakeTokenizer:
    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [ord(char) for char in text]

    def decode(self, ids: list[int], *, skip_special_tokens: bool = False) -> str:
        del skip_special_tokens
        return "".join(chr(token_id) for token_id in ids)


def test_build_seed_schedule_repeats_with_offsets() -> None:
    schedule = build_seed_schedule(base_seeds=[7, 11], requested_rollouts=5, seed_offset_stride=100)
    assert schedule == [7, 11, 107, 111, 207]


def test_project_rollout_to_text_row_contains_actions_and_reward() -> None:
    rollout = {
        "trace_correlation_id": "trace_001",
        "rollout_id": "rollout_001",
        "success_status": "success",
        "reward_info": {
            "outcome_reward": 2.0,
            "details": {"achievements": ["collect_wood", "place_table"], "llm_call_count": 1},
        },
        "metadata": {"seed": 17, "action_history": ["move_right", "do"]},
        "artifact": [
            {
                "turns": [
                    {
                        "turn_index": 0,
                        "prompt_messages": [
                            {"role": "system", "content": "teacher"},
                            {"role": "user", "content": "tree nearby"},
                        ],
                        "reasoning_text": "move to the tree and chop it",
                        "assistant_text": "",
                        "actions": ["move_right", "do"],
                        "decision_reward": 1.0,
                        "return_to_go": 1.0,
                    }
                ]
            }
        ],
    }

    row = project_rollout_to_text_row(rollout, include_reasoning=True, include_action_history=True)

    assert row["metadata"]["outcome_reward"] == 2.0
    assert row["metadata"]["seed"] == 17
    assert "Chosen actions: move_right, do" in row["text"]
    assert "Outcome reward: 2.00" in row["text"]
    assert "Final action history:" in row["text"]


def test_append_rows_until_token_budget_truncates_last_row() -> None:
    rows = [
        {"text": "abcd", "metadata": {}},
        {"text": "efgh", "metadata": {}},
        {"text": "ijkl", "metadata": {}},
    ]
    state = BudgetState(token_budget=10)

    written = append_rows_until_token_budget(rows=rows, tokenizer=FakeTokenizer(), budget_state=state)

    assert [row["text"] for row in written] == ["abcd", "efgh", "ij"]
    assert state.total_tokens == 10
    assert state.rows_written == 3
    assert state.truncated_rows == 1


def test_restore_rows_with_budget_replays_existing_rows() -> None:
    rows = [
        {"text": "abcd", "metadata": {"trace_correlation_id": "a"}},
        {"text": "efgh", "metadata": {"trace_correlation_id": "b"}},
        {"text": "ijkl", "metadata": {"trace_correlation_id": "c"}},
    ]

    restored, state = restore_rows_with_budget(rows=rows, tokenizer=FakeTokenizer(), token_budget=10)

    assert [row["text"] for row in restored] == ["abcd", "efgh", "ij"]
    assert state.total_tokens == 10
    assert state.rows_written == 3
    assert state.truncated_rows == 1


def test_collect_rollouts_direct_emits_progress(monkeypatch) -> None:
    def fake_run_rollout_request(request: dict[str, object]) -> dict[str, object]:
        seed = int(request["env"]["seed"])  # type: ignore[index]
        trace_correlation_id = str(request["trace_correlation_id"])
        return {
            "rollout_id": f"rollout_{seed}",
            "trace_correlation_id": trace_correlation_id,
            "trial_id": trace_correlation_id,
            "success_status": "success",
            "reward_info": {
                "outcome_reward": float(seed),
                "outcome_objectives": {"unique_achievements": float(seed), "reward": float(seed)},
                "details": {"achievements": ["collect_wood"], "llm_call_count": 1},
            },
            "metadata": {"seed": seed, "llm_call_count": 1, "achievements": ["collect_wood"]},
            "trace": {"inference": {"turns": []}},
            "artifact": [{"turns": []}],
        }

    monkeypatch.setattr(craftax_data, "run_rollout_request", fake_run_rollout_request)
    progress_events: list[dict[str, object]] = []

    rollouts, summary = asyncio.run(
        craftax_data.collect_rollouts_concurrently_with_summary(
            container_url="direct://local",
            inference_url="https://example.invalid/v1/chat/completions",
            model="Qwen/Qwen3.5-27B",
            api_key="dummy",
            seeds=[3, 7],
            max_steps=4,
            system_prompt="teacher",
            temperature=0.0,
            max_tokens=32,
            enable_thinking=False,
            thinking_budget_tokens=0,
            policy_version="test",
            target_action_batch_size=4,
            min_action_batch_size=3,
            request_timeout_seconds=30.0,
            max_concurrent_rollouts=2,
            trace_prefix="test",
            rollout_concurrency=2,
            rollout_semaphore_limit=2,
            request_logprobs=False,
            progress_callback=progress_events.append,
        )
    )

    assert len(rollouts) == 2
    assert summary["completed_rollouts"] == 2
    assert len(progress_events) == 2
    assert progress_events[-1]["completed_rollouts"] == 2
    assert progress_events[-1]["num_structured_rollouts"] == 2
