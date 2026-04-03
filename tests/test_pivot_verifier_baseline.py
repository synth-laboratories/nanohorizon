from __future__ import annotations

import json
from pathlib import Path

from nanohorizon.baselines import pivot_verifier


def _build_rollout() -> dict[str, object]:
    return {
        "success_status": "success",
        "reward_info": {"outcome_reward": 1.0},
        "trace": {
            "inference": {
                "turns": [
                    {
                        "turn_index": 0,
                        "prompt_messages": [
                            {"role": "system", "content": "system"},
                            {
                                "role": "user",
                                "content": (
                                    "Craftax state summary\n"
                                    "inventory: wood=0, stone=0\n"
                                    "achievements: none\n"
                                    "there is a tree nearby"
                                ),
                            },
                        ],
                        "assistant_text": "<tool_call>{\"name\":\"craftax_interact\",\"arguments\":{\"actions_list\":[\"do\"]}}</tool_call>",
                        "reasoning_text": "There is a tree nearby, so use do to collect wood first.",
                        "actions": ["do"],
                        "decision_reward": 1.0,
                        "return_to_go": 0.0,
                        "invalid_parse": False,
                    },
                    {
                        "turn_index": 1,
                        "prompt_messages": [
                            {"role": "system", "content": "system"},
                            {
                                "role": "user",
                                "content": (
                                    "Craftax state summary\n"
                                    "inventory: wood=1, stone=0\n"
                                    "achievements: collect_wood\n"
                                    "there is a table spot nearby"
                                ),
                            },
                        ],
                        "assistant_text": "",
                        "reasoning_text": "",
                        "actions": ["move_right"],
                        "decision_reward": 0.0,
                        "return_to_go": 1.0,
                        "invalid_parse": False,
                    },
                ]
            }
        },
        "trace_correlation_id": "trace-1",
        "_request_seed": 1,
    }


def test_build_verifier_examples_from_rollouts_emits_scored_examples():
    rows, summary = pivot_verifier.build_verifier_examples_from_rollouts(
        rollouts=[_build_rollout()],
        lookahead=1,
        max_examples=16,
    )

    assert summary["example_count"] == 1
    row = rows[0]
    assert row["actions"] == ["do"]
    assert row["metadata"]["inventory_delta"] == {"wood": 1}
    assert row["metadata"]["new_achievements"] == ["collect_wood"]
    assert row["labels"]["progress_reward"] > 0.5
    assert row["labels"]["process_reward"] > 0.4
    payload = json.loads(row["response"])
    assert payload["scores"]["total_reward"] == row["labels"]["total_reward"]
    assert payload["verdict"] in {"good", "strong"}
    assert row["tools"][0]["function"]["name"] == pivot_verifier.VERIFIER_TOOL_NAME
    assistant_message = row["messages"][-1]
    assert assistant_message["tool_calls"][0]["function"]["name"] == pivot_verifier.VERIFIER_TOOL_NAME
    assert assistant_message["tool_calls"][0]["function"]["arguments"]["scores"]["total_reward"] == row["labels"]["total_reward"]


def test_split_examples_keeps_nonempty_train_and_holdout():
    examples = [
        {"example_id": "a"},
        {"example_id": "b"},
        {"example_id": "c"},
        {"example_id": "d"},
    ]
    train_rows, holdout_rows = pivot_verifier.split_examples(
        examples,
        holdout_fraction=0.25,
        seed=7,
    )

    assert train_rows
    assert holdout_rows
    assert len(train_rows) + len(holdout_rows) == len(examples)


def test_load_golden_eval_examples_builds_rubric_scored_examples(tmp_path):
    eval_path = tmp_path / "golden_eval.jsonl"
    eval_path.write_text(
        json.dumps(
            {
                "example_id": "golden-1",
                "target_achievement": "collect_wood",
                "state_text": "Craftax state summary\ninventory: wood=0\nachievements: none\ntree adjacent",
                "thinking_text": "Harvest the adjacent tree now.",
                "actions": ["do"],
                "next_state_text": "Craftax state summary\ninventory: wood=1\nachievements: collect_wood",
                "gold_process_reward": 0.9,
                "rubric_reward": 1.0,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    rows = pivot_verifier.load_golden_eval_examples(
        config={"evaluation": {"golden_eval_path": str(eval_path)}}
    )

    assert len(rows) == 1
    row = rows[0]
    assert row["labels"]["progress_reward"] == 1.0
    assert "Progress rubric:" in row["prompt_messages"][-1]["content"]
    assert row["metadata"]["rubric_target_achievement"] == "collect_wood"


def test_export_downstream_verifier_artifacts_builds_preference_pairs(tmp_path):
    examples = [
        {
            "example_id": "good",
            "state_text": "Craftax state summary\ninventory: wood=0\nachievements: none\ntree adjacent",
            "thinking_text": "Harvest now.",
            "actions": ["do"],
            "next_state_text": "Craftax state summary\ninventory: wood=1\nachievements: collect_wood",
            "labels": {"process_reward": 0.9, "progress_reward": 1.0, "total_reward": 0.96},
            "metadata": {"rubric_target_achievement": "collect_wood"},
        },
        {
            "example_id": "bad",
            "state_text": "Craftax state summary\ninventory: wood=0\nachievements: none\ntree adjacent",
            "thinking_text": "Sleep now.",
            "actions": ["sleep"],
            "next_state_text": "Craftax state summary\ninventory: wood=0\nachievements: none\ntree adjacent",
            "labels": {"process_reward": 0.1, "progress_reward": 0.1, "total_reward": 0.12},
            "metadata": {"rubric_target_achievement": "collect_wood"},
        },
    ]

    summary = pivot_verifier.export_downstream_verifier_artifacts(
        output_root=tmp_path,
        examples=examples,
    )

    assert summary["candidate_group_count"] == 1
    assert summary["preference_pair_count"] == 1
    preference_rows = (tmp_path / "artifacts" / "pivot_verifier_preference_pairs.jsonl").read_text(encoding="utf-8")
    assert "group_00000" in preference_rows
    assert "craftax_interact" in preference_rows


def test_run_pipeline_writes_record_bundle_from_saved_rollouts(monkeypatch, tmp_path):
    rollouts_path = tmp_path / "rollouts.jsonl"
    rollouts_path.write_text(json.dumps(_build_rollout()) + "\n", encoding="utf-8")

    fake_adapter_dir = tmp_path / "artifacts" / "pivot_verifier_adapter"

    def fake_train_verifier_model(*, config, output_root, train_rows):  # type: ignore[no-untyped-def]
        fake_adapter_dir.mkdir(parents=True, exist_ok=True)
        summary = {
            "adapter_dir": str(fake_adapter_dir),
            "examples_seen": len(train_rows),
            "optimizer_steps": 2,
            "mean_loss": 0.12,
        }
        pivot_verifier.write_json(output_root / "artifacts" / "training_summary.json", summary)
        return fake_adapter_dir, summary

    def fake_evaluate_verifier_model(*, config, output_root, adapter_dir, holdout_rows):  # type: ignore[no-untyped-def]
        summary = {
            "evaluated_examples": len(holdout_rows),
            "parsed_examples": len(holdout_rows),
            "json_parse_rate": 1.0,
            "mae_total_reward": 0.05,
            "pairwise_accuracy": 1.0,
            "skipped": False,
        }
        pivot_verifier.write_json(output_root / "artifacts" / "eval_summary.json", summary)
        return summary

    monkeypatch.setattr(pivot_verifier, "train_verifier_model", fake_train_verifier_model)
    monkeypatch.setattr(pivot_verifier, "evaluate_verifier_model", fake_evaluate_verifier_model)
    monkeypatch.setattr(pivot_verifier, "train_downstream_policy_with_preferences", lambda **kwargs: (None, None))

    config = {
        "task": {"track": "pivot_verifier_qwen35_4b", "method_name": "spct_qwen35_4b_baseline"},
        "data": {"bootstrap_rollouts_path": str(rollouts_path)},
        "dataset": {"lookahead": 1, "max_examples": 16, "holdout_fraction": 0.5, "split_seed": 3},
        "verifier": {"base_model": "Qwen/Qwen3.5-4B", "max_length": 4096},
        "training": {"max_steps": 2},
        "evaluation": {"max_examples": 8},
    }

    result = pivot_verifier.run_pipeline(
        config=config,
        output_root=tmp_path,
        command="python -m nanohorizon.baselines.pivot_verifier",
        skip_train=False,
        skip_eval=False,
    )

    assert Path(result["output_root"]) == tmp_path
    assert (tmp_path / "metadata.json").exists()
    assert (tmp_path / "metrics.json").exists()
    assert (tmp_path / "artifacts" / "pivot_verifier_dataset.jsonl").exists()
    assert (tmp_path / "artifacts" / "pivot_verifier_result.json").exists()
    metrics = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["verifier_base_model"] == "Qwen/Qwen3.5-4B"
    assert metrics["dataset_example_count"] == 1
    assert metrics["eval_json_parse_rate"] == 1.0
