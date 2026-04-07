#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

try:
    from actorbench_parent_lane import run_lane_cli
except ModuleNotFoundError:  # pragma: no cover
    shared_root = Path(__file__).resolve().parents[3] / "shared"
    if str(shared_root) not in sys.path:
        sys.path.insert(0, str(shared_root))
    from actorbench_parent_lane import run_lane_cli


CONFIG = {
    "task_id": "actorbench_nh__craftax_hello_world__worker_v1",
    "parent_task_id": "nanohorizon_craftax_hello_world",
    "target_actor_type": "worker",
    "focus": "Minimal Craftax rollout and artifact discipline.",
    "summary": "ActorBench score for the NanoHorizon Craftax hello-world worker path.",
    "wrapper_path": __file__,
    "bundled_parent_runner": "parent_runner.py",
    "host_parent_runner": "run_nanohorizon_craftax_hello_world_task.py",
    "parent_output_path": "artifacts/reportbench_output.json",
    "primary_metric_name": "mean_outcome_reward",
    "primary_metric_artifact": "artifacts/eval_summary.json",
    "primary_metric_key": "mean_outcome_reward",
    "report_path": "reports/reproduction.md",
    "criteria": [
        {
            "id": "artifact_bundle",
            "kind": "all_files_exist",
            "weight": 0.25,
            "paths": [
                "artifacts/eval_summary.json",
                "artifacts/rollouts.jsonl",
                "artifacts/result_manifest.json",
                "artifacts/reportbench_output.json",
                "reports/reproduction.md",
            ],
        },
        {
            "id": "runner_success",
            "kind": "json_equals",
            "weight": 0.20,
            "path": "artifacts/result_manifest.json",
            "key": "status",
            "value": "succeeded",
        },
        {
            "id": "contract_values",
            "kind": "json_equals_map",
            "weight": 0.25,
            "path": "artifacts/eval_summary.json",
            "items": {
                "requested_rollouts": 10,
                "requested_total_llm_calls": 10,
                "requested_max_steps_per_rollout": 1,
                "requested_rollout_concurrency": 10,
            },
        },
        {
            "id": "reward_present",
            "kind": "json_present",
            "weight": 0.15,
            "path": "artifacts/eval_summary.json",
            "key": "mean_outcome_reward",
        },
        {
            "id": "report_grounding",
            "kind": "text_contains_all",
            "weight": 0.15,
            "path": "reports/reproduction.md",
            "substrings": ["Mean reward", "Requested trajectories"],
        },
    ],
}


if __name__ == "__main__":
    raise SystemExit(run_lane_cli(CONFIG))
