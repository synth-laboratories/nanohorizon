from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from nanohorizon.craftax_core.http_shim import build_task_info  # noqa: E402
from nanohorizon.craftax_core.metadata import (  # noqa: E402
    DEFAULT_CANDIDATE_LABEL,
    build_server_push_e2e_metadata,
    refresh_todo_items as refresh_runtime_todo_items,
)
from nanohorizon.craftax_core.runner import build_runner_output  # noqa: E402
from nanohorizon.baselines.prompt_opt import (  # noqa: E402
    TODO_SCRATCHPAD_REQUIREMENTS,
    candidate_record,
    refresh_todo_items,
    validate_candidate_record,
)


def test_metadata_renders_a_compact_three_item_scratchpad() -> None:
    metadata = build_server_push_e2e_metadata()
    task_info = metadata.to_task_info()

    assert metadata.candidate_label == DEFAULT_CANDIDATE_LABEL
    assert task_info["todo_item_count"] == 3
    assert task_info["scratchpad_mode"] == "compact-three-item"
    assert len(task_info["todo_items"]) == 3
    assert "Todo Scratchpad" in task_info["todo_scratchpad"]


def test_http_shim_and_runner_share_the_same_payload_shape() -> None:
    metadata = build_server_push_e2e_metadata()
    task_payload = build_task_info(metadata)
    runner_payload = build_runner_output()

    assert task_payload["health"] == "ok"
    assert task_payload["task_info"]["candidate_label"] == DEFAULT_CANDIDATE_LABEL
    assert runner_payload["payload"] == task_payload
    json.dumps(runner_payload)


def test_experiment_verifier_records_the_same_candidate() -> None:
    verifier_path = ROOT / "experiments/server_push_e2e/results/verifier.json"
    verifier = json.loads(verifier_path.read_text(encoding="utf-8"))

    assert verifier["candidate_label"] == DEFAULT_CANDIDATE_LABEL
    assert verifier["status"] == "pass"
    assert verifier["checks"]["todo_scratchpad_rendered"] is True


def test_prompt_helper_and_refresh_contract_stay_in_sync() -> None:
    record = candidate_record()
    refreshed = refresh_todo_items(
        ("Confirm the task constraints and keep the change narrow.", "Stale item", "Surface the scratchpad through task info and the runner."),
        completed_items=("Stale item",),
        next_action="Validate the output with a local smoke verifier.",
    )

    assert record["scratchpad_requirements"] == list(TODO_SCRATCHPAD_REQUIREMENTS)
    assert record["scratchpad_refresh_example"] == (
        "Confirm the task constraints and keep the change narrow.",
        "Surface the scratchpad through task info and the runner.",
        "Validate the output with a local smoke verifier.",
    )
    assert refreshed == refresh_runtime_todo_items(
        (
            "Confirm the task constraints and keep the change narrow.",
            "Stale item",
            "Surface the scratchpad through task info and the runner.",
        ),
        completed_items=("Stale item",),
        next_action="Validate the output with a local smoke verifier.",
    )
    assert len(refreshed) == 3
    assert validate_candidate_record(record) == []
