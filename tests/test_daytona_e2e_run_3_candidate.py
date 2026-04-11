from __future__ import annotations

import json
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs" / "craftax_prompt_opt_qwen35_4b_codex_daytona_e2e_run_3.yaml"
SOURCE_PATH = REPO_ROOT / "src" / "nanohorizon" / "baselines" / "prompt_opt.py"
RECORD_DIR = (
    REPO_ROOT
    / "records"
    / "prompt_opt_1usd_gpt54_family"
    / "2026-04-11_daytona_e2e_run_3"
)


def test_candidate_config_uses_daytona_todo_prompt() -> None:
    payload = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    seed_prompt = str(payload["prompt"]["seed_prompt"])

    assert "Daytona E2E Run 3" not in seed_prompt
    assert "tiny private" in seed_prompt
    assert "todo list with exactly three items" in seed_prompt
    assert "fallback action that breaks a loop" in seed_prompt
    assert "replace the stale target item" in seed_prompt
    assert "end-to-end handoff guard" in seed_prompt


def test_prompt_opt_source_still_centralizes_todo_contract() -> None:
    source = SOURCE_PATH.read_text(encoding="utf-8")

    assert "TODO_SCRATCHPAD_REQUIREMENTS" in source
    assert "todo_scratchpad_directive()" in source
    assert "replace the stale target item" in source
    assert "Do not reveal the todo list or scratchpad" in source


def test_candidate_record_bundle_is_present_and_marked_not_run() -> None:
    metadata = json.loads((RECORD_DIR / "metadata.json").read_text(encoding="utf-8"))
    metrics = json.loads((RECORD_DIR / "metrics.json").read_text(encoding="utf-8"))
    command = (RECORD_DIR / "command.txt").read_text(encoding="utf-8").strip()

    assert metadata["implementation_status"] == "candidate_not_run"
    assert metadata["name"] == "daytona_e2e_run_3"
    assert metrics["status"] == "not_run"
    assert "craftax_prompt_opt_qwen35_4b_codex_daytona_e2e_run_3.yaml" in command
