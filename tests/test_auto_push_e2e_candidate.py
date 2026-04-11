from __future__ import annotations

import json
from pathlib import Path

import yaml

from nanohorizon.baselines.prompt_opt import (
    build_reflection_system_directive,
    build_seed_prompt,
    todo_scratchpad_directive,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs" / "craftax_prompt_opt_qwen35_4b_codex_auto_push_e2e.yaml"
SOURCE_PATH = REPO_ROOT / "src" / "nanohorizon" / "baselines" / "prompt_opt.py"
SCRIPT_PATH = REPO_ROOT / "scripts" / "run_craftax_prompt_opt_qwen35_4b_gpt54_budget.sh"
DOC_PATH = REPO_ROOT / "docs" / "task-craftax.md"
RECORD_DIR = (
    REPO_ROOT
    / "records"
    / "prompt_opt_1usd_gpt54_family"
    / "2026-04-11_auto_push_e2e"
)


def test_task_doc_names_the_candidate() -> None:
    text = DOC_PATH.read_text(encoding="utf-8")
    assert "Auto Push E2E" in text
    assert "Todo Tool" in text


def test_candidate_config_uses_auto_push_todo_prompt() -> None:
    payload = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    seed_prompt = str(payload["prompt"]["seed_prompt"])

    assert seed_prompt == build_seed_prompt()
    assert "Auto Push E2E" not in seed_prompt
    assert todo_scratchpad_directive() in seed_prompt
    assert "end-to-end handoff guard" in seed_prompt
    assert "craftax_interact" in seed_prompt


def test_prompt_opt_source_centralizes_todo_contract() -> None:
    source = SOURCE_PATH.read_text(encoding="utf-8")

    assert "TODO_SCRATCHPAD_REQUIREMENTS" in source
    assert "_build_seed_prompt_body()" in source
    assert "todo_scratchpad_directive()" in source
    assert "Do not reveal the todo list or scratchpad" in source
    assert todo_scratchpad_directive() in build_reflection_system_directive()


def test_candidate_record_bundle_is_present_and_marked_not_run() -> None:
    metadata = json.loads((RECORD_DIR / "metadata.json").read_text(encoding="utf-8"))
    metrics = json.loads((RECORD_DIR / "metrics.json").read_text(encoding="utf-8"))
    command = (RECORD_DIR / "command.txt").read_text(encoding="utf-8").strip()

    assert metadata["implementation_status"] == "candidate_not_run"
    assert metadata["name"] == "auto_push_e2e"
    assert metrics["status"] == "not_run"
    assert "craftax_prompt_opt_qwen35_4b_codex_auto_push_e2e.yaml" in command
    assert SCRIPT_PATH.exists()
