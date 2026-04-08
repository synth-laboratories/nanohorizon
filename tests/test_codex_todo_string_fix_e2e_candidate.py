from __future__ import annotations

import json
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs" / "craftax_prompt_opt_qwen35_4b_codex_todo_string_fix_e2e.yaml"
SOURCE_PATH = REPO_ROOT / "src" / "nanohorizon" / "baselines" / "prompt_opt.py"
RECORD_DIR = (
    REPO_ROOT
    / "records"
    / "prompt_opt_1usd_gpt54_family"
    / "2026-04-08_codex_todo_string_fix_e2e"
)


def test_candidate_config_uses_short_string_and_e2e_todo_prompt() -> None:
    payload = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    seed_prompt = str(payload["prompt"]["seed_prompt"])

    assert "todo list with exactly three items" in seed_prompt
    assert "short string fragment" in seed_prompt
    assert "Refresh completed todo items every turn" in seed_prompt
    assert "replace the stale target item" in seed_prompt
    assert "one short end-to-end 3 or 4 action batch" in seed_prompt
    assert "without mixing unrelated goals" in seed_prompt


def test_prompt_opt_source_centralizes_short_string_and_e2e_todo_contract() -> None:
    source = SOURCE_PATH.read_text(encoding="utf-8")

    assert "TODO_SCRATCHPAD_REQUIREMENTS" in source
    assert "short string fragment" in source
    assert "end-to-end action batch" in source
    assert "enforce_todo_scratchpad_contract" in source


def test_candidate_record_bundle_is_present_and_cites_baseline_comparison() -> None:
    metadata = json.loads((RECORD_DIR / "metadata.json").read_text(encoding="utf-8"))
    metrics = json.loads((RECORD_DIR / "metrics.json").read_text(encoding="utf-8"))
    notes = (RECORD_DIR / "notes.md").read_text(encoding="utf-8")

    assert metadata["implementation_status"] == "candidate_not_run"
    assert metrics["status"] == "not_run"
    assert metrics["baseline_primary_score"] == 0.35
    assert metrics["bootstrap_primary_score"] == 0.6
    assert "0.6" in notes
    assert "0.35" in notes
