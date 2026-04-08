from __future__ import annotations

import json
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs" / "craftax_prompt_opt_qwen35_4b_codex_dispatch_fix_e2e.yaml"
SOURCE_PATH = REPO_ROOT / "src" / "nanohorizon" / "baselines" / "prompt_opt.py"
RECORD_DIR = (
    REPO_ROOT
    / "records"
    / "prompt_opt_1usd_gpt54_family"
    / "2026-04-08_codex_dispatch_fix_e2e"
)


def test_candidate_config_adds_end_to_end_dispatch_rule() -> None:
    payload = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    seed_prompt = str(payload["prompt"]["seed_prompt"])

    assert "tiny private" in seed_prompt
    assert "todo list with exactly three items" in seed_prompt
    assert "Dispatch the 3 or 4 action batch" in seed_prompt
    assert "until it is satisfied, blocked, or" in seed_prompt
    assert "remaining actions on item two or the fallback" in seed_prompt
    assert "Do not reveal the todo list" in seed_prompt


def test_prompt_opt_source_centralizes_dispatch_rule() -> None:
    source = SOURCE_PATH.read_text(encoding="utf-8")

    assert "TODO_SCRATCHPAD_REQUIREMENTS" in source
    assert "Dispatch the 3-4 action batch end-to-end from the current first todo item" in source
    assert "preserve this todo-tool contract" in source


def test_candidate_record_bundle_is_present_and_marked_not_run() -> None:
    metadata = json.loads((RECORD_DIR / "metadata.json").read_text(encoding="utf-8"))
    metrics = json.loads((RECORD_DIR / "metrics.json").read_text(encoding="utf-8"))
    command = (RECORD_DIR / "command.txt").read_text(encoding="utf-8").strip()

    assert metadata["implementation_status"] == "candidate_not_run"
    assert metrics["status"] == "not_run"
    assert "run_craftax_prompt_opt_qwen35_4b_gpt54_budget.sh" in command
