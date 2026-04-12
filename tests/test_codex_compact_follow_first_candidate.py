from __future__ import annotations

import json
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs" / "craftax_prompt_opt_qwen35_4b_codex_compact_follow_first.yaml"
SOURCE_PATH = REPO_ROOT / "src" / "nanohorizon" / "baselines" / "prompt_opt.py"


def test_candidate_config_uses_compact_follow_first_prompt() -> None:
    payload = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    seed_prompt = str(payload["prompt"]["seed_prompt"])

    assert "exactly three items" in seed_prompt
    assert "nearest useful tile, object, or resource target" in seed_prompt
    assert "loop-break fallback action" in seed_prompt
    assert "replace the stale target item" in seed_prompt
    assert "Choose a short 3 or 4 action batch that follows the first todo item" in seed_prompt
    assert "Do not reveal the todo list" in seed_prompt


def test_prompt_opt_source_keeps_compact_action_directive() -> None:
    source = SOURCE_PATH.read_text(encoding="utf-8")

    assert "Keep the prompt compact and action-directed" in source
    assert "short valid full-Craftax action batch" in source
    assert "tree or other gatherable resource" in source
    assert "todo_scratchpad_directive()" in source


def test_candidate_record_bundle_is_not_run_yet() -> None:
    record_dir = (
        REPO_ROOT
        / "records"
        / "prompt_opt_1usd_gpt54_family"
        / "2026-04-12_codex_compact_follow_first"
    )
    metadata = json.loads((record_dir / "metadata.json").read_text(encoding="utf-8"))
    metrics = json.loads((record_dir / "metrics.json").read_text(encoding="utf-8"))
    command = (record_dir / "command.txt").read_text(encoding="utf-8").strip()

    assert metadata["implementation_status"] == "candidate_not_run"
    assert metrics["status"] == "not_run"
    assert "run_craftax_prompt_opt_qwen35_4b_gpt54_budget.sh" in command
