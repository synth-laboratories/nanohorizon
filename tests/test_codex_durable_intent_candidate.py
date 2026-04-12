from __future__ import annotations

import json
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs" / "craftax_prompt_opt_qwen35_4b_codex_durable_intent_fix.yaml"
SOURCE_PATH = REPO_ROOT / "src" / "nanohorizon" / "baselines" / "prompt_opt.py"
RECORD_DIR = (
    REPO_ROOT
    / "records"
    / "prompt_opt_1usd_gpt54_family"
    / "2026-04-07_codex_durable_intent_fix"
)


def test_candidate_config_uses_private_todo_prompt() -> None:
    payload = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    seed_prompt = str(payload["prompt"]["seed_prompt"])

    assert "tiny private" in seed_prompt
    assert "todo list with exactly three items" in seed_prompt
    assert "best immediate survival or" in seed_prompt
    assert "Refresh completed items every turn" in seed_prompt
    assert "replace the target" in seed_prompt
    assert "Do not reveal the todo list to the" in seed_prompt


def test_prompt_opt_source_preserves_todo_guidance_during_reflection() -> None:
    source = SOURCE_PATH.read_text(encoding="utf-8")

    assert "Keep the prompt compact and action-directed" in source
    assert "nearest useful tile, object, or resource target" in source
    assert "loop-break or fallback action" in source
    assert "build_reflection_system_directive" in source


def test_candidate_record_bundle_is_present_and_marked_not_run() -> None:
    metadata = json.loads((RECORD_DIR / "metadata.json").read_text(encoding="utf-8"))
    metrics = json.loads((RECORD_DIR / "metrics.json").read_text(encoding="utf-8"))
    command = (RECORD_DIR / "command.txt").read_text(encoding="utf-8").strip()

    assert metadata["implementation_status"] == "candidate_not_run"
    assert metrics["status"] == "not_run"
    assert "run_craftax_prompt_opt_qwen35_4b_gpt54_budget.sh" in command
