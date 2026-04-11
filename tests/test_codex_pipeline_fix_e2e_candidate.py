from __future__ import annotations

import json
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs" / "craftax_prompt_opt_qwen35_4b_codex_pipeline_fix_e2e.yaml"
SCRIPT_PATH = REPO_ROOT / "scripts" / "run_craftax_prompt_opt_pipeline_fix_e2e.sh"
RECORD_DIR = (
    REPO_ROOT
    / "records"
    / "prompt_opt_1usd_gpt54_family"
    / "2026-04-11_codex_pipeline_fix_e2e"
)


def test_candidate_config_uses_pipeline_fix_prompt() -> None:
    payload = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    seed_prompt = str(payload["prompt"]["seed_prompt"])

    assert "end-to-end pipeline" in seed_prompt
    assert "refresh a tiny private todo list with exactly three items" in seed_prompt
    assert "follow the active todo item" in seed_prompt
    assert "replace the stale target item immediately" in seed_prompt
    assert "Do not reveal the todo list" in seed_prompt


def test_candidate_wrapper_targets_the_candidate_config() -> None:
    script = SCRIPT_PATH.read_text(encoding="utf-8")

    assert "craftax_prompt_opt_qwen35_4b_codex_pipeline_fix_e2e.yaml" in script
    assert "run_craftax_prompt_opt_qwen35_4b_gpt54_budget.sh" in script


def test_candidate_record_bundle_is_present_and_marked_not_run() -> None:
    metadata = json.loads((RECORD_DIR / "metadata.json").read_text(encoding="utf-8"))
    metrics = json.loads((RECORD_DIR / "metrics.json").read_text(encoding="utf-8"))
    command = (RECORD_DIR / "command.txt").read_text(encoding="utf-8").strip()

    assert metadata["implementation_status"] == "candidate_not_run"
    assert metrics["status"] == "not_run"
    assert "run_craftax_prompt_opt_pipeline_fix_e2e.sh" in command
