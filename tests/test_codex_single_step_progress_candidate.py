from __future__ import annotations

import json
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs" / "craftax_prompt_opt_qwen35_4b_codex_single_step_progress.yaml"
RECORD_DIR = (
    REPO_ROOT
    / "records"
    / "prompt_opt_1usd_gpt54_family"
    / "2026-04-13_codex_single_step_progress"
)


def test_candidate_config_reverts_to_single_step_progress() -> None:
    payload = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    seed_prompt = str(payload["prompt"]["seed_prompt"])

    assert "Return 1 valid full-Craftax action" in seed_prompt
    assert "move toward nearby trees" in seed_prompt
    assert "use `do` only when adjacent" in seed_prompt
    assert "smallest useful movement change" in seed_prompt
    assert "todo list" not in seed_prompt


def test_candidate_record_bundle_is_present_and_marked_not_run() -> None:
    metadata = json.loads((RECORD_DIR / "metadata.json").read_text(encoding="utf-8"))
    metrics = json.loads((RECORD_DIR / "metrics.json").read_text(encoding="utf-8"))
    command = (RECORD_DIR / "command.txt").read_text(encoding="utf-8").strip()

    assert metadata["implementation_status"] == "candidate_not_run"
    assert metrics["status"] == "not_run"
    assert "run_craftax_prompt_opt_qwen35_4b_gpt54_budget.sh" in command
