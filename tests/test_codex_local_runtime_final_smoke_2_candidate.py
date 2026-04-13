from __future__ import annotations

import json
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs" / "craftax_prompt_opt_qwen35_4b_codex_local_runtime_final_smoke_2.yaml"
RECORD_DIR = (
    REPO_ROOT
    / "records"
    / "prompt_opt_1usd_gpt54_family"
    / "2026-04-13_codex_local_runtime_final_smoke_2"
)


def test_candidate_config_pushes_a_wood_first_bootstrap() -> None:
    payload = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    seed_prompt = str(payload["prompt"]["seed_prompt"])

    assert "tiny private" in seed_prompt
    assert "wood-first bootstrap" in seed_prompt
    assert "collect_wood" in seed_prompt
    assert "place_table" in seed_prompt
    assert "make_wood_pickaxe" in seed_prompt
    assert "collect_stone" in seed_prompt
    assert "do not reveal the todo list" in seed_prompt.lower()


def test_candidate_record_bundle_carries_proxy_smoke_results() -> None:
    metadata = json.loads((RECORD_DIR / "metadata.json").read_text(encoding="utf-8"))
    metrics = json.loads((RECORD_DIR / "metrics.json").read_text(encoding="utf-8"))
    prompt_bundle = json.loads((RECORD_DIR / "prompt_bundle.json").read_text(encoding="utf-8"))
    command = (RECORD_DIR / "command.txt").read_text(encoding="utf-8").strip()

    assert metadata["implementation_status"] == "candidate_smoke_passed"
    assert metrics["status"] == "proxy_smoke"
    assert metrics["candidate_mean_outcome_reward"] > metrics["baseline_mean_outcome_reward"]
    assert prompt_bundle["candidate_system_prompt"] == seed_prompt_from_config()
    assert "proxy_smoke.py" in command


def seed_prompt_from_config() -> str:
    payload = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    return str(payload["prompt"]["seed_prompt"])
