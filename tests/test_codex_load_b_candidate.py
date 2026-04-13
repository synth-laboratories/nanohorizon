from __future__ import annotations

import json
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs" / "craftax_prompt_opt_qwen35_4b_codex_load_b.yaml"
RECORD_DIR = (
    REPO_ROOT
    / "records"
    / "prompt_opt_1usd_gpt54_family"
    / "2026-04-13_codex_load_b"
)


def test_candidate_config_uses_load_order_prompt() -> None:
    payload = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    seed_prompt = str(payload["prompt"]["seed_prompt"])

    assert "tiny private plan with exactly three items" in seed_prompt
    assert "Load the state in that order before acting" in seed_prompt
    assert "replace the stale target item" in seed_prompt
    assert "use `do` only when adjacent to a useful target" in seed_prompt
    assert "Return 3 or 4 valid full-Craftax actions" in seed_prompt
    assert "Do not output JSON, prose, or a plain-text action list" in seed_prompt


def test_candidate_record_bundle_is_present_and_proxy_evaluated() -> None:
    metadata = json.loads((RECORD_DIR / "metadata.json").read_text(encoding="utf-8"))
    metrics = json.loads((RECORD_DIR / "metrics.json").read_text(encoding="utf-8"))
    comparison = json.loads((RECORD_DIR / "comparison.json").read_text(encoding="utf-8"))

    assert metadata["implementation_status"] == "candidate_proxy_evaluated"
    assert metrics["status"] == "proxy_evaluated"
    assert comparison["candidate_mean_outcome_reward"] > comparison["baseline_mean_outcome_reward"]
    assert comparison["num_eval_rollouts"] == 8
