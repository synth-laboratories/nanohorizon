from __future__ import annotations

from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs" / "craftax_prompt_opt_qwen35_4b_codex_decision_brief.yaml"
SOURCE_PATH = REPO_ROOT / "src" / "nanohorizon" / "craftax_core" / "rollout.py"
METADATA_PATH = REPO_ROOT / "src" / "nanohorizon" / "craftax_core" / "metadata.py"


def test_candidate_config_uses_decision_brief_prompt() -> None:
    payload = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    seed_prompt = str(payload["prompt"]["seed_prompt"])

    assert "compact decision brief" in seed_prompt
    assert "reward-history window" in seed_prompt
    assert "stabilize, gather, craft, or explore" in seed_prompt
    assert "loop risk is high" in seed_prompt
    assert "craftax_interact" in seed_prompt


def test_rollout_source_threads_decision_context_json() -> None:
    source = SOURCE_PATH.read_text(encoding="utf-8")

    assert "Decision context JSON:" in source
    assert "_decision_context_prompt" in source
    assert "render_prompt_turn(" in source


def test_metadata_source_exposes_decision_brief() -> None:
    source = METADATA_PATH.read_text(encoding="utf-8")

    assert "DecisionBrief" in source
    assert "derive_decision_brief" in source
    assert "decision_brief_summary" in source
