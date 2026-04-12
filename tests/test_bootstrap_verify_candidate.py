from __future__ import annotations

from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs" / "craftax_prompt_opt_qwen35_4b_codex_bootstrap_verify.yaml"
SOURCE_PATH = REPO_ROOT / "src" / "nanohorizon" / "baselines" / "prompt_opt.py"


def test_bootstrap_verify_config_adds_next_turn_positioning() -> None:
    payload = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    seed_prompt = str(payload["prompt"]["seed_prompt"])

    assert "finish the batch adjacent to a useful target" in seed_prompt
    assert "Prefer 3-action batches by default" in seed_prompt
    assert "non-redundant" in seed_prompt
    assert "craftax_interact" in seed_prompt


def test_prompt_opt_source_carries_adjacent_target_guidance() -> None:
    source = SOURCE_PATH.read_text(encoding="utf-8")

    assert "END_ADJACENT_TARGET_REQUIREMENT" in source
    assert "end the batch adjacent to a useful target" in source
    assert "next turn has a clear follow-up" in source
    assert "non-redundant 3-action batches" in source
