from __future__ import annotations

from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs" / "craftax_prompt_opt_qwen35_4b_local_runtime_final_smoke_1.yaml"


def test_local_runtime_final_smoke_1_config_uses_loop_breaking_prompt() -> None:
    payload = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    seed_prompt = str(payload["prompt"]["seed_prompt"])

    assert "tiny private" in seed_prompt
    assert "todo list with exactly three items" in seed_prompt
    assert "fallback action that changes state" in seed_prompt
    assert "replace the stale target item" in seed_prompt
    assert "Prefer nearby trees" in seed_prompt
    assert "move toward the current target or use the fallback action" in seed_prompt
    assert "craftax_interact" in seed_prompt
