from __future__ import annotations

from pathlib import Path

import yaml


PROMPT_CONFIGS = [
    Path("configs/craftax_prompt_opt_qwen35_4b_gpt54_budget.yaml"),
    Path("configs/craftax_prompt_opt_qwen35_4b_gepa_smoke.yaml"),
    Path("configs/craftax_prompt_opt_qwen35_4b_gepa_eval20.yaml"),
    Path("configs/craftax_prompt_opt_gemini25_flash_lite_local_eval20.yaml"),
]


def test_craftax_prompt_opt_configs_keep_reasoning_first_contract() -> None:
    for config_path in PROMPT_CONFIGS:
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        prompt = str(config["prompt"]["seed_prompt"])

        assert "First reason briefly and privately" in prompt, config_path
        assert "recent action history" in prompt, config_path
        assert "trajectory is looping" in prompt, config_path
        assert "use `do` only when a useful nearby object or resource is actually reachable" in prompt, config_path
        assert "Do not output JSON, prose, or a plain-text action list outside the tool call." in prompt, config_path
