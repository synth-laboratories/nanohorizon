#!/usr/bin/env python3
"""Structural verifier for the Video Validation Run prompt-opt candidate."""

from __future__ import annotations

import json
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs" / "craftax_prompt_opt_qwen35_4b_codex_video_validation_run.yaml"
RECORD_DIR = (
    REPO_ROOT
    / "records"
    / "prompt_opt_1usd_gpt54_family"
    / "2026-04-11_codex_video_validation_run"
)


def main() -> int:
    payload = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    seed_prompt = str(payload["prompt"]["seed_prompt"])

    assert "video validation run" in seed_prompt.lower()
    assert "tiny private todo list with exactly three items" in seed_prompt
    assert "most urgent blocker or danger" in seed_prompt
    assert "fallback progress action" in seed_prompt
    assert "replace the stale target item" in seed_prompt
    assert "inspect quickly during a video validation run" in seed_prompt
    assert "Do not reveal the todo list or scratchpad" in seed_prompt

    metadata = json.loads((RECORD_DIR / "metadata.json").read_text(encoding="utf-8"))
    metrics = json.loads((RECORD_DIR / "metrics.json").read_text(encoding="utf-8"))
    command = (RECORD_DIR / "command.txt").read_text(encoding="utf-8").strip()

    assert metadata["name"] == "codex_video_validation_run"
    assert metadata["implementation_status"] == "candidate_not_run"
    assert metrics["status"] == "not_run"
    assert "run_craftax_prompt_opt_qwen35_4b_gpt54_budget.sh" in command

    print("verification: passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

