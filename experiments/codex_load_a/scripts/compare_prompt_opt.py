from __future__ import annotations

import json
import os
from pathlib import Path
from statistics import mean

from nanohorizon.baselines import prompt_opt


REPO_ROOT = Path(__file__).resolve().parents[3]
BASELINE_CONFIG = REPO_ROOT / "configs" / "craftax_prompt_opt_qwen35_4b_gpt54_budget.yaml"
CANDIDATE_CONFIG = REPO_ROOT / "configs" / "craftax_prompt_opt_qwen35_4b_codex_todo_refresh_gate.yaml"
RESULT_PATH = REPO_ROOT / "experiments" / "codex_load_a" / "results" / "baseline_vs_candidate.json"

# The Modal vLLM endpoint recorded in the checked-in reference run.
REFERENCE_INFERENCE_URL = (
    "https://synth-laboratories--nanohorizon-craftax-prompt-opt-p-a5e20a-dev.modal.run"
    "/v1/chat/completions"
)
REQUEST_MODEL = "qwen35-4b-prompt-opt"
REQUEST_API_KEY = "nanohorizon-prompt-opt-key"


def _load_candidate_prompt(config_path: Path) -> str:
    payload = prompt_opt.load_config(config_path)
    return str(payload["prompt"]["seed_prompt"]).strip()


def _load_evalset(config_path: Path) -> tuple[list[prompt_opt.PromptOptExample], dict[str, object]]:
    config = prompt_opt.load_config(config_path)
    trainset, valset = prompt_opt._load_seed_splits(config, config_path.parent)
    return valset, dict(config["rollout"])


def _evaluate(prompt_text: str, *, config_path: Path, label: str) -> dict[str, object]:
    dataset, rollout_cfg = _load_evalset(config_path)
    adapter = prompt_opt.CraftaxPromptOptAdapter(
        container_url="direct://local",
        inference_url=REFERENCE_INFERENCE_URL,
        inference_api_key=REQUEST_API_KEY,
        request_model=REQUEST_MODEL,
        rollout_cfg=rollout_cfg,
    )
    batch = adapter.evaluate(dataset, {"system_prompt": prompt_text}, capture_traces=True)
    rewards = [
        float(output.get("reward_info", {}).get("outcome_reward", 0.0))
        for output in batch.outputs
        if isinstance(output, dict)
    ]
    details = []
    for example, reward, output in zip(dataset, rewards, batch.outputs, strict=False):
        details.append(
            {
                "seed": example.seed,
                "reward": reward,
                "llm_call_count": int(prompt_opt.rollout_llm_call_count(output) if isinstance(output, dict) else 0),
                "achievements": prompt_opt._achievement_summary(output) if isinstance(output, dict) else [],
                "success_status": output.get("success_status") if isinstance(output, dict) else None,
                "error": output.get("error") if isinstance(output, dict) else None,
            }
        )
    return {
        "label": label,
        "requested_rollouts": len(dataset),
        "mean_outcome_reward": mean(rewards) if rewards else 0.0,
        "max_outcome_reward": max(rewards) if rewards else 0.0,
        "details": details,
    }


def main() -> int:
    baseline_prompt = _load_candidate_prompt(BASELINE_CONFIG)
    candidate_prompt = _load_candidate_prompt(CANDIDATE_CONFIG)
    baseline = _evaluate(baseline_prompt, config_path=CANDIDATE_CONFIG, label="baseline")
    candidate = _evaluate(candidate_prompt, config_path=CANDIDATE_CONFIG, label="candidate")
    result = {
        "baseline": baseline,
        "candidate": candidate,
        "delta_mean_outcome_reward": candidate["mean_outcome_reward"] - baseline["mean_outcome_reward"],
        "baseline_config": str(BASELINE_CONFIG.relative_to(REPO_ROOT)),
        "candidate_config": str(CANDIDATE_CONFIG.relative_to(REPO_ROOT)),
        "inference_url": REFERENCE_INFERENCE_URL,
        "request_model": REQUEST_MODEL,
        "request_api_key": REQUEST_API_KEY,
        "environment": {
            "container_url": "direct://local",
            "uv_command": "uv run --no-sync --with pytest --with pyyaml --with httpx --with gepa --with modal --with numpy --with pillow --with-editable /workspace python experiments/codex_load_a/scripts/compare_prompt_opt.py",
        },
    }
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
