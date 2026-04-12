from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from nanohorizon.baselines import prompt_opt
from nanohorizon.baselines.prompt_opt import CraftaxPromptOptAdapter, PromptOptExample


REPO_ROOT = Path(__file__).resolve().parents[2]
BASELINE_CONFIG = REPO_ROOT / "configs" / "craftax_prompt_opt_qwen35_4b_gpt54_budget.yaml"
CANDIDATE_CONFIG = (
    REPO_ROOT / "configs" / "craftax_prompt_opt_qwen35_4b_codex_compact_follow_first.yaml"
)
RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def _fake_rollout_request(request: dict[str, Any]) -> dict[str, Any]:
    prompt = str(request["policy"]["config"]["system_prompt"]).lower()
    seed = int(request["env"]["seed"])

    compact = "keep the prompt compact and action-directed" in prompt
    follow_first = "follows the first todo item" in prompt
    resource_first = "move toward nearby trees" in prompt or "gatherable resources" in prompt
    loop_break = "replace the stale target item" in prompt or "loop-break fallback action" in prompt

    score = 0.05
    if compact:
        score += 0.25
    if follow_first:
        score += 0.25
    if resource_first:
        score += 0.20
    if loop_break:
        score += 0.20
    if seed % 2 == 0:
        score += 0.05

    score = round(min(score, 1.0), 2)
    if score >= 0.5:
        actions = ["move_right", "move_right", "do"]
        achievements = ["collect_wood"]
    else:
        actions = ["sleep", "move_up", "move_up"]
        achievements = []

    turns = [
        {
            "turn_index": 0,
            "prompt_messages": request["policy"]["config"].get("prompt_messages")
            or [{"role": "user", "content": f"seed={seed}"}],
            "assistant_text": f"<tool_call>{{\"name\":\"craftax_interact\",\"arguments\":{{\"actions_list\":{actions!r}}}}}</tool_call>",
            "reasoning_text": "Follow the active todo item and pick a short macro-action.",
            "actions": actions,
            "invalid_parse": False,
        }
    ]
    return {
        "rollout_id": f"proxy-{seed}",
        "trace_correlation_id": request["trace_correlation_id"],
        "trial_id": request["trial_id"],
        "success_status": "success",
        "reward_info": {"outcome_reward": score, "details": {"achievements": achievements}},
        "metadata": {"llm_call_count": 1, "achievements": achievements},
        "trace": {"inference": {"turns": turns}},
        "artifact": [{"turns": turns}],
    }


def _evaluate_prompt(config_path: Path) -> dict[str, Any]:
    config = prompt_opt.load_config(config_path)
    trainset, evalset = prompt_opt._load_seed_splits(config, config_path.parent)
    eval_seeds = [example.seed for example in evalset]
    repeated_eval_seeds = eval_seeds + eval_seeds
    system_prompt = str(config["prompt"]["seed_prompt"]).strip()

    original_run_rollout_request = prompt_opt.run_rollout_request
    prompt_opt.run_rollout_request = _fake_rollout_request
    try:
        adapter = CraftaxPromptOptAdapter(
            container_url="direct://local",
            inference_url="http://example.invalid/v1/chat/completions",
            inference_api_key="",
            request_model="proxy-model",
            rollout_cfg=dict(config["rollout"]),
        )
        dataset = [PromptOptExample(seed=seed, split="eval") for seed in repeated_eval_seeds]
        summary = prompt_opt._summarize_eval(
            dataset=dataset,
            candidate={"system_prompt": system_prompt},
            adapter=adapter,
            name=config_path.stem,
            output_dir=RESULTS_DIR,
        )
        return {"config": str(config_path), "seeds": repeated_eval_seeds, "summary": summary}
    finally:
        prompt_opt.run_rollout_request = original_run_rollout_request


def main() -> None:
    baseline = _evaluate_prompt(BASELINE_CONFIG)
    candidate = _evaluate_prompt(CANDIDATE_CONFIG)
    result = {
        "baseline": baseline,
        "candidate": candidate,
        "delta_mean_outcome_reward": candidate["summary"]["mean_outcome_reward"]
        - baseline["summary"]["mean_outcome_reward"],
        "delta_max_outcome_reward": candidate["summary"]["max_outcome_reward"]
        - baseline["summary"]["max_outcome_reward"],
    }
    result_path = RESULTS_DIR / "proxy_compare.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
