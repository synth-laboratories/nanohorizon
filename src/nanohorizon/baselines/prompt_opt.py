from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from nanohorizon.common import Timer, ensure_dir, load_config, read_jsonl, system_info, write_json, write_text
from nanohorizon.openai_compat import chat_completion


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NanoHorizon Crafter prompt optimization baseline")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", default="")
    return parser.parse_args()


def _score_prompt(prompt: str, tasks: list[dict[str, Any]]) -> float:
    score = 0.0
    lowered = prompt.lower()
    for task in tasks:
        target_actions = [str(item).lower() for item in task.get("target_actions", [])]
        for action in target_actions:
            if action in lowered:
                score += 1.0
        if "progression" in lowered:
            score += 0.25
        if "sleep" in lowered:
            score += 0.1
    return score / max(len(tasks), 1)


def _local_mutation(seed_prompt: str) -> str:
    variants = [
        seed_prompt + " Focus on early wood-table-pickaxe progression.",
        seed_prompt + " Avoid repeating actions when no state changes.",
        seed_prompt + " Prefer valid adjacent interactions before crafting.",
    ]
    return random.choice(variants)


def _propose_candidate(seed_prompt: str, proposer_model: str) -> str:
    try:
        return chat_completion(
            model=proposer_model,
            messages=[
                {
                    "role": "system",
                    "content": "You improve prompts for Crafter agents. Return only the improved system prompt.",
                },
                {
                    "role": "user",
                    "content": f"Improve this Crafter system prompt for a Qwen/Qwen3.5-0.8B policy:\n\n{seed_prompt}",
                },
            ],
            max_tokens=300,
            temperature=0.6,
        )
    except Exception:
        return _local_mutation(seed_prompt)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    output_dir = ensure_dir(args.output_dir or config["output"]["root_dir"])
    timer = Timer()

    tasks = read_jsonl(config["data"]["task_jsonl"])
    seed_prompt = str(config["prompt"]["seed_prompt"])
    proposer_model = str(config["optimizer"]["proposer_model"])
    num_candidates = int(config["search"]["num_candidates"])

    best_prompt = seed_prompt
    best_score = _score_prompt(seed_prompt, tasks)
    evaluated: list[dict[str, Any]] = [{"prompt": seed_prompt, "score": best_score, "source": "seed"}]

    for _ in range(max(1, num_candidates)):
        candidate = _propose_candidate(best_prompt, proposer_model)
        candidate_score = _score_prompt(candidate, tasks)
        evaluated.append({"prompt": candidate, "score": candidate_score, "source": proposer_model})
        if candidate_score > best_score:
            best_prompt = candidate
            best_score = candidate_score

    write_json(output_dir / "prompt_bundle.json", {"system_prompt": best_prompt, "evaluated": evaluated})
    write_json(
        output_dir / "metrics.json",
        {
            "track": "prompt_opt_1usd_gpt54_family",
            "baseline": "gepa_style_prompt_search",
            "best_score": best_score,
            "num_candidates": len(evaluated),
            "optimizer_budget_usd": float(config["optimizer"]["budget_usd"]),
            "elapsed_minutes": timer.elapsed_minutes,
        },
    )
    write_json(output_dir / "system_info.json", system_info())
    write_text(output_dir / "command.txt", f"python -m nanohorizon.baselines.prompt_opt --config {Path(args.config).resolve()}\n")


if __name__ == "__main__":
    main()
