from __future__ import annotations

import argparse
import os
import random
from pathlib import Path
from typing import Any

import yaml

from nanohorizon.common import (
    Timer,
    ensure_dir,
    load_config,
    now_utc_iso,
    read_jsonl,
    resolve_path,
    system_info,
    write_json,
    write_text,
)
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
                    "content": f"Improve this Crafter system prompt for a Qwen/Qwen3.5-4B policy:\n\n{seed_prompt}",
                },
            ],
            max_tokens=300,
            temperature=0.6,
        )
    except Exception:
        return _local_mutation(seed_prompt)


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).expanduser().resolve()
    config = load_config(config_path)
    base_dir = config_path.parent
    output_dir = ensure_dir(
        args.output_dir or resolve_path(config["output"]["root_dir"], base_dir=base_dir)
    )
    timer = Timer()

    tasks = read_jsonl(resolve_path(config["data"]["task_jsonl"], base_dir=base_dir))
    seed_prompt = str(config["prompt"]["seed_prompt"])
    proposer_model = str(config["optimizer"]["proposer_model"])
    num_candidates = int(config["search"]["num_candidates"])

    best_prompt = seed_prompt
    best_score = _score_prompt(seed_prompt, tasks)
    evaluated: list[dict[str, Any]] = [
        {"prompt": seed_prompt, "score": best_score, "source": "seed"}
    ]

    for _ in range(max(1, num_candidates)):
        candidate = _propose_candidate(best_prompt, proposer_model)
        candidate_score = _score_prompt(candidate, tasks)
        evaluated.append({"prompt": candidate, "score": candidate_score, "source": proposer_model})
        if candidate_score > best_score:
            best_prompt = candidate
            best_score = candidate_score

    write_json(
        output_dir / "prompt_bundle.json", {"system_prompt": best_prompt, "evaluated": evaluated}
    )
    track_id = "prompt_opt_1usd_gpt54_family"
    run_config: dict[str, Any] = {
        "track": track_id,
        "task": str(config["task"]["name"]),
        "base_model": str(config["policy"]["model"]),
        "optimizer_budget_usd": float(config["optimizer"]["budget_usd"]),
        "optimizer_models": list(config["optimizer"]["allowed_models"]),
    }
    write_text(output_dir / "run_config.yaml", yaml.safe_dump(run_config, sort_keys=True))
    write_json(
        output_dir / "metadata.json",
        {
            "name": os.environ.get("NANOHORIZON_RECORD_NAME", "prompt_opt_run"),
            "track": track_id,
            "task": str(config["task"]["name"]),
            "base_model": str(config["policy"]["model"]),
            "optimizer_budget_usd": float(config["optimizer"]["budget_usd"]),
            "optimizer_models": list(config["optimizer"]["allowed_models"]),
            "created_at": now_utc_iso()[:10],
        },
    )
    write_json(
        output_dir / "metrics.json",
        {
            "track": track_id,
            "baseline": "gepa_style_prompt_search",
            "best_score": best_score,
            "num_candidates": len(evaluated),
            "optimizer_budget_usd": float(config["optimizer"]["budget_usd"]),
            "elapsed_minutes": timer.elapsed_minutes,
        },
    )
    write_json(output_dir / "system_info.json", system_info())
    write_text(
        output_dir / "command.txt",
        f"python -m nanohorizon.baselines.prompt_opt --config {config_path}\n",
    )


if __name__ == "__main__":
    main()
