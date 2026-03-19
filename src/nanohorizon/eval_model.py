from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from nanohorizon.common import ensure_dir, read_jsonl, write_json, write_text
from nanohorizon.crafter_data import flatten_messages
from nanohorizon.train_lora import generate_with_model


def parse_actions(text: str) -> list[str]:
    stripped = text.strip()
    if not stripped:
        return []
    try:
        payload = json.loads(stripped)
        actions = payload.get("actions")
        if isinstance(actions, list):
            return [str(item).strip() for item in actions if str(item).strip()]
    except Exception:
        pass
    return []


def reward_heuristic(user_observation: str, assistant_text: str) -> float:
    obs = user_observation.lower()
    actions = [action.lower() for action in parse_actions(assistant_text)]
    if not actions:
        return 0.0

    score = 0.0
    if "tree adjacent" in obs and any(action in {"move_right", "do"} for action in actions):
        score += 1.0
    if "wood=3" in obs and "table not placed" in obs and any(action == "place_table" for action in actions):
        score += 1.0
    if "wood pickaxe" in obs and any(action == "make_wood_pickaxe" for action in actions):
        score += 1.0
    if "energy is low" in obs and "night" in obs and any(action == "sleep" for action in actions):
        score += 1.0
    if all(action in {"move_left", "move_right", "move_up", "move_down", "do", "sleep", "place_table", "make_wood_pickaxe"} for action in actions):
        score += 0.25
    return score


def evaluate_model(
    *,
    base_model: str,
    eval_prompts_jsonl: str | Path,
    output_dir: str | Path,
    max_length: int,
    max_new_tokens: int,
    adapter_dir: str | Path | None = None,
    summary_name: str = "eval_summary.json",
) -> dict[str, Any]:
    rows = read_jsonl(eval_prompts_jsonl)
    prompts: list[str] = []
    observations: list[str] = []
    target_actions: list[list[str]] = []
    for row in rows:
        messages = row.get("messages")
        if not isinstance(messages, list):
            continue
        prompts.append(flatten_messages(messages))
        user_message = next((message for message in messages if isinstance(message, dict) and message.get("role") == "user"), {})
        observations.append(str(user_message.get("content") or ""))
        metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        target_actions.append([str(item).lower() for item in metadata.get("target_actions", []) if str(item).strip()])

    predictions = generate_with_model(
        base_model=base_model,
        prompts=prompts,
        max_length=max_length,
        max_new_tokens=max_new_tokens,
        adapter_dir=adapter_dir,
    )
    exact = 0
    rewards: list[float] = []
    details: list[dict[str, Any]] = []
    for prompt, observation, expected, predicted in zip(prompts, observations, target_actions, predictions, strict=False):
        predicted_actions = [action.lower() for action in parse_actions(predicted)]
        hit = all(action in predicted_actions for action in expected)
        reward = reward_heuristic(observation, predicted)
        exact += int(hit)
        rewards.append(reward)
        details.append(
            {
                "prompt": prompt,
                "observation": observation,
                "expected": expected,
                "predicted_actions": predicted_actions,
                "raw_prediction": predicted,
                "exact_match": hit,
                "heuristic_reward": reward,
            }
        )
    result = {
        "num_eval_examples": len(details),
        "num_exact_matches": exact,
        "exact_match_rate": (exact / len(details)) if details else 0.0,
        "mean_heuristic_reward": (sum(rewards) / len(rewards)) if rewards else 0.0,
        "details": details,
    }
    out_dir = ensure_dir(output_dir)
    write_json(out_dir / summary_name, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a Crafter model or adapter with the standard NanoHorizon heuristic.")
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter-dir", default="")
    parser.add_argument("--eval-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    args = parser.parse_args()

    result = evaluate_model(
        base_model=args.base_model,
        adapter_dir=Path(args.adapter_dir).expanduser().resolve() if args.adapter_dir else None,
        eval_prompts_jsonl=args.eval_jsonl,
        output_dir=args.output_dir,
        max_length=args.max_length,
        max_new_tokens=args.max_new_tokens,
    )
    write_text(
        Path(args.output_dir).expanduser().resolve() / "eval_command.txt",
        " ".join(
            [
                "python",
                "-m",
                "nanohorizon.eval_model",
                "--base-model",
                args.base_model,
                "--eval-jsonl",
                str(Path(args.eval_jsonl).expanduser().resolve()),
                "--output-dir",
                str(Path(args.output_dir).expanduser().resolve()),
            ]
            + (["--adapter-dir", str(Path(args.adapter_dir).expanduser().resolve())] if args.adapter_dir else [])
        )
        + "\n",
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
