from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from nanohorizon.common import Timer, ensure_dir, load_config, read_jsonl, system_info, write_json, write_text
from nanohorizon.crafter_data import build_sft_examples, flatten_messages
from nanohorizon.openai_compat import chat_completion
from nanohorizon.train_lora import generate_with_adapter, train_sft_with_trl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NanoHorizon offline Crafter SFT baseline")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", default="")
    return parser.parse_args()


def _generate_teacher_dataset(*, config: dict, output_dir: Path) -> Path:
    teacher_cfg = config["teacher_generation"]
    seed_rows = read_jsonl(teacher_cfg["seed_prompts_jsonl"])
    dataset_path = output_dir / "generated_sft_data.jsonl"
    generated_rows: list[str] = []
    for row in seed_rows:
        messages = row.get("messages")
        if not isinstance(messages, list):
            continue
        normalized_messages = [
            {"role": str(message.get("role") or "user"), "content": str(message.get("content") or "")}
            for message in messages
            if isinstance(message, dict)
        ]
        assistant_text = chat_completion(
            model=str(teacher_cfg["teacher_model"]),
            messages=normalized_messages,
            max_tokens=int(teacher_cfg.get("max_tokens", 128)),
            temperature=float(teacher_cfg.get("temperature", 0.2)),
            base_url=str(
                os.getenv("NANOHORIZON_TEACHER_BASE_URL")
                or teacher_cfg.get("teacher_base_url")
                or os.getenv("OPENAI_BASE_URL")
                or "https://api.openai.com/v1"
            ),
            api_key=str(
                os.getenv("NANOHORIZON_TEACHER_API_KEY")
                or os.getenv("OPENAI_API_KEY")
                or ""
            ),
        )
        generated_rows.append(
            json.dumps(
                {
                    "messages": [*normalized_messages, {"role": "assistant", "content": assistant_text}],
                    "metadata": {
                        **(row.get("metadata") if isinstance(row.get("metadata"), dict) else {}),
                        "teacher_model": str(teacher_cfg["teacher_model"]),
                        "generated_within_budget_window": True,
                    },
                },
                sort_keys=True,
            )
        )
    dataset_path.write_text("\n".join(generated_rows) + ("\n" if generated_rows else ""), encoding="utf-8")
    return dataset_path


def _parse_actions(text: str) -> list[str]:
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


def _reward_heuristic(user_observation: str, assistant_text: str) -> float:
    obs = user_observation.lower()
    actions = [action.lower() for action in _parse_actions(assistant_text)]
    if not actions:
        return 0.0

    score = 0.0
    if "tree adjacent" in obs and any(action in {"move_right", "do"} for action in actions):
        score += 1.0
    if "wood=3" in obs and "table not placed" in obs and any(action == "place_table" for action in actions):
        score += 1.0
    if "wood pickaxe yet" in obs and any(action == "make_wood_pickaxe" for action in actions):
        score += 1.0
    if "energy is low" in obs and "night" in obs and any(action == "sleep" for action in actions):
        score += 1.0
    if all(action in {"move_left", "move_right", "move_up", "move_down", "do", "sleep", "place_table", "make_wood_pickaxe"} for action in actions):
        score += 0.25
    return score


def _filter_teacher_dataset(*, dataset_path: Path, output_dir: Path, min_reward: float) -> tuple[Path, dict]:
    rows = read_jsonl(dataset_path)
    kept: list[str] = []
    rewards: list[float] = []
    for row in rows:
        messages = row.get("messages")
        if not isinstance(messages, list) or len(messages) < 2:
            continue
        user_message = next((message for message in messages if isinstance(message, dict) and message.get("role") == "user"), {})
        assistant_message = messages[-1] if isinstance(messages[-1], dict) else {}
        reward = _reward_heuristic(str(user_message.get("content") or ""), str(assistant_message.get("content") or ""))
        rewards.append(reward)
        if reward >= min_reward:
            row = {
                **row,
                "metadata": {
                    **(row.get("metadata") if isinstance(row.get("metadata"), dict) else {}),
                    "heuristic_reward": reward,
                },
            }
            kept.append(json.dumps(row, sort_keys=True))
    filtered_path = output_dir / "filtered_sft_data.jsonl"
    filtered_path.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")
    summary = {
        "input_rows": len(rows),
        "kept_rows": len(kept),
        "min_reward": min_reward,
        "mean_reward": (sum(rewards) / len(rewards)) if rewards else 0.0,
    }
    return filtered_path, summary


def _evaluate_adapter(*, config: dict, adapter_dir: Path, output_dir: Path) -> dict:
    eval_cfg = config.get("evaluation", {})
    eval_rows = read_jsonl(eval_cfg["eval_prompts_jsonl"])
    prompts: list[str] = []
    target_actions: list[list[str]] = []
    for row in eval_rows:
        messages = row.get("messages")
        if not isinstance(messages, list):
            continue
        prompts.append(flatten_messages(messages))
        metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        target_actions.append([str(item).lower() for item in metadata.get("target_actions", []) if str(item).strip()])
    predictions = generate_with_adapter(
        base_model=config["model"]["model"],
        adapter_dir=adapter_dir,
        prompts=prompts,
        max_length=int(config["training"]["max_length"]),
        max_new_tokens=int(eval_cfg.get("max_new_tokens", 64)),
    )
    correct = 0
    details: list[dict] = []
    for prompt, expected, predicted in zip(prompts, target_actions, predictions, strict=False):
        predicted_actions = [action.lower() for action in _parse_actions(predicted)]
        hit = all(action in predicted_actions for action in expected)
        correct += int(hit)
        details.append({"prompt": prompt, "expected": expected, "predicted": predicted_actions, "raw_prediction": predicted, "match": hit})
    result = {
        "num_eval_examples": len(details),
        "num_exact_matches": correct,
        "exact_match_rate": (correct / len(details)) if details else 0.0,
        "details": details,
    }
    write_json(output_dir / "eval_summary.json", result)
    return result


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    output_dir = ensure_dir(args.output_dir or config["output"]["root_dir"])
    timer = Timer()

    teacher_generation = config.get("teacher_generation", {})
    teacher_enabled = bool(teacher_generation.get("enabled", False))
    dataset_path = Path(config["data"]["dataset_jsonl"]).expanduser().resolve()
    if teacher_enabled:
        dataset_path = _generate_teacher_dataset(config=config, output_dir=output_dir)
    filter_cfg = config.get("filtering", {})
    if bool(filter_cfg.get("enabled", False)):
        dataset_path, filter_summary = _filter_teacher_dataset(
            dataset_path=dataset_path,
            output_dir=output_dir,
            min_reward=float(filter_cfg.get("min_reward", 0.5)),
        )
        write_json(output_dir / "filter_summary.json", filter_summary)

    examples = build_sft_examples(read_jsonl(dataset_path))
    result = train_sft_with_trl(
        base_model=config["model"]["model"],
        examples=examples,
        output_dir=output_dir / "adapter",
        learning_rate=float(config["training"]["learning_rate"]),
        epochs=int(config["training"]["epochs"]),
        max_length=int(config["training"]["max_length"]),
        max_steps=int(config["training"]["max_steps"]),
        lora_rank=int(config["training"]["lora_rank"]),
        per_device_train_batch_size=int(config["training"].get("per_device_train_batch_size", 1)),
        gradient_accumulation_steps=int(config["training"].get("gradient_accumulation_steps", 1)),
    )
    eval_summary = _evaluate_adapter(
        config=config,
        adapter_dir=output_dir / "adapter",
        output_dir=output_dir,
    )

    metrics = {
        "track": "offline_20min_1xa100_40gb",
        "baseline": "teacher_sft_filtered_eval",
        "started_at": timer.started_at,
        "ended_at": timer.ended_at,
        "examples_seen": result.examples_seen,
        "optimizer_steps": result.optimizer_steps,
        "mean_loss": result.mean_loss,
        "elapsed_minutes": timer.elapsed_minutes,
        "dataset_jsonl": str(dataset_path),
        "teacher_generation_enabled": teacher_enabled,
        "teacher_model": teacher_generation.get("teacher_model", ""),
        "eval_exact_match_rate": eval_summary["exact_match_rate"],
    }
    write_json(output_dir / "metrics.json", metrics)
    write_json(output_dir / "system_info.json", system_info())
    write_json(
        output_dir / "run_timing.json",
        {
            "started_at": timer.started_at,
            "ended_at": timer.ended_at,
            "elapsed_minutes": timer.elapsed_minutes,
            "budget_minutes": float(config["budget"]["wall_clock_minutes"]),
        },
    )
    write_text(output_dir / "command.txt", f"python -m nanohorizon.baselines.offline_sft --config {Path(args.config).resolve()}\n")


if __name__ == "__main__":
    main()
