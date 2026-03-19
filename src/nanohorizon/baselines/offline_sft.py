from __future__ import annotations

import argparse
import json
import os
import signal
import time
from pathlib import Path
from statistics import mean
from typing import Any

from nanohorizon.common import (
    Timer,
    ensure_dir,
    load_config,
    read_jsonl,
    resolve_path,
    system_info,
    write_json,
    write_text,
)
from nanohorizon.crafter_data import build_sft_examples
from nanohorizon.eval_model import evaluate_model, reward_heuristic
from nanohorizon.openai_compat import chat_completion
from nanohorizon.train_lora import train_sft_with_trl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NanoHorizon offline Crafter SFT baseline")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", default="")
    return parser.parse_args()


def _generate_teacher_dataset(*, config: dict, config_dir: Path, output_dir: Path) -> Path:
    teacher_cfg = config["teacher_generation"]
    seed_rows = read_jsonl(resolve_path(teacher_cfg["seed_prompts_jsonl"], base_dir=config_dir))
    dataset_path = output_dir / "generated_sft_data.jsonl"
    generated_rows: list[str] = []
    for row in seed_rows:
        messages = row.get("messages")
        if not isinstance(messages, list):
            continue
        normalized_messages = [
            {
                "role": str(message.get("role") or "user"),
                "content": str(message.get("content") or ""),
            }
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
                os.getenv("NANOHORIZON_TEACHER_API_KEY") or os.getenv("OPENAI_API_KEY") or ""
            ),
        )
        raw_meta = row.get("metadata")
        meta_base: dict[str, Any] = raw_meta if isinstance(raw_meta, dict) else {}
        generated_rows.append(
            json.dumps(
                {
                    "messages": [
                        *normalized_messages,
                        {"role": "assistant", "content": assistant_text},
                    ],
                    "metadata": {
                        **meta_base,
                        "teacher_model": str(teacher_cfg["teacher_model"]),
                        "generated_within_budget_window": True,
                    },
                },
                sort_keys=True,
            )
        )
    dataset_path.write_text(
        "\n".join(generated_rows) + ("\n" if generated_rows else ""), encoding="utf-8"
    )
    return dataset_path


def _filter_teacher_dataset(
    *, dataset_path: Path, output_dir: Path, keep_top_fraction: float
) -> tuple[Path, dict]:
    rows = read_jsonl(dataset_path)
    scored_rows: list[tuple[float, dict]] = []
    for row in rows:
        messages = row.get("messages")
        if not isinstance(messages, list) or len(messages) < 2:
            continue
        user_message = next(
            (
                message
                for message in messages
                if isinstance(message, dict) and message.get("role") == "user"
            ),
            {},
        )
        assistant_message = messages[-1] if isinstance(messages[-1], dict) else {}
        reward = reward_heuristic(
            str(user_message.get("content") or ""), str(assistant_message.get("content") or "")
        )
        raw_meta = row.get("metadata")
        meta_base: dict[str, Any] = raw_meta if isinstance(raw_meta, dict) else {}
        scored_rows.append(
            (
                reward,
                {
                    **row,
                    "metadata": {
                        **meta_base,
                        "heuristic_reward": reward,
                    },
                },
            )
        )
    scored_rows.sort(key=lambda item: item[0], reverse=True)
    keep_count = max(1, int(len(scored_rows) * keep_top_fraction)) if scored_rows else 0
    kept_rows = scored_rows[:keep_count]
    kept = [json.dumps(row, sort_keys=True) for _, row in kept_rows]
    filtered_path = output_dir / "filtered_sft_data.jsonl"
    filtered_path.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")
    rewards = [reward for reward, _ in scored_rows]
    kept_rewards = [reward for reward, _ in kept_rows]
    summary = {
        "input_rows": len(rows),
        "kept_rows": len(kept),
        "keep_top_fraction": keep_top_fraction,
        "cutoff_reward": kept_rewards[-1] if kept_rewards else None,
        "mean_reward": mean(rewards) if rewards else 0.0,
        "kept_mean_reward": mean(kept_rewards) if kept_rewards else 0.0,
    }
    return filtered_path, summary


def _shutdown_local_teacher_if_requested() -> None:
    raw_pid = os.getenv("NANOHORIZON_LOCAL_TEACHER_PID", "").strip()
    if not raw_pid:
        return
    try:
        pid = int(raw_pid)
    except ValueError:
        return
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    except OSError:
        return
    for _ in range(20):
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            break
        time.sleep(1.0)
    os.environ.pop("NANOHORIZON_LOCAL_TEACHER_PID", None)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    config_dir = Path(args.config).expanduser().resolve().parent
    output_dir = ensure_dir(args.output_dir or config["output"]["root_dir"])
    timer = Timer()

    teacher_generation = config.get("teacher_generation", {})
    teacher_enabled = bool(teacher_generation.get("enabled", False))
    dataset_path = resolve_path(config["data"]["dataset_jsonl"], base_dir=config_dir)
    if teacher_enabled:
        dataset_path = _generate_teacher_dataset(
            config=config, config_dir=config_dir, output_dir=output_dir
        )
        _shutdown_local_teacher_if_requested()
    filter_cfg = config.get("filtering", {})
    if bool(filter_cfg.get("enabled", False)):
        dataset_path, filter_summary = _filter_teacher_dataset(
            dataset_path=dataset_path,
            output_dir=output_dir,
            keep_top_fraction=float(filter_cfg.get("keep_top_fraction", 0.5)),
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
    eval_summary = evaluate_model(
        base_model=config["model"]["model"],
        adapter_dir=output_dir / "adapter",
        eval_prompts_jsonl=resolve_path(
            config["evaluation"]["eval_prompts_jsonl"], base_dir=config_dir
        ),
        output_dir=output_dir,
        max_length=int(config["training"]["max_length"]),
        max_new_tokens=int(config["evaluation"].get("max_new_tokens", 64)),
        summary_name="finetuned_eval_summary.json",
    )
    base_summary = evaluate_model(
        base_model=config["model"]["model"],
        adapter_dir=None,
        eval_prompts_jsonl=resolve_path(
            config["evaluation"]["eval_prompts_jsonl"], base_dir=config_dir
        ),
        output_dir=output_dir,
        max_length=int(config["training"]["max_length"]),
        max_new_tokens=int(config["evaluation"].get("max_new_tokens", 64)),
        summary_name="base_eval_summary.json",
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
        "finetuned_exact_match_rate": eval_summary["exact_match_rate"],
        "finetuned_mean_heuristic_reward": eval_summary["mean_heuristic_reward"],
        "base_exact_match_rate": base_summary["exact_match_rate"],
        "base_mean_heuristic_reward": base_summary["mean_heuristic_reward"],
        "reward_delta": eval_summary["mean_heuristic_reward"]
        - base_summary["mean_heuristic_reward"],
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
    write_text(
        output_dir / "command.txt",
        f"python -m nanohorizon.baselines.offline_sft --config {Path(args.config).resolve()}\n",
    )


if __name__ == "__main__":
    main()
