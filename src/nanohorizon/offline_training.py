from __future__ import annotations

import argparse
import asyncio
import json
import os
import signal
import time
from dataclasses import dataclass
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
from nanohorizon.crafter_data import (
    build_openai_sft_rows_from_rollouts,
    build_sft_examples,
    collect_rollouts_concurrently_with_summary,
    reward_quantile_threshold,
    rollout_llm_call_count,
    rollout_outcome_reward,
    summarize_rollouts,
)
from nanohorizon.eval_model import evaluate_model
from nanohorizon.train_lora import train_sft_with_trl


@dataclass
class TrainingExecution:
    output_dir: str
    adapter_dir: str
    examples_seen: int
    optimizer_steps: int
    mean_loss: float
    backend: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NanoHorizon offline Crafter training")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", default="")
    return parser.parse_args()


def _rollout_system_prompt(*, thinking_budget_tokens: int) -> str:
    return (
        "You are a Crafter teacher policy.\n"
        f"You may think for up to about {thinking_budget_tokens} tokens before answering.\n"
        "Return a short useful macro-action with 3-4 valid Crafter actions.\n"
        "Use movement to explore when nothing useful is adjacent.\n"
        "Use 'do' only when facing a useful nearby object or resource.\n"
        "Read the recent action history and avoid repeating unproductive loops.\n"
        "Use the provided `crafter_interact` tool exactly once for the final answer.\n"
        "Do not return plain text actions or JSON.\n"
        "Your final assistant action must be a tool call with valid Crafter actions."
    )


def _normalize_inference_url(raw_url: str) -> str:
    url = str(raw_url or "").strip()
    if not url:
        return ""
    if url.endswith("/chat/completions"):
        return url
    if url.endswith("/v1"):
        return f"{url}/chat/completions"
    if url.endswith("/v1/"):
        return f"{url}chat/completions"
    return url.rstrip("/") + "/v1/chat/completions"


def _row_priority(row: dict[str, Any]) -> tuple[float, float, float, int]:
    metadata = row.get("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
    outcome_reward = float(metadata.get("outcome_reward", 0.0) or 0.0)
    return_to_go = float(metadata.get("return_to_go", 0.0) or 0.0)
    decision_reward = float(metadata.get("decision_reward", 0.0) or 0.0)
    turn_index = int(metadata.get("turn_index", 0) or 0)
    return (outcome_reward, return_to_go, decision_reward, -turn_index)


def _filter_rows_by_priority(rows: list[dict[str, Any]], *, keep_count: int) -> list[dict[str, Any]]:
    if keep_count <= 0:
        return []
    ranked_rows = sorted(
        [row for row in rows if isinstance(row, dict)],
        key=_row_priority,
        reverse=True,
    )
    return ranked_rows[:keep_count]


def _generate_teacher_dataset(*, config: dict, config_dir: Path, output_dir: Path) -> Path:
    teacher_cfg = config["teacher_generation"]
    container_url = str(
        os.getenv("NANOHORIZON_CRAFTER_CONTAINER_URL")
        or teacher_cfg.get("container_url")
        or os.getenv("NANOHORIZON_CONTAINER_URL")
        or ""
    ).strip()
    container_worker_token = str(
        os.getenv("NANOHORIZON_CRAFTER_CONTAINER_WORKER_TOKEN")
        or teacher_cfg.get("container_worker_token")
        or ""
    ).strip()
    if not container_url:
        raise RuntimeError("teacher_generation.container_url or NANOHORIZON_CONTAINER_URL is required")

    dataset_path = output_dir / "generated_sft_data.jsonl"
    rollouts_path = output_dir / "teacher_rollouts.jsonl"
    progress_path = output_dir / "teacher_generation_progress.json"
    rollout_summary_path = output_dir / "teacher_rollout_summary.json"
    dataset_path.write_text("", encoding="utf-8")
    rollouts_path.write_text("", encoding="utf-8")

    thinking_budget_tokens = int(teacher_cfg.get("thinking_budget_tokens", 2000))
    seed_prompts = read_jsonl(resolve_path(teacher_cfg["seed_prompts_jsonl"], base_dir=config_dir))
    if not seed_prompts:
        raise RuntimeError("teacher_generation.seed_prompts_jsonl yielded no seed prompts")
    seed_start = int(teacher_cfg.get("seed_start", 0))
    samples_per_seed = max(
        1,
        int(os.getenv("NANOHORIZON_TEACHER_SAMPLES_PER_SEED", teacher_cfg.get("samples_per_seed", 1))),
    )
    rollout_concurrency = max(
        1,
        int(
            os.getenv(
                "NANOHORIZON_TEACHER_ROLLOUT_CONCURRENCY",
                teacher_cfg.get(
                    "rollout_concurrency",
                    teacher_cfg.get("max_concurrent_rollouts", 8),
                ),
            )
        ),
    )
    rollout_semaphore_limit = max(
        1,
        int(
            os.getenv(
                "NANOHORIZON_TEACHER_ROLLOUT_SEMAPHORE_LIMIT",
                teacher_cfg.get("rollout_semaphore_limit", rollout_concurrency),
            )
        ),
    )
    max_steps = max(
        1,
        int(os.getenv("NANOHORIZON_TEACHER_MAX_STEPS", teacher_cfg.get("max_steps", 48))),
    )
    request_timeout_seconds = float(
        os.getenv(
            "NANOHORIZON_TEACHER_REQUEST_TIMEOUT_SECONDS",
            teacher_cfg.get("request_timeout_seconds", 90.0),
        )
    )
    max_generated_rows = int(os.getenv("NANOHORIZON_MAX_TEACHER_ROWS", "0"))
    max_tokens = max(int(teacher_cfg.get("max_tokens", 180)), 64)
    rollout_quantile = float(teacher_cfg.get("reward_quantile", 0.75))
    min_reward = float(os.getenv("NANOHORIZON_MIN_TEACHER_REWARD", "0.0"))

    seeds: list[int] = []
    for index in range(len(seed_prompts)):
        for sample_idx in range(samples_per_seed):
            seeds.append(seed_start + index * samples_per_seed + sample_idx)
    if max_generated_rows > 0:
        seeds = seeds[:max_generated_rows]

    rollouts, rollout_collection_summary = asyncio.run(
        collect_rollouts_concurrently_with_summary(
            container_url=container_url,
            container_worker_token=container_worker_token,
            inference_url=_normalize_inference_url(
                os.getenv("NANOHORIZON_TEACHER_INFERENCE_URL")
                or os.getenv("NANOHORIZON_TEACHER_BASE_URL")
                or teacher_cfg.get("teacher_base_url")
                or ""
            ),
            model=str(teacher_cfg["teacher_model"]),
            api_key=str(os.getenv("NANOHORIZON_TEACHER_API_KEY") or ""),
            seeds=seeds,
            max_steps=max_steps,
            system_prompt=_rollout_system_prompt(thinking_budget_tokens=thinking_budget_tokens),
            temperature=float(teacher_cfg.get("temperature", 0.2)),
            max_tokens=max_tokens,
            enable_thinking=bool(teacher_cfg.get("enable_thinking", True)),
            thinking_budget_tokens=thinking_budget_tokens,
            policy_version="teacher-rollout",
            target_action_batch_size=int(teacher_cfg.get("target_action_batch_size", 4)),
            min_action_batch_size=int(teacher_cfg.get("min_action_batch_size", 3)),
            request_timeout_seconds=request_timeout_seconds,
            max_concurrent_rollouts=rollout_concurrency,
            trace_prefix="nanohorizon_teacher",
            rollout_concurrency=rollout_concurrency,
            rollout_semaphore_limit=rollout_semaphore_limit,
        )
    )
    with rollouts_path.open("w", encoding="utf-8") as handle:
        for rollout in rollouts:
            handle.write(json.dumps(rollout, sort_keys=True) + "\n")

    successful_rollouts = [rollout for rollout in rollouts if isinstance(rollout, dict) and not rollout.get("error")]
    rewards = [rollout_outcome_reward(rollout) for rollout in successful_rollouts]
    threshold = reward_quantile_threshold(
        rewards,
        quantile=rollout_quantile,
        minimum_threshold=min_reward,
    )
    openai_rows = build_openai_sft_rows_from_rollouts(
        successful_rollouts,
        reward_threshold=threshold,
    )
    ranked_rows = _filter_rows_by_priority(openai_rows, keep_count=len(openai_rows))
    with dataset_path.open("w", encoding="utf-8") as handle:
        for row in ranked_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    rollout_summary = {
        **rollout_collection_summary,
        "reward_threshold": threshold,
        "accepted_rows": len(ranked_rows),
        "successful_rollout_summary": summarize_rollouts(successful_rollouts),
        "mean_llm_calls_per_rollout": (
            mean([rollout_llm_call_count(rollout) for rollout in successful_rollouts])
            if successful_rollouts
            else 0.0
        ),
    }
    write_json(rollout_summary_path, rollout_summary)
    write_json(
        progress_path,
        {
            "completed_candidates": len(rollouts),
            "total_candidates": len(seeds),
            "accepted_rows": len(ranked_rows),
            "reward_threshold": threshold,
            "rollout_summary": summarize_rollouts(successful_rollouts),
            "rollout_collection": rollout_collection_summary,
            "mean_outcome_reward": mean(rewards) if rewards else 0.0,
            "max_outcome_reward": max(rewards) if rewards else 0.0,
            "mean_llm_calls_per_rollout": (
                mean([rollout_llm_call_count(rollout) for rollout in successful_rollouts])
                if successful_rollouts
                else 0.0
            ),
        },
    )
    print(
        json.dumps(
            {
                "stage": "teacher_generation_complete",
                "successful_rollouts": len(successful_rollouts),
                "generated_rows": len(ranked_rows),
                "reward_threshold": threshold,
                "mean_outcome_reward": mean(rewards) if rewards else 0.0,
                "max_outcome_reward": max(rewards) if rewards else 0.0,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return dataset_path


def _filter_teacher_dataset(
    *, dataset_path: Path, output_dir: Path, keep_top_fraction: float
) -> tuple[Path, dict[str, Any]]:
    rows = read_jsonl(dataset_path)
    max_keep = int(os.getenv("NANOHORIZON_MAX_TEACHER_ROWS", "0"))
    keep_count = max(1, int(len(rows) * keep_top_fraction)) if rows else 0
    if max_keep > 0:
        keep_count = min(keep_count, max_keep)
    kept_rows = _filter_rows_by_priority(rows, keep_count=keep_count)
    filtered_path = output_dir / "filtered_sft_data.jsonl"
    with filtered_path.open("w", encoding="utf-8") as handle:
        for row in kept_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    summary = {
        "input_rows": len(rows),
        "kept_rows": len(kept_rows),
        "keep_top_fraction": keep_top_fraction,
        "max_keep": max_keep,
    }
    print(json.dumps({"stage": "filter_complete", **summary}, sort_keys=True), flush=True)
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


def _train_examples(
    *,
    config: dict[str, Any],
    examples: list[dict[str, Any]],
    output_dir: Path,
) -> TrainingExecution:
    training_cfg = config["training"]
    base_model = str(config["model"]["model"])
    use_modal_training = bool(int(os.getenv("NANOHORIZON_TRAIN_ON_MODAL", "0") or "0"))
    if not use_modal_training:
        result = train_sft_with_trl(
            base_model=base_model,
            examples=examples,
            output_dir=output_dir / "adapter",
            learning_rate=float(training_cfg["learning_rate"]),
            epochs=int(training_cfg["epochs"]),
            max_length=int(training_cfg["max_length"]),
            max_steps=int(training_cfg["max_steps"]),
            lora_rank=int(training_cfg["lora_rank"]),
            per_device_train_batch_size=int(training_cfg.get("per_device_train_batch_size", 1)),
            gradient_accumulation_steps=int(training_cfg.get("gradient_accumulation_steps", 1)),
        )
        return TrainingExecution(
            output_dir=str(output_dir),
            adapter_dir=str(output_dir / "adapter"),
            examples_seen=int(result.examples_seen),
            optimizer_steps=int(result.optimizer_steps),
            mean_loss=float(result.mean_loss),
            backend="local",
        )

    import modal

    app_name = os.getenv("NANOHORIZON_MODAL_SFT_APP_NAME", "nanohorizon-crafter-sft")
    function = modal.Function.from_name(app_name, "train_sft")
    remote_output_dir = str(
        os.getenv("NANOHORIZON_MODAL_SFT_OUTPUT_DIR")
        or os.getenv("NANOHORIZON_MODAL_SFT_OUTPUT_ROOT")
        or ""
    ).strip()
    payload = function.remote(
        base_model=base_model,
        examples=examples,
        output_dir=remote_output_dir,
        learning_rate=float(training_cfg["learning_rate"]),
        epochs=int(training_cfg["epochs"]),
        max_length=int(training_cfg["max_length"]),
        max_steps=int(training_cfg["max_steps"]),
        lora_rank=int(training_cfg["lora_rank"]),
        per_device_train_batch_size=int(training_cfg.get("per_device_train_batch_size", 1)),
        gradient_accumulation_steps=int(training_cfg.get("gradient_accumulation_steps", 1)),
    )
    if not isinstance(payload, dict):
        raise RuntimeError("modal SFT returned a non-dict payload")
    write_json(output_dir / "modal_train_result.json", payload)
    return TrainingExecution(
        output_dir=str(payload.get("output_dir") or ""),
        adapter_dir=str(payload.get("adapter_dir") or ""),
        examples_seen=int(payload.get("examples_seen") or 0),
        optimizer_steps=int(payload.get("optimizer_steps") or 0),
        mean_loss=float(payload.get("mean_loss") or 0.0),
        backend="modal",
    )


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
            config=config,
            config_dir=config_dir,
            output_dir=output_dir,
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
    print(
        json.dumps(
            {
                "stage": "sft_examples_ready",
                "dataset_path": str(dataset_path),
                "num_examples": len(examples),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    training_result = _train_examples(
        config=config,
        examples=examples,
        output_dir=output_dir,
    )

    evaluation_cfg = config.get("evaluation", {})
    evaluation_enabled = bool(evaluation_cfg.get("enabled", True))
    eval_base_model = str(config["model"]["model"])
    eval_container_url = str(
        os.getenv("NANOHORIZON_CRAFTER_CONTAINER_URL")
        or evaluation_cfg.get("container_url")
        or os.getenv("NANOHORIZON_CONTAINER_URL")
        or config.get("teacher_generation", {}).get("container_url")
        or ""
    )
    eval_seed_start = int(evaluation_cfg.get("seed_start", 10_000))
    eval_num_rollouts = int(evaluation_cfg.get("num_rollouts", 8))
    eval_max_steps = int(evaluation_cfg.get("max_steps", 48))
    eval_max_concurrent_rollouts = int(evaluation_cfg.get("max_concurrent_rollouts", 8))
    eval_max_length = int(evaluation_cfg.get("max_length", 4096))
    eval_max_new_tokens = int(evaluation_cfg.get("max_new_tokens", 64))
    eval_enable_thinking = bool(evaluation_cfg.get("enable_thinking", False))
    eval_thinking_budget_tokens = int(evaluation_cfg.get("thinking_budget_tokens", 0))
    adapter_dir: Path | None
    if training_result.backend == "modal":
        adapter_dir = None
        if evaluation_enabled:
            raise RuntimeError(
                "internal evaluation is not supported for modal-trained adapters; "
                "run the final compare through scripts/run_offline_training.sh"
            )
    else:
        adapter_dir = output_dir / "adapter"
    if evaluation_enabled:
        finetuned_eval_summary = evaluate_model(
            base_model=eval_base_model,
            adapter_dir=adapter_dir,
            output_dir=output_dir,
            container_url=eval_container_url,
            seed_start=eval_seed_start,
            num_rollouts=eval_num_rollouts,
            max_steps=eval_max_steps,
            max_concurrent_rollouts=eval_max_concurrent_rollouts,
            max_length=eval_max_length,
            max_new_tokens=eval_max_new_tokens,
            enable_thinking=eval_enable_thinking,
            thinking_budget_tokens=eval_thinking_budget_tokens,
            enforce_eager=True,
            summary_name="finetuned_eval_summary.json",
        )
        base_eval_summary = evaluate_model(
            base_model=eval_base_model,
            adapter_dir=None,
            output_dir=output_dir,
            container_url=eval_container_url,
            seed_start=eval_seed_start,
            num_rollouts=eval_num_rollouts,
            max_steps=eval_max_steps,
            max_concurrent_rollouts=eval_max_concurrent_rollouts,
            max_length=eval_max_length,
            max_new_tokens=eval_max_new_tokens,
            enable_thinking=eval_enable_thinking,
            thinking_budget_tokens=eval_thinking_budget_tokens,
            enforce_eager=True,
            summary_name="base_eval_summary.json",
        )
        comparison_summary = {
            "base_mean_outcome_reward": base_eval_summary["mean_outcome_reward"],
            "finetuned_mean_outcome_reward": finetuned_eval_summary["mean_outcome_reward"],
            "reward_delta": finetuned_eval_summary["mean_outcome_reward"] - base_eval_summary["mean_outcome_reward"],
            "base_num_eval_rollouts": base_eval_summary["num_eval_rollouts"],
            "finetuned_num_eval_rollouts": finetuned_eval_summary["num_eval_rollouts"],
        }
    else:
        base_eval_summary = {
            "num_eval_rollouts": 0,
            "mean_outcome_reward": 0.0,
            "skipped": True,
        }
        finetuned_eval_summary = {
            "num_eval_rollouts": 0,
            "mean_outcome_reward": 0.0,
            "skipped": True,
        }
        comparison_summary = {
            "base_mean_outcome_reward": 0.0,
            "finetuned_mean_outcome_reward": 0.0,
            "reward_delta": 0.0,
            "base_num_eval_rollouts": 0,
            "finetuned_num_eval_rollouts": 0,
            "skipped": True,
        }
    write_json(output_dir / "comparison_summary.json", comparison_summary)

    metrics = {
        "track": config.get("task", {}).get("track", "offline_20min_1xa100_40gb"),
        "baseline": "filtered_behavior_cloning",
        "started_at": timer.started_at,
        "ended_at": timer.ended_at,
        "examples_seen": training_result.examples_seen,
        "optimizer_steps": training_result.optimizer_steps,
        "mean_loss": training_result.mean_loss,
        "elapsed_minutes": timer.elapsed_minutes,
        "dataset_jsonl": str(dataset_path),
        "teacher_generation_enabled": teacher_enabled,
        "teacher_model": teacher_generation.get("teacher_model", ""),
        "training_backend": training_result.backend,
        "adapter_dir": training_result.adapter_dir,
        "training_output_dir": training_result.output_dir,
        "finetuned_num_eval_rollouts": finetuned_eval_summary["num_eval_rollouts"],
        "finetuned_mean_outcome_reward": finetuned_eval_summary["mean_outcome_reward"],
        "base_num_eval_rollouts": base_eval_summary["num_eval_rollouts"],
        "base_mean_outcome_reward": base_eval_summary["mean_outcome_reward"],
        "reward_delta": comparison_summary["reward_delta"],
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
        f"python -m nanohorizon.offline_training --config {Path(args.config).resolve()}\n",
    )


if __name__ == "__main__":
    main()
