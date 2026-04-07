from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from nanohorizon.baselines.cpt import ensure_dir, resolve_path, write_json, write_text
from nanohorizon.shared.craftax_data import (
    collect_rollouts_concurrently_with_summary,
    flatten_messages,
    rollout_achievements,
    rollout_llm_call_count,
    rollout_outcome_reward,
    rollout_turns,
    summarize_rollouts,
)


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    text = config_path.read_text(encoding="utf-8")
    payload = json.loads(text) if config_path.suffix.lower() == ".json" else yaml.safe_load(text)
    if not isinstance(payload, dict):
        raise ValueError(f"config must decode to an object: {config_path}")
    return payload


def read_jsonl(path: str | Path) -> list[Any]:
    rows: list[Any] = []
    for raw in Path(path).expanduser().resolve().read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def _load_tokenizer(model_name: str) -> Any:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


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


def _rollout_system_prompt(
    *,
    thinking_budget_tokens: int,
    target_action_batch_size: int,
    min_action_batch_size: int,
) -> str:
    if target_action_batch_size == min_action_batch_size:
        action_instruction = f"Return exactly {target_action_batch_size} valid full-Craftax actions."
    else:
        action_instruction = (
            f"Return a short useful macro-action with {min_action_batch_size}-{target_action_batch_size} "
            "valid full-Craftax actions."
        )
    return (
        "You are a Craftax teacher policy collecting high-signal long-horizon trajectories.\n"
        f"You may think for up to about {thinking_budget_tokens} tokens before answering.\n"
        f"{action_instruction}\n"
        "Use movement to explore when nothing useful is adjacent.\n"
        "Use 'do' only when facing a useful nearby object or resource.\n"
        "Read the recent action history and avoid repeating unproductive loops.\n"
        "Use the provided `craftax_interact` tool exactly once for the final answer.\n"
        "Do not return plain text actions or JSON.\n"
        "Your final assistant action must be a tool call with valid full-Craftax actions."
    )


def build_seed_schedule(
    *,
    base_seeds: list[int],
    requested_rollouts: int,
    seed_offset_stride: int = 1000,
) -> list[int]:
    if requested_rollouts <= 0:
        return []
    if not base_seeds:
        raise ValueError("base_seeds must not be empty")
    schedule: list[int] = []
    for index in range(requested_rollouts):
        base_seed = int(base_seeds[index % len(base_seeds)])
        round_index = index // len(base_seeds)
        schedule.append(base_seed + (round_index * int(seed_offset_stride)))
    return schedule


def rollout_to_cpt_text(
    rollout: dict[str, Any],
    *,
    include_reasoning: bool = True,
    include_action_history: bool = False,
) -> str:
    trace_id = str(rollout.get("trace_correlation_id") or rollout.get("rollout_id") or "").strip()
    metadata = rollout.get("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
    seed = metadata.get("seed")
    achievements = rollout_achievements(rollout)
    lines = [
        "### NanoHorizon Craftax Rollout",
        f"Trace ID: {trace_id or 'unknown'}",
        f"Seed: {seed if seed is not None else 'unknown'}",
        f"Outcome reward: {rollout_outcome_reward(rollout):.2f}",
        f"LLM calls: {rollout_llm_call_count(rollout)}",
        f"Achievements: {', '.join(achievements) if achievements else 'none'}",
        "",
    ]
    for turn in rollout_turns(rollout):
        prompt_messages = turn.get("prompt_messages")
        prompt_text = flatten_messages([item for item in prompt_messages if isinstance(item, dict)]) if isinstance(prompt_messages, list) else ""
        reasoning_text = str(turn.get("reasoning_text") or "").strip()
        assistant_text = str(turn.get("assistant_text") or "").strip()
        actions = turn.get("actions")
        safe_actions = [str(item).strip() for item in actions if str(item).strip()] if isinstance(actions, list) else []
        lines.extend(
            [
                f"## Turn {int(turn.get('turn_index') or 0)}",
                "Prompt:",
                prompt_text or "(empty)",
            ]
        )
        if include_reasoning and reasoning_text:
            lines.extend(["Reasoning:", reasoning_text])
        elif assistant_text:
            lines.extend(["Assistant:", assistant_text])
        if safe_actions:
            lines.append(f"Chosen actions: {', '.join(safe_actions)}")
        lines.append(f"Decision reward: {float(turn.get('decision_reward') or 0.0):.2f}")
        lines.append(f"Return to go: {float(turn.get('return_to_go') or 0.0):.2f}")
        lines.append("")
    if include_action_history:
        history = metadata.get("action_history")
        if isinstance(history, list) and history:
            rendered_history = ", ".join(str(item).strip() for item in history if str(item).strip())
            if rendered_history:
                lines.extend(["Final action history:", rendered_history, ""])
    return "\n".join(lines).strip()


def project_rollout_to_text_row(
    rollout: dict[str, Any],
    *,
    include_reasoning: bool = True,
    include_action_history: bool = False,
) -> dict[str, Any]:
    metadata = rollout.get("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
    return {
        "text": rollout_to_cpt_text(
            rollout,
            include_reasoning=include_reasoning,
            include_action_history=include_action_history,
        ),
        "metadata": {
            "trace_correlation_id": str(rollout.get("trace_correlation_id") or ""),
            "rollout_id": str(rollout.get("rollout_id") or ""),
            "seed": metadata.get("seed"),
            "outcome_reward": float(rollout_outcome_reward(rollout)),
            "achievements": rollout_achievements(rollout),
            "llm_call_count": int(rollout_llm_call_count(rollout)),
            "num_turns": len(rollout_turns(rollout)),
        },
    }


@dataclass
class BudgetState:
    token_budget: int
    total_tokens: int = 0
    rows_written: int = 0
    rollouts_projected: int = 0
    truncated_rows: int = 0


def append_rows_until_token_budget(
    *,
    rows: list[dict[str, Any]],
    tokenizer: Any,
    budget_state: BudgetState,
) -> list[dict[str, Any]]:
    written: list[dict[str, Any]] = []
    for row in rows:
        text = str(row.get("text") or "").strip()
        if not text:
            continue
        remaining = budget_state.token_budget - budget_state.total_tokens
        if remaining <= 0:
            break
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        if not token_ids:
            continue
        kept_ids = token_ids[:remaining]
        kept_text = tokenizer.decode(kept_ids, skip_special_tokens=False).strip()
        if not kept_text:
            break
        truncated = len(kept_ids) < len(token_ids)
        metadata = row.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}
        enriched_row = {
            **row,
            "text": kept_text,
            "metadata": {
                **metadata,
                "source_token_count": len(token_ids),
                "kept_token_count": len(kept_ids),
                "truncated": truncated,
            },
        }
        written.append(enriched_row)
        budget_state.total_tokens += len(kept_ids)
        budget_state.rows_written += 1
        if truncated:
            budget_state.truncated_rows += 1
            break
    return written


def restore_rows_with_budget(
    *,
    rows: list[dict[str, Any]],
    tokenizer: Any,
    token_budget: int,
) -> tuple[list[dict[str, Any]], BudgetState]:
    budget_state = BudgetState(token_budget=token_budget)
    restored_rows = append_rows_until_token_budget(
        rows=rows,
        tokenizer=tokenizer,
        budget_state=budget_state,
    )
    return restored_rows, budget_state


def maybe_upload_rows_to_hub(
    *,
    rows: list[dict[str, Any]],
    hf_cfg: dict[str, Any],
) -> dict[str, Any]:
    if not bool(hf_cfg.get("enabled", False)):
        return {"enabled": False, "skipped": True}
    repo_id = str(hf_cfg.get("repo_id") or "").strip()
    if not repo_id:
        raise ValueError("hf_upload.repo_id is required when hf_upload.enabled=true")
    split = str(hf_cfg.get("split", "train") or "train")
    config_name = str(hf_cfg.get("config_name", "default") or "default")
    private = bool(hf_cfg.get("private", True))
    commit_message = str(hf_cfg.get("commit_message") or "Add NanoHorizon CPT shard").strip()

    try:
        from datasets import Dataset
    except Exception as exc:
        raise RuntimeError("datasets is required for Hugging Face upload support") from exc

    dataset = Dataset.from_list(rows)
    dataset.push_to_hub(
        repo_id,
        config_name=config_name,
        split=split,
        private=private,
        commit_message=commit_message,
    )
    return {
        "enabled": True,
        "repo_id": repo_id,
        "split": split,
        "config_name": config_name,
        "rows_uploaded": len(rows),
    }


def _progress_enabled() -> bool:
    raw = str(os.getenv("NANOHORIZON_CPT_DATA_PROGRESS", "1")).strip().lower()
    return raw not in {"", "0", "false", "no", "off"}


def _emit_progress(event: dict[str, Any]) -> None:
    if not _progress_enabled():
        return
    print(json.dumps(event, sort_keys=True), file=sys.stderr, flush=True)


def _load_train_seeds(seed_manifest_path: Path) -> list[int]:
    payload = json.loads(seed_manifest_path.read_text(encoding="utf-8"))
    train_seeds = payload.get("train_seeds")
    if not isinstance(train_seeds, list) or not train_seeds:
        raise ValueError(f"seed manifest missing train_seeds: {seed_manifest_path}")
    return [int(item) for item in train_seeds]


def run_generation(*, config_path: str | Path, output_dir: str | Path = "") -> dict[str, Any]:
    config_path = Path(config_path).expanduser().resolve()
    config = load_config(config_path)
    config_dir = config_path.parent
    out_dir = ensure_dir(
        resolve_path(
            output_dir or config.get("output", {}).get("root_dir") or "artifacts/cpt_data",
            base_dir=config_dir,
        )
    )

    collection_cfg = config.get("collection", {})
    projection_cfg = config.get("projection", {})
    output_cfg = config.get("output", {})
    hf_cfg = config.get("hf_upload", {})

    inference_url = _normalize_inference_url(
        str(
            os.getenv("NANOHORIZON_TEACHER_INFERENCE_URL")
            or collection_cfg.get("inference_url")
            or ""
        )
    )
    if not inference_url:
        raise ValueError("collection.inference_url or NANOHORIZON_TEACHER_INFERENCE_URL is required")
    container_url = str(
        os.getenv("NANOHORIZON_CRAFTAX_CONTAINER_URL")
        or collection_cfg.get("container_url")
        or "direct://local"
    ).strip()
    container_worker_token = str(
        os.getenv("NANOHORIZON_CRAFTAX_CONTAINER_WORKER_TOKEN")
        or collection_cfg.get("container_worker_token")
        or ""
    ).strip()
    environment_api_key = str(collection_cfg.get("environment_api_key") or "").strip()

    teacher_model = str(
        collection_cfg.get("request_model")
        or os.getenv("NANOHORIZON_TEACHER_MODEL")
        or config.get("teacher", {}).get("model")
        or ""
    ).strip()
    if not teacher_model:
        raise ValueError("teacher.model or collection.request_model is required")
    inference_api_key = str(
        os.getenv("NANOHORIZON_TEACHER_API_KEY")
        or os.getenv("NANOHORIZON_VLLM_API_KEY")
        or collection_cfg.get("api_key")
        or "dummy-local-key"
    ).strip()

    seed_manifest_path = resolve_path(collection_cfg["seed_manifest_json"], base_dir=config_dir)
    base_seeds = _load_train_seeds(seed_manifest_path)
    requested_rollouts = int(collection_cfg.get("max_rollouts", len(base_seeds)))
    seed_schedule = build_seed_schedule(
        base_seeds=base_seeds,
        requested_rollouts=requested_rollouts,
        seed_offset_stride=int(collection_cfg.get("seed_offset_stride", 1000)),
    )

    tokenizer_model = str(
        projection_cfg.get("tokenizer_model")
        or config.get("model", {}).get("model")
        or "Qwen/Qwen3.5-0.8B"
    ).strip()
    tokenizer = _load_tokenizer(tokenizer_model)
    token_budget = int(projection_cfg.get("token_budget", 100_000))
    budget_state = BudgetState(token_budget=token_budget)
    include_reasoning = bool(projection_cfg.get("include_reasoning", True))
    include_action_history = bool(projection_cfg.get("include_action_history", False))
    min_outcome_reward = float(collection_cfg.get("min_outcome_reward", 0.0))
    resume_existing = bool(output_cfg.get("resume", True))

    rollouts_path = out_dir / "rollouts.jsonl"
    text_rows_path = out_dir / "cpt_rollouts_text.jsonl"
    existing_rollouts = (
        [row for row in read_jsonl(rollouts_path) if isinstance(row, dict)]
        if resume_existing and rollouts_path.exists()
        else []
    )
    existing_rows = (
        [row for row in read_jsonl(text_rows_path) if isinstance(row, dict)]
        if resume_existing and text_rows_path.exists()
        else []
    )
    if resume_existing and existing_rows:
        restored_rows, budget_state = restore_rows_with_budget(
            rows=existing_rows,
            tokenizer=tokenizer,
            token_budget=token_budget,
        )
        if restored_rows != existing_rows:
            with text_rows_path.open("w", encoding="utf-8") as handle:
                for row in restored_rows:
                    handle.write(json.dumps(row, sort_keys=True) + "\n")
        existing_rows = restored_rows
    else:
        rollouts_path.write_text("", encoding="utf-8")
        text_rows_path.write_text("", encoding="utf-8")

    batch_size = max(1, int(collection_cfg.get("rollout_batch_size", min(8, len(base_seeds) or 1))))
    all_rollouts: list[dict[str, Any]] = list(existing_rollouts)
    all_rows: list[dict[str, Any]] = list(existing_rows)
    batch_summaries: list[dict[str, Any]] = []
    if existing_rollouts:
        budget_state.rollouts_projected = len(existing_rows)
    remaining_seed_schedule = seed_schedule[len(existing_rollouts) :]

    _emit_progress(
        {
            "stage": "cpt_data_start",
            "output_dir": str(out_dir),
            "teacher_model": teacher_model,
            "container_url": container_url,
            "inference_url": inference_url,
            "requested_rollouts": len(seed_schedule),
            "resume_existing": resume_existing,
            "existing_rollouts": len(existing_rollouts),
            "existing_rows": len(existing_rows),
            "starting_tokens": budget_state.total_tokens,
            "token_budget": budget_state.token_budget,
            "batch_size": batch_size,
        }
    )

    def _summary(upload_summary: dict[str, Any] | None = None) -> dict[str, Any]:
        return {
            "teacher_model": teacher_model,
            "inference_url": inference_url,
            "container_url": container_url,
            "requested_rollouts": len(seed_schedule),
            "collected_rollouts": len(all_rollouts),
            "projected_rollouts": budget_state.rollouts_projected,
            "rows_written": budget_state.rows_written,
            "total_tokens": budget_state.total_tokens,
            "token_budget": budget_state.token_budget,
            "truncated_rows": budget_state.truncated_rows,
            "rollout_summary": summarize_rollouts(all_rollouts),
            "batch_summaries": batch_summaries,
            "text_rows_path": str(text_rows_path),
            "rollouts_path": str(rollouts_path),
            "canonical_dataset_path": "",
            "hf_upload": upload_summary or {"enabled": False, "skipped": True},
        }

    if budget_state.total_tokens >= budget_state.token_budget:
        write_json(out_dir / "summary.json", _summary())

    for start_index in range(0, len(remaining_seed_schedule), batch_size):
        if budget_state.total_tokens >= budget_state.token_budget:
            break
        batch_seeds = remaining_seed_schedule[start_index : start_index + batch_size]
        batch_index = (len(existing_rollouts) + start_index) // batch_size + 1
        system_prompt = _rollout_system_prompt(
            thinking_budget_tokens=int(collection_cfg.get("thinking_budget_tokens", 2000)),
            target_action_batch_size=int(collection_cfg.get("target_action_batch_size", 4)),
            min_action_batch_size=int(collection_cfg.get("min_action_batch_size", 3)),
        )
        rollouts, rollout_summary = asyncio.run(
            collect_rollouts_concurrently_with_summary(
                container_url=container_url,
                container_worker_token=container_worker_token,
                environment_api_key=environment_api_key,
                inference_url=inference_url,
                model=teacher_model,
                api_key=inference_api_key,
                seeds=batch_seeds,
                max_steps=int(collection_cfg.get("max_steps", 48)),
                system_prompt=system_prompt,
                temperature=float(collection_cfg.get("temperature", 0.4)),
                max_tokens=int(collection_cfg.get("max_tokens", 3072)),
                enable_thinking=bool(collection_cfg.get("enable_thinking", True)),
                thinking_budget_tokens=int(collection_cfg.get("thinking_budget_tokens", 2000)),
                policy_version=str(collection_cfg.get("policy_version", "cpt_teacher")),
                target_action_batch_size=int(collection_cfg.get("target_action_batch_size", 4)),
                min_action_batch_size=int(collection_cfg.get("min_action_batch_size", 3)),
                request_timeout_seconds=float(collection_cfg.get("request_timeout_seconds", 300.0)),
                max_concurrent_rollouts=int(collection_cfg.get("rollout_concurrency", 4)),
                trace_prefix=str(collection_cfg.get("trace_prefix", "cpt_collect")),
                rollout_concurrency=int(collection_cfg.get("rollout_concurrency", 4)),
                rollout_semaphore_limit=int(collection_cfg.get("rollout_semaphore_limit", 4)),
                request_logprobs=bool(collection_cfg.get("request_logprobs", True)),
                progress_callback=lambda event, batch_index=batch_index, batch_seeds=list(batch_seeds): _emit_progress(
                    {
                        "stage": "teacher_rollout_progress",
                        "batch_index": batch_index,
                        "batch_seeds": batch_seeds,
                        **event,
                    }
                ),
            )
        )
        batch_summaries.append(rollout_summary)
        all_rollouts.extend(rollouts)
        with rollouts_path.open("a", encoding="utf-8") as handle:
            for rollout in rollouts:
                handle.write(json.dumps(rollout, sort_keys=True) + "\n")

        projected_rows = [
            project_rollout_to_text_row(
                rollout,
                include_reasoning=include_reasoning,
                include_action_history=include_action_history,
            )
            for rollout in rollouts
            if rollout_outcome_reward(rollout) >= min_outcome_reward and not rollout.get("error")
        ]
        budgeted_rows = append_rows_until_token_budget(
            rows=projected_rows,
            tokenizer=tokenizer,
            budget_state=budget_state,
        )
        budget_state.rollouts_projected += len(projected_rows)
        all_rows.extend(budgeted_rows)
        with text_rows_path.open("a", encoding="utf-8") as handle:
            for row in budgeted_rows:
                handle.write(json.dumps(row, sort_keys=True) + "\n")

        _emit_progress(
            {
                "stage": "cpt_data_batch_complete",
                "batch_index": batch_index,
                "batch_seeds": batch_seeds,
                "collected_rollouts": len(all_rollouts),
                "projected_rollouts": budget_state.rollouts_projected,
                "rows_written": budget_state.rows_written,
                "total_tokens": budget_state.total_tokens,
                "token_budget": budget_state.token_budget,
                "truncated_rows": budget_state.truncated_rows,
                "rollouts_path": str(rollouts_path),
                "text_rows_path": str(text_rows_path),
                "batch_summary": rollout_summary,
            }
        )
        write_json(out_dir / "summary.json", _summary())

    canonical_dataset_path_str = str(output_cfg.get("dataset_jsonl") or "").strip()
    canonical_dataset_path = (
        resolve_path(canonical_dataset_path_str, base_dir=config_dir) if canonical_dataset_path_str else None
    )
    if canonical_dataset_path is not None:
        canonical_dataset_path.parent.mkdir(parents=True, exist_ok=True)
        canonical_dataset_path.write_text(text_rows_path.read_text(encoding="utf-8"), encoding="utf-8")

    upload_summary = maybe_upload_rows_to_hub(rows=all_rows, hf_cfg=hf_cfg)

    summary = _summary(upload_summary=upload_summary)
    summary["canonical_dataset_path"] = str(canonical_dataset_path) if canonical_dataset_path is not None else ""
    write_json(out_dir / "summary.json", summary)
    write_text(
        out_dir / "command.txt",
        f"python -m nanohorizon.baselines.cpt_data run --config {config_path}\n",
    )
    _emit_progress({"stage": "cpt_data_done", **summary})
    return summary


def project_rollouts_file(
    *,
    input_rollouts: str | Path,
    output_jsonl: str | Path,
    tokenizer_model: str,
    token_budget: int,
    include_reasoning: bool = True,
    include_action_history: bool = False,
) -> dict[str, Any]:
    tokenizer = _load_tokenizer(tokenizer_model)
    rows = [
        project_rollout_to_text_row(
            row,
            include_reasoning=include_reasoning,
            include_action_history=include_action_history,
        )
        for row in read_jsonl(input_rollouts)
        if isinstance(row, dict) and not row.get("error")
    ]
    budget_state = BudgetState(token_budget=token_budget)
    budgeted_rows = append_rows_until_token_budget(
        rows=rows,
        tokenizer=tokenizer,
        budget_state=budget_state,
    )
    destination = Path(output_jsonl).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for row in budgeted_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    summary = {
        "rows_written": budget_state.rows_written,
        "total_tokens": budget_state.total_tokens,
        "token_budget": budget_state.token_budget,
        "truncated_rows": budget_state.truncated_rows,
        "output_jsonl": str(destination),
    }
    write_json(destination.with_suffix(".summary.json"), summary)
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NanoHorizon CPT rollout data generation utilities.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Collect Craftax rollouts and project them into CPT text.")
    run_parser.add_argument("--config", required=True, help="Path to the rollout generation config.")
    run_parser.add_argument("--output-dir", default="", help="Optional override for output root.")

    project_parser = subparsers.add_parser("project-rollouts", help="Project an existing rollouts.jsonl into CPT text.")
    project_parser.add_argument("--input-rollouts", required=True)
    project_parser.add_argument("--output-jsonl", required=True)
    project_parser.add_argument("--tokenizer-model", required=True)
    project_parser.add_argument("--token-budget", type=int, default=100_000)
    project_parser.add_argument("--exclude-reasoning", action="store_true")
    project_parser.add_argument("--include-action-history", action="store_true")

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.command == "run":
        run_generation(config_path=args.config, output_dir=args.output_dir)
        return
    if args.command == "project-rollouts":
        project_rollouts_file(
            input_rollouts=args.input_rollouts,
            output_jsonl=args.output_jsonl,
            tokenizer_model=args.tokenizer_model,
            token_budget=int(args.token_budget),
            include_reasoning=not bool(args.exclude_reasoning),
            include_action_history=bool(args.include_action_history),
        )
        return
    raise ValueError(f"unknown command: {args.command}")


if __name__ == "__main__":
    main()
