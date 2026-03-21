from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, cast

import httpx

from nanohorizon.common import (
    Timer,
    ensure_dir,
    load_config,
    resolve_path,
    system_info,
    write_json,
    write_text,
)
from nanohorizon.crafter_data import (
    CRAFTER_INTERACT_TOOL,
    collect_rollouts_concurrently_with_summary,
    is_rollout_payload,
    rollout_outcome_reward,
    rollout_turns,
)
from nanohorizon.train_lora import (
    DEFAULT_TARGET_MODULES,
    _load_text_only_causal_lm,
    release_cuda_memory,
)

DEFAULT_SYSTEM_PROMPT = (
    "You are a Crafter RL policy.\n"
    "Use the provided `crafter_interact` tool exactly once for the final answer.\n"
    "Return a short useful macro-action with 3-4 valid Crafter actions.\n"
    "Use movement to explore when nothing useful is adjacent.\n"
    "Use 'do' only when facing a useful nearby object or resource.\n"
    "Read the recent action history and avoid repeating unproductive loops.\n"
    "Do not return plain text actions or JSON.\n"
    "After reasoning, your final assistant action must be a tool call."
)


@dataclass
class TurnSample:
    group_id: str
    seed: int
    rollout_id: str
    trace_correlation_id: str
    turn_index: int
    prompt_messages: list[dict[str, Any]]
    full_messages: list[dict[str, Any]]
    old_logprob: float
    advantage: float
    outcome_reward: float
    decision_reward: float
    policy_version: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NanoHorizon Crafter RLVR training")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--container-url", default=os.getenv("NANOHORIZON_RLVR_CONTAINER_URL", ""))
    parser.add_argument("--inference-url", default=os.getenv("NANOHORIZON_RLVR_INFERENCE_URL", ""))
    parser.add_argument("--inference-admin-url", default=os.getenv("NANOHORIZON_RLVR_INFERENCE_ADMIN_URL", ""))
    parser.add_argument("--inference-api-key", default=os.getenv("NANOHORIZON_RLVR_INFERENCE_API_KEY", ""))
    parser.add_argument("--request-model", default=os.getenv("NANOHORIZON_RLVR_REQUEST_MODEL", ""))
    return parser.parse_args()


def _normalize_inference_url(raw_url: str) -> str:
    value = str(raw_url or "").strip().rstrip("/")
    if not value:
        return ""
    if value.endswith("/chat/completions"):
        return value
    if value.endswith("/v1"):
        return f"{value}/chat/completions"
    return f"{value}/v1/chat/completions"


def _normalize_admin_url(raw_url: str) -> str:
    value = str(raw_url or "").strip().rstrip("/")
    if not value:
        return ""
    if value.endswith("/admin"):
        return value
    return f"{value}/admin"


def _normalize_tool_calls(tool_calls: Any) -> list[dict[str, Any]]:
    if not isinstance(tool_calls, list):
        return []
    normalized: list[dict[str, Any]] = []
    for index, item in enumerate(tool_calls):
        if not isinstance(item, dict):
            continue
        item_dict = cast(dict[str, Any], item)
        raw_function = item_dict.get("function")
        function = raw_function if isinstance(raw_function, dict) else {}
        if not isinstance(function, dict):
            continue
        name = str(function.get("name") or "").strip()
        if not name:
            continue
        arguments = function.get("arguments", {})
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except Exception:
                arguments = arguments
        normalized.append(
            {
                "id": str(item_dict.get("id") or f"call_{index}"),
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": arguments,
                },
            }
        )
    return normalized


def _normalize_messages_for_chat_template(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for item in messages:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role") or "user").strip() or "user"
        entry: dict[str, Any] = {
            "role": role,
            "content": item.get("content") if item.get("content") is not None else "",
        }
        tool_calls = _normalize_tool_calls(item.get("tool_calls"))
        if tool_calls:
            entry["tool_calls"] = tool_calls
            if role == "assistant" and not isinstance(entry["content"], str):
                entry["content"] = ""
        reasoning_content = item.get("reasoning_content")
        if reasoning_content is not None:
            entry["reasoning_content"] = str(reasoning_content)
        normalized.append(entry)
    return normalized


def _render_messages(tokenizer: Any, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> str:
    safe_messages = _normalize_messages_for_chat_template(messages)
    if hasattr(tokenizer, "apply_chat_template"):
        rendered = tokenizer.apply_chat_template(
            safe_messages,
            tools=tools or None,
            tokenize=False,
            add_generation_prompt=False,
        )
        if isinstance(rendered, str):
            return rendered
    rendered_lines: list[str] = []
    for item in safe_messages:
        rendered_lines.append(f"<|{item.get('role', 'user')}|>\n{item.get('content', '')}")
    return "\n".join(rendered_lines)


def _tokenize_messages_with_assistant_mask(
    tokenizer: Any,
    prompt_messages: list[dict[str, Any]],
    full_messages: list[dict[str, Any]],
    *,
    max_length: int,
) -> dict[str, Any]:
    import torch

    normalized_prompt_messages = _normalize_messages_for_chat_template(prompt_messages)
    normalized_full_messages = _normalize_messages_for_chat_template(full_messages)
    prompt_text = tokenizer.apply_chat_template(
        normalized_prompt_messages,
        tools=[CRAFTER_INTERACT_TOOL],
        tokenize=False,
        add_generation_prompt=True,
    )
    full_text = _render_messages(tokenizer, normalized_full_messages, [CRAFTER_INTERACT_TOOL])
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
    if len(full_ids) < len(prompt_ids):
        raise ValueError("full chat template render shorter than prompt render")
    input_ids = full_ids[:max_length]
    labels = ([-100] * len(prompt_ids) + full_ids[len(prompt_ids) :])[:max_length]
    attention_mask = [1] * len(input_ids)
    return {
        "input_ids": torch.tensor([input_ids], dtype=torch.long),
        "labels": torch.tensor([labels], dtype=torch.long),
        "attention_mask": torch.tensor([attention_mask], dtype=torch.long),
    }


def _selected_token_logprobs(logits: Any, shifted_labels: Any) -> tuple[Any, Any]:
    import torch

    mask = shifted_labels != -100
    safe_targets = torch.clamp(shifted_labels, min=0)
    selected_logits = logits.gather(dim=-1, index=safe_targets.unsqueeze(-1)).squeeze(-1)
    lse = torch.logsumexp(logits, dim=-1)
    return (selected_logits - lse) * mask, mask


def _assistant_message_from_turn(turn: dict[str, Any], *, trace_correlation_id: str, rollout_id: str) -> dict[str, Any] | None:
    raw_actions = turn.get("actions")
    if not isinstance(raw_actions, list):
        return None
    actions = [str(item).strip() for item in raw_actions if str(item).strip()]
    if not actions:
        return None
    turn_index = int(turn.get("turn_index") or 0)
    return {
        "role": "assistant",
        "content": "",
        "reasoning_content": str(turn.get("reasoning_text") or turn.get("assistant_text") or "").strip(),
        "tool_calls": [
            {
                "id": f"call_{trace_correlation_id or rollout_id}_{turn_index}",
                "type": "function",
                "function": {
                    "name": "crafter_interact",
                    "arguments": {
                        "actions_list": actions,
                    },
                },
            }
        ],
    }


def _load_seed_manifest(path: Path) -> tuple[list[int], list[int]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"seed manifest must decode to an object: {path}")
    train_seeds = [int(item) for item in payload.get("train_seeds") or []]
    eval_seeds = [int(item) for item in payload.get("eval_seeds") or []]
    if not train_seeds:
        raise ValueError(f"seed manifest missing train_seeds: {path}")
    if not eval_seeds:
        raise ValueError(f"seed manifest missing eval_seeds: {path}")
    return train_seeds, eval_seeds


def _iteration_train_seeds(*, all_train_seeds: list[int], iteration_index: int, groups_per_iteration: int) -> list[int]:
    if not all_train_seeds:
        return []
    size = max(1, int(groups_per_iteration))
    start = (iteration_index * size) % len(all_train_seeds)
    seeds: list[int] = []
    for offset in range(size):
        seeds.append(int(all_train_seeds[(start + offset) % len(all_train_seeds)]))
    return seeds


def _build_rollout_seed_schedule(*, seeds: list[int], group_size: int) -> list[int]:
    schedule: list[int] = []
    for seed in seeds:
        schedule.extend([int(seed)] * max(1, int(group_size)))
    return schedule


def _group_advantages(rewards: list[float]) -> list[float]:
    if not rewards:
        return []
    mean_reward = sum(rewards) / len(rewards)
    variance = sum((reward - mean_reward) ** 2 for reward in rewards) / len(rewards)
    std = math.sqrt(max(variance, 1e-12))
    if std <= 1e-6:
        return [float(reward - mean_reward) for reward in rewards]
    return [float((reward - mean_reward) / std) for reward in rewards]


def _build_turn_samples(*, rollouts: list[dict[str, Any]], group_size: int) -> tuple[list[TurnSample], dict[str, Any]]:
    samples: list[TurnSample] = []
    valid_rollouts = [
        rollout
        for rollout in rollouts
        if isinstance(rollout, dict) and not rollout.get("error") and is_rollout_payload(rollout)
    ]
    grouped_rewards: list[float] = []
    group_count = 0
    for group_start in range(0, len(rollouts), max(1, int(group_size))):
        group_rollouts = rollouts[group_start : group_start + max(1, int(group_size))]
        if len(group_rollouts) < max(1, int(group_size)):
            continue
        valid_group = [
            rollout
            for rollout in group_rollouts
            if isinstance(rollout, dict) and not rollout.get("error") and is_rollout_payload(rollout)
        ]
        if len(valid_group) != max(1, int(group_size)):
            continue
        rewards = [float(rollout_outcome_reward(rollout)) for rollout in valid_group]
        advantages = _group_advantages(rewards)
        group_count += 1
        grouped_rewards.extend(rewards)
        for rollout, advantage, reward in zip(valid_group, advantages, rewards, strict=False):
            trace_correlation_id = str(rollout.get("trace_correlation_id") or "")
            rollout_id = str(rollout.get("rollout_id") or "")
            policy_version = str(rollout.get("policy_version") or "bootstrap")
            seed = int(rollout.get("_request_seed") or 0)
            for turn in rollout_turns(rollout):
                if not bool(turn.get("trainable", True)):
                    continue
                old_logprob_raw = turn.get("behavior_sequence_logprob")
                if old_logprob_raw is None:
                    continue
                try:
                    old_logprob = float(old_logprob_raw)
                except (TypeError, ValueError):
                    continue
                prompt_messages = turn.get("prompt_messages")
                if not isinstance(prompt_messages, list) or not prompt_messages:
                    continue
                assistant_message = _assistant_message_from_turn(
                    turn,
                    trace_correlation_id=trace_correlation_id,
                    rollout_id=rollout_id,
                )
                if assistant_message is None:
                    continue
                full_messages = [*prompt_messages, assistant_message]
                samples.append(
                    TurnSample(
                        group_id=f"group_{group_count:05d}",
                        seed=seed,
                        rollout_id=rollout_id,
                        trace_correlation_id=trace_correlation_id,
                        turn_index=int(turn.get("turn_index") or 0),
                        prompt_messages=[item for item in prompt_messages if isinstance(item, dict)],
                        full_messages=[item for item in full_messages if isinstance(item, dict)],
                        old_logprob=old_logprob,
                        advantage=float(advantage),
                        outcome_reward=float(reward),
                        decision_reward=float(turn.get("decision_reward") or 0.0),
                        policy_version=policy_version,
                    )
                )
    summary = {
        "valid_rollouts": len(valid_rollouts),
        "group_count": group_count,
        "trainable_turns": len(samples),
        "mean_outcome_reward": mean(grouped_rewards) if grouped_rewards else 0.0,
        "max_outcome_reward": max(grouped_rewards) if grouped_rewards else 0.0,
    }
    return samples, summary


def _evaluate_policy(
    *,
    container_url: str,
    inference_url: str,
    inference_api_key: str,
    request_model: str,
    seeds: list[int],
    max_steps: int,
    max_concurrent_rollouts: int,
    request_timeout_seconds: float,
    max_tokens: int,
    thinking_budget_tokens: int,
    enable_thinking: bool,
    output_dir: Path,
    label: str,
    policy_version: str,
) -> dict[str, Any]:
    rollouts, collection_summary = asyncio_run(
        collect_rollouts_concurrently_with_summary(
            container_url=container_url,
            inference_url=inference_url,
            model=request_model,
            api_key=inference_api_key,
            seeds=seeds,
            max_steps=max_steps,
            system_prompt=DEFAULT_SYSTEM_PROMPT,
            temperature=0.0,
            max_tokens=max_tokens,
            enable_thinking=enable_thinking,
            thinking_budget_tokens=thinking_budget_tokens,
            policy_version=policy_version,
            target_action_batch_size=4,
            min_action_batch_size=3,
            request_timeout_seconds=request_timeout_seconds,
            max_concurrent_rollouts=max_concurrent_rollouts,
            trace_prefix=label,
            rollout_concurrency=max_concurrent_rollouts,
            rollout_semaphore_limit=max_concurrent_rollouts,
        )
    )
    valid = [
        rollout
        for rollout in rollouts
        if isinstance(rollout, dict) and not rollout.get("error") and is_rollout_payload(rollout)
    ]
    rewards = [float(rollout_outcome_reward(item)) for item in valid]
    summary = {
        "label": label,
        "policy_version": policy_version,
        "seed_count": len(seeds),
        "num_eval_rollouts": len(valid),
        "num_rollout_errors": len(rollouts) - len(valid),
        "mean_outcome_reward": mean(rewards) if rewards else 0.0,
        "max_outcome_reward": max(rewards) if rewards else 0.0,
        "collection": collection_summary,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "summary.json", summary)
    with (output_dir / "rollouts.jsonl").open("w", encoding="utf-8") as handle:
        for rollout in rollouts:
            handle.write(json.dumps(rollout, sort_keys=True) + "\n")
    return summary


def _reload_adapter(
    *,
    inference_admin_url: str,
    inference_api_key: str,
    adapter_dir: str,
    adapter_name: str,
    policy_version: str,
) -> dict[str, Any]:
    headers = {"Content-Type": "application/json"}
    if inference_api_key:
        headers["Authorization"] = f"Bearer {inference_api_key}"
    payload = {
        "lora_name": adapter_name,
        "lora_path": adapter_dir,
        "policy_version": policy_version,
    }
    with httpx.Client(timeout=120.0) as client:
        response = client.post(f"{inference_admin_url.rstrip('/')}/load_adapter", json=payload, headers=headers)
        response.raise_for_status()
        return response.json()


def _reset_to_base(
    *,
    inference_admin_url: str,
    inference_api_key: str,
) -> dict[str, Any]:
    headers = {"Content-Type": "application/json"}
    if inference_api_key:
        headers["Authorization"] = f"Bearer {inference_api_key}"
    with httpx.Client(timeout=30.0) as client:
        response = client.post(
            f"{inference_admin_url.rstrip('/')}/reset",
            json={"policy_version": "bootstrap"},
            headers=headers,
        )
        response.raise_for_status()
        return response.json()


def _initialize_model_and_optimizer(*, base_model: str, lora_rank: int, learning_rate: float) -> tuple[Any, Any, Any]:
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = _load_text_only_causal_lm(base_model=base_model, device=device, use_cache=False)
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_rank * 2,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=DEFAULT_TARGET_MODULES,
    )
    model = get_peft_model(model, lora_config)
    with suppress(Exception):
        model.gradient_checkpointing_enable()
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    return tokenizer, model, optimizer


def _train_iteration(
    *,
    tokenizer: Any,
    model: Any,
    optimizer: Any,
    samples: list[TurnSample],
    clip_epsilon: float,
    max_length: int,
    max_steps: int,
) -> dict[str, Any]:
    import torch

    if not samples:
        return {
            "optimizer_steps": 0,
            "mean_loss": 0.0,
            "mean_old_logprob": 0.0,
            "mean_new_logprob": 0.0,
            "mean_ratio": 0.0,
            "clip_fraction": 0.0,
            "num_samples": 0,
            "skipped": True,
        }

    shuffled = list(samples)
    random.shuffle(shuffled)
    step_limit = max(1, int(max_steps))
    losses: list[float] = []
    old_logprobs: list[float] = []
    new_logprobs: list[float] = []
    ratios: list[float] = []
    clipped_count = 0
    optimizer_steps = 0

    for step_index in range(step_limit):
        sample = shuffled[step_index % len(shuffled)]
        batch = _tokenize_messages_with_assistant_mask(
            tokenizer,
            sample.prompt_messages,
            sample.full_messages,
            max_length=max_length,
        )
        batch = {key: value.to(model.device) for key, value in batch.items()}
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            use_cache=False,
        )
        shifted_logits = outputs.logits[:, :-1, :]
        shifted_labels = batch["labels"][:, 1:]
        selected_logprobs, mask = _selected_token_logprobs(shifted_logits, shifted_labels)
        sequence_new_logprob = selected_logprobs.sum(dim=1)
        del mask
        old_logprob_tensor = torch.tensor([sample.old_logprob], device=model.device, dtype=sequence_new_logprob.dtype)
        advantage_tensor = torch.tensor([sample.advantage], device=model.device, dtype=sequence_new_logprob.dtype)
        ratio = torch.exp(sequence_new_logprob - old_logprob_tensor)
        clipped_ratio = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon)
        objective = torch.minimum(ratio * advantage_tensor, clipped_ratio * advantage_tensor)
        loss = -objective.mean()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        optimizer_steps += 1
        loss_value = float(loss.detach().cpu().item())
        new_logprob_value = float(sequence_new_logprob.detach().cpu().item())
        ratio_value = float(ratio.detach().cpu().item())
        losses.append(loss_value)
        old_logprobs.append(float(sample.old_logprob))
        new_logprobs.append(new_logprob_value)
        ratios.append(ratio_value)
        if ratio_value < (1.0 - clip_epsilon) or ratio_value > (1.0 + clip_epsilon):
            clipped_count += 1

    return {
        "optimizer_steps": optimizer_steps,
        "mean_loss": mean(losses) if losses else 0.0,
        "mean_old_logprob": mean(old_logprobs) if old_logprobs else 0.0,
        "mean_new_logprob": mean(new_logprobs) if new_logprobs else 0.0,
        "mean_ratio": mean(ratios) if ratios else 0.0,
        "clip_fraction": (float(clipped_count) / float(optimizer_steps)) if optimizer_steps else 0.0,
        "num_samples": len(samples),
        "skipped": False,
    }


def _save_adapter(*, model: Any, tokenizer: Any, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(destination)
    tokenizer.save_pretrained(destination)


def _copy_config_bundle(*, config_path: Path, output_dir: Path) -> None:
    shutil.copy2(config_path, output_dir / "run_config.yaml")


def _build_metadata(*, config: dict[str, Any], config_path: Path, output_dir: Path) -> dict[str, Any]:
    return {
        "track": str(config.get("task", {}).get("track") or "rlvr_20min_2xa100_40gb"),
        "task": str(config.get("task", {}).get("name") or "crafter"),
        "baseline": "modal_crafter_grpo",
        "model": str(config.get("model", {}).get("model") or ""),
        "config_path": str(config_path),
        "output_dir": str(output_dir),
        "user_editable_training_script": "src/nanohorizon/rlvr_training.py",
        "runner_script": "scripts/run_crafter_rlvr_qwen35_4b_2xa100_20min.sh",
    }


def run_training(
    *,
    config_path: str | Path,
    output_dir: str | Path,
    container_url: str,
    inference_url: str,
    inference_admin_url: str,
    inference_api_key: str,
    request_model: str,
) -> dict[str, Any]:
    config_path = Path(config_path).expanduser().resolve()
    config = load_config(config_path)
    config_dir = config_path.parent
    out_dir = ensure_dir(output_dir)
    timer = Timer()
    random.seed(int(config.get("training", {}).get("seed", 0) or 0))

    _copy_config_bundle(config_path=config_path, output_dir=out_dir)
    metadata = _build_metadata(config=config, config_path=config_path, output_dir=out_dir)
    write_json(out_dir / "metadata.json", metadata)

    seed_manifest_path = resolve_path(config["data"]["seed_manifest_json"], base_dir=config_dir)
    train_seeds, eval_seeds = _load_seed_manifest(seed_manifest_path)
    rollout_cfg = config.get("rollout", {})
    training_cfg = config.get("training", {})
    evaluation_cfg = config.get("evaluation", {})
    budget_minutes = float(config.get("budget", {}).get("wall_clock_minutes", 20))
    max_iterations = max(1, int(rollout_cfg.get("num_iterations", 4)))
    group_size = max(1, int(rollout_cfg.get("group_size", 4)))
    groups_per_iteration = max(1, int(rollout_cfg.get("groups_per_iteration", 2)))
    rollout_max_steps = max(1, int(rollout_cfg.get("max_steps", 48)))
    rollout_concurrency = max(1, int(rollout_cfg.get("rollout_concurrency", 8)))
    rollout_semaphore_limit = max(1, int(rollout_cfg.get("rollout_semaphore_limit", 4)))
    request_timeout_seconds = float(rollout_cfg.get("request_timeout_seconds", 300.0))
    max_tokens = max(int(rollout_cfg.get("max_tokens", 3072)), 2048)
    thinking_budget_tokens = int(rollout_cfg.get("thinking_budget_tokens", 2000))
    enable_thinking = bool(rollout_cfg.get("enable_thinking", True))
    rollout_temperature = float(rollout_cfg.get("temperature", 0.8))
    target_action_batch_size = int(rollout_cfg.get("target_action_batch_size", 4))
    min_action_batch_size = int(rollout_cfg.get("min_action_batch_size", 3))
    eval_concurrency = max(1, int(evaluation_cfg.get("max_concurrent_rollouts", 4)))
    eval_max_steps = max(1, int(evaluation_cfg.get("max_steps", rollout_max_steps)))
    eval_max_tokens = max(int(evaluation_cfg.get("max_tokens", max_tokens)), 2048)
    eval_rollouts_per_seed = max(1, int(evaluation_cfg.get("rollouts_per_seed", 1)))
    clip_epsilon = float(training_cfg.get("clip_epsilon", 0.2))
    train_max_length = int(training_cfg.get("max_length", 4096))
    train_steps_per_iteration = int(training_cfg.get("max_steps_per_iteration", training_cfg.get("max_steps", 8)))

    _reset_to_base(
        inference_admin_url=_normalize_admin_url(inference_admin_url),
        inference_api_key=inference_api_key,
    )

    tokenizer, model, optimizer = _initialize_model_and_optimizer(
        base_model=str(config["model"]["model"]),
        lora_rank=int(training_cfg.get("lora_rank", 16)),
        learning_rate=float(training_cfg.get("learning_rate", 1.0e-5)),
    )

    periodic_eval_root = out_dir / "periodic_eval"
    adapters_root = out_dir / "adapters"
    iterations_root = out_dir / "iterations"
    policy_history: list[dict[str, Any]] = []
    iteration_summaries: list[dict[str, Any]] = []

    eval_seed_schedule = [seed for seed in eval_seeds for _ in range(eval_rollouts_per_seed)]
    step0_summary = _evaluate_policy(
        container_url=container_url,
        inference_url=_normalize_inference_url(inference_url),
        inference_api_key=inference_api_key,
        request_model=request_model,
        seeds=eval_seed_schedule,
        max_steps=eval_max_steps,
        max_concurrent_rollouts=eval_concurrency,
        request_timeout_seconds=request_timeout_seconds,
        max_tokens=eval_max_tokens,
        thinking_budget_tokens=thinking_budget_tokens,
        enable_thinking=enable_thinking,
        output_dir=periodic_eval_root / "step_000",
        label="periodic_eval_step_000",
        policy_version="bootstrap",
    )

    for iteration_index in range(max_iterations):
        if timer.elapsed_minutes >= budget_minutes:
            break

        iteration_dir = ensure_dir(iterations_root / f"iter_{iteration_index:03d}")
        seeds_for_iteration = _iteration_train_seeds(
            all_train_seeds=train_seeds,
            iteration_index=iteration_index,
            groups_per_iteration=groups_per_iteration,
        )
        rollout_seed_schedule = _build_rollout_seed_schedule(seeds=seeds_for_iteration, group_size=group_size)
        rollouts, rollout_summary = asyncio_run(
            collect_rollouts_concurrently_with_summary(
                container_url=container_url,
                inference_url=_normalize_inference_url(inference_url),
                model=request_model,
                api_key=inference_api_key,
                seeds=rollout_seed_schedule,
                max_steps=rollout_max_steps,
                system_prompt=DEFAULT_SYSTEM_PROMPT,
                temperature=rollout_temperature,
                max_tokens=max_tokens,
                enable_thinking=enable_thinking,
                thinking_budget_tokens=thinking_budget_tokens,
                policy_version=policy_history[-1]["policy_version"] if policy_history else "bootstrap",
                target_action_batch_size=target_action_batch_size,
                min_action_batch_size=min_action_batch_size,
                request_timeout_seconds=request_timeout_seconds,
                max_concurrent_rollouts=rollout_concurrency,
                trace_prefix=f"rlvr_iter_{iteration_index:03d}",
                rollout_concurrency=rollout_concurrency,
                rollout_semaphore_limit=rollout_semaphore_limit,
            )
        )
        with (Path(iteration_dir) / "rollouts.jsonl").open("w", encoding="utf-8") as handle:
            for rollout in rollouts:
                handle.write(json.dumps(rollout, sort_keys=True) + "\n")

        turn_samples, sample_summary = _build_turn_samples(rollouts=rollouts, group_size=group_size)
        train_summary = _train_iteration(
            tokenizer=tokenizer,
            model=model,
            optimizer=optimizer,
            samples=turn_samples,
            clip_epsilon=clip_epsilon,
            max_length=train_max_length,
            max_steps=train_steps_per_iteration,
        )
        adapter_name = f"policy-iter-{iteration_index:03d}"
        policy_version = f"iter_{iteration_index:03d}"
        adapter_dir = adapters_root / f"iter_{iteration_index:03d}"
        _save_adapter(model=model, tokenizer=tokenizer, destination=adapter_dir)
        reload_payload = _reload_adapter(
            inference_admin_url=_normalize_admin_url(inference_admin_url),
            inference_api_key=inference_api_key,
            adapter_dir=str(adapter_dir),
            adapter_name=adapter_name,
            policy_version=policy_version,
        )
        summary_payload = {
            "rollout": rollout_summary,
            "sample_summary": sample_summary,
            "train": train_summary,
            "adapter_name": adapter_name,
            "adapter_dir": str(adapter_dir),
            "policy_version": policy_version,
            "reload": reload_payload,
        }
        write_json(Path(iteration_dir) / "summary.json", summary_payload)
        policy_history.append(
            {
                "iteration_index": iteration_index,
                "policy_version": policy_version,
                "adapter_name": adapter_name,
                "adapter_dir": str(adapter_dir),
                "mean_outcome_reward": float(sample_summary["mean_outcome_reward"]),
            }
        )
        iteration_summaries.append(summary_payload)

        periodic_summary = _evaluate_policy(
            container_url=container_url,
            inference_url=_normalize_inference_url(inference_url),
            inference_api_key=inference_api_key,
            request_model=request_model,
            seeds=eval_seed_schedule,
            max_steps=eval_max_steps,
            max_concurrent_rollouts=eval_concurrency,
            request_timeout_seconds=request_timeout_seconds,
            max_tokens=eval_max_tokens,
            thinking_budget_tokens=thinking_budget_tokens,
            enable_thinking=enable_thinking,
            output_dir=periodic_eval_root / f"step_{iteration_index + 1:03d}",
            label=f"periodic_eval_step_{iteration_index + 1:03d}",
            policy_version=policy_version,
        )
        summary_payload["periodic_eval"] = periodic_summary
        write_json(Path(iteration_dir) / "summary.json", summary_payload)

        if timer.elapsed_minutes >= budget_minutes:
            break

    final_policy_version = policy_history[-1]["policy_version"] if policy_history else "bootstrap"
    final_eval_summary = _evaluate_policy(
        container_url=container_url,
        inference_url=_normalize_inference_url(inference_url),
        inference_api_key=inference_api_key,
        request_model=request_model,
        seeds=eval_seed_schedule,
        max_steps=eval_max_steps,
        max_concurrent_rollouts=eval_concurrency,
        request_timeout_seconds=request_timeout_seconds,
        max_tokens=eval_max_tokens,
        thinking_budget_tokens=thinking_budget_tokens,
        enable_thinking=enable_thinking,
        output_dir=out_dir / "final_eval",
        label="final_eval",
        policy_version=final_policy_version,
    )

    metrics = {
        "track": str(config.get("task", {}).get("track") or "rlvr_20min_2xa100_40gb"),
        "baseline": "modal_crafter_grpo",
        "model": str(config.get("model", {}).get("model") or ""),
        "started_at": timer.started_at,
        "ended_at": timer.ended_at,
        "elapsed_minutes": timer.elapsed_minutes,
        "budget_minutes": budget_minutes,
        "iterations_completed": len(policy_history),
        "step0_mean_outcome_reward": float(step0_summary["mean_outcome_reward"]),
        "final_mean_outcome_reward": float(final_eval_summary["mean_outcome_reward"]),
        "reward_delta_from_bootstrap": float(final_eval_summary["mean_outcome_reward"]) - float(step0_summary["mean_outcome_reward"]),
        "final_policy_version": final_policy_version,
        "periodic_eval_count": len(list(periodic_eval_root.glob("step_*/summary.json"))),
    }
    write_json(out_dir / "metrics.json", metrics)
    write_json(out_dir / "system_info.json", system_info())
    write_json(out_dir / "policy_history.json", policy_history)
    write_json(out_dir / "iteration_summaries.json", iteration_summaries)
    write_json(out_dir / "final_eval_summary.json", final_eval_summary)
    write_json(
        out_dir / "run_timing.json",
        {
            "started_at": timer.started_at,
            "ended_at": timer.ended_at,
            "elapsed_minutes": timer.elapsed_minutes,
            "budget_minutes": budget_minutes,
        },
    )
    write_text(
        out_dir / "command.txt",
        f"NANOHORIZON_RLVR_OUTPUT_DIR={out_dir} ./scripts/run_crafter_rlvr_qwen35_4b_2xa100_20min.sh\n",
    )

    release_cuda_memory()
    return {
        "output_dir": str(out_dir),
        "metrics": metrics,
        "final_eval_summary": final_eval_summary,
        "policy_history": policy_history,
    }


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    output_dir = args.output_dir or config["output"]["root_dir"]
    result = run_training(
        config_path=args.config,
        output_dir=output_dir,
        container_url=str(args.container_url),
        inference_url=str(args.inference_url),
        inference_admin_url=str(args.inference_admin_url),
        inference_api_key=str(args.inference_api_key),
        request_model=str(args.request_model or config.get("model", {}).get("served_model_name") or config["model"]["model"]),
    )
    print(json.dumps(result, indent=2, sort_keys=True))


def asyncio_run(coro: Any) -> Any:
    import asyncio

    return asyncio.run(coro)


if __name__ == "__main__":
    main()
