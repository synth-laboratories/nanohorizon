from __future__ import annotations

import argparse
import asyncio
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import yaml
from gepa import EvaluationBatch, GEPAAdapter, optimize

from nanohorizon.common import (
    Timer,
    ensure_dir,
    load_config,
    now_utc_iso,
    resolve_path,
    system_info,
    write_json,
    write_text,
)
from nanohorizon.crafter_data import (
    collect_rollouts_concurrently_with_summary,
    flatten_messages,
    is_rollout_payload,
    rollout_llm_call_count,
    rollout_outcome_reward,
    rollout_turns,
)
from nanohorizon.custom_vllm.runtime import build_thinking_budget_request_overrides
from nanohorizon.openai_compat import create_chat_completion

TRACK_ID = "prompt_opt_1usd_gpt54_family"
REFLECTION_PROMPT_TEMPLATE = """I provided an assistant with the following Crafter system prompt:
```
<curr_param>
```

The following are examples of task inputs, model outputs, and feedback:
```
<side_info>
```

Write a revised Crafter system prompt.

Hard requirements you must preserve:
- The policy must think if needed, then use the `crafter_interact` tool exactly once.
- The final answer must not be plain text actions, JSON, or prose outside the tool call.
- The prompt should ask for exactly 4 valid Crafter actions unless the episode is already done.
- The prompt should prioritize early-game resource gathering and avoid repeated movement loops.

Return only the revised system prompt inside ``` blocks."""


@dataclass(frozen=True)
class PromptOptExample:
    seed: int
    split: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GEPA-based Crafter prompt optimization baseline")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--container-url", required=True)
    parser.add_argument("--inference-url", required=True)
    parser.add_argument("--inference-api-key", default="")
    parser.add_argument("--request-model", required=True)
    return parser.parse_args()


def _truncate(text: str, limit: int = 1200) -> str:
    value = str(text or "").strip()
    if len(value) <= limit:
        return value
    return value[: limit - 3].rstrip() + "..."


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


def _first_user_observation(rollout: dict[str, Any]) -> str:
    for turn in rollout_turns(rollout):
        messages = turn.get("prompt_messages")
        if isinstance(messages, list) and messages:
            return _truncate(flatten_messages([m for m in messages if isinstance(m, dict)]), 2000)
    return ""


def _actions_summary(rollout: dict[str, Any]) -> str:
    chunks: list[str] = []
    for turn in rollout_turns(rollout):
        actions = turn.get("actions")
        if isinstance(actions, list) and actions:
            chunks.append(",".join(str(item) for item in actions))
    return _truncate(" | ".join(chunks), 600)


def _assistant_summary(rollout: dict[str, Any]) -> str:
    snippets: list[str] = []
    for turn in rollout_turns(rollout):
        text = str(turn.get("assistant_text") or "").strip()
        if text:
            snippets.append(text)
    return _truncate("\n\n".join(snippets), 1200)


def _reasoning_summary(rollout: dict[str, Any]) -> str:
    snippets: list[str] = []
    for turn in rollout_turns(rollout):
        text = str(turn.get("reasoning_text") or "").strip()
        if text:
            snippets.append(text)
    return _truncate("\n\n".join(snippets), 1200)


def _achievement_summary(rollout: dict[str, Any]) -> list[str]:
    metadata = rollout.get("metadata")
    if isinstance(metadata, dict):
        achievements = metadata.get("achievements")
        if isinstance(achievements, list):
            return [str(item) for item in achievements if str(item).strip()]
    reward_info = rollout.get("reward_info")
    if isinstance(reward_info, dict):
        details = reward_info.get("details")
        if isinstance(details, dict):
            achievements = details.get("achievements")
            if isinstance(achievements, list):
                return [str(item) for item in achievements if str(item).strip()]
    return []


def _invalid_parse_count(rollout: dict[str, Any]) -> int:
    count = 0
    for turn in rollout_turns(rollout):
        if bool(turn.get("invalid_parse")):
            count += 1
    return count


def _feedback_for_rollout(rollout: dict[str, Any], score: float) -> str:
    achievements = _achievement_summary(rollout)
    llm_calls = rollout_llm_call_count(rollout)
    invalid_parses = _invalid_parse_count(rollout)
    action_summary = _actions_summary(rollout)
    if score > 0.0:
        parts = [
            f"This rollout achieved reward {score:.2f}.",
            "Preserve the behaviors that led to progress.",
        ]
        if achievements:
            parts.append(f"Unlocked or observed achievements: {', '.join(achievements[:6])}.")
        if action_summary:
            parts.append(f"Observed action sequence: {action_summary}.")
        parts.append(
            "Keep the tool-calling contract strict: think if needed, then use the `crafter_interact` tool exactly once with 3-4 valid Crafter actions. Strengthen instructions about gathering nearby resources, using `do` only when adjacent to a useful target, and avoiding repeated no-op movement loops."
        )
        return " ".join(parts)
    parts = [f"This rollout achieved reward {score:.2f} and failed to make progress."]
    if invalid_parses:
        parts.append(
            f"It produced {invalid_parses} invalid parse(s); make the prompt stricter about one tool call only with 1-4 valid Crafter actions."
        )
    if llm_calls <= 1:
        parts.append(
            "The model had very few decision opportunities, so the prompt should encourage a short but useful macro-action that approaches a tree or other gatherable resource immediately."
        )
    if action_summary:
        parts.append(f"Observed action sequence: {action_summary}.")
    parts.append(
        "Emphasize early-game progression: move toward trees, use `do` when adjacent, avoid sleep or crafting unless the inventory and local state justify it, and break out of repeated movement loops. The final answer must be one `crafter_interact` tool call, not a plain-text action list or JSON blob."
    )
    return " ".join(parts)


def _resource_progress_bonus(rollout: dict[str, Any]) -> float:
    metadata = rollout.get("metadata")
    inventory = metadata.get("inventory") if isinstance(metadata, dict) else None
    if not isinstance(inventory, dict):
        return 0.0
    useful_keys = (
        "sapling",
        "wood",
        "stone",
        "coal",
        "iron",
        "wood_pickaxe",
        "stone_pickaxe",
        "iron_pickaxe",
        "wood_sword",
        "stone_sword",
        "iron_sword",
    )
    total = 0.0
    for key in useful_keys:
        value = inventory.get(key)
        if isinstance(value, (int, float)) and value > 0:
            total += float(value)
    return min(total, 3.0)


def _action_quality_bonus(rollout: dict[str, Any]) -> float:
    flattened: list[str] = []
    for turn in rollout_turns(rollout):
        actions = turn.get("actions")
        if isinstance(actions, list):
            flattened.extend(str(item) for item in actions if str(item).strip())
    if not flattened:
        return -0.05
    has_move = any(action.startswith("move_") for action in flattened)
    has_do = any(action == "do" for action in flattened)
    repeated_single = len(set(flattened)) == 1
    bonus = 0.0
    if has_move:
        bonus += 0.05
    if has_do:
        bonus += 0.05
    if 3 <= len(flattened) <= 4:
        bonus += 0.05
    if repeated_single:
        bonus -= 0.05
    return bonus


def _search_score(rollout: dict[str, Any]) -> float:
    if not is_rollout_payload(rollout):
        return 0.0
    if str(rollout.get("success_status") or "").strip().lower() != "success":
        return 0.0
    outcome = rollout_outcome_reward(rollout)
    achievements = len(_achievement_summary(rollout))
    invalid = _invalid_parse_count(rollout)
    shaped = (
        float(outcome)
        + 0.10 * min(float(achievements), 2.0)
        + 0.05 * _resource_progress_bonus(rollout)
        + 0.25 * _action_quality_bonus(rollout)
        - 0.15 * float(invalid)
    )
    return max(shaped, 0.0)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    path.write_text(content, encoding="utf-8")


def _load_seed_splits(config: dict[str, Any], base_dir: Path) -> tuple[list[PromptOptExample], list[PromptOptExample]]:
    data_cfg = config["data"]
    seed_file = resolve_path(str(data_cfg["seed_file"]), base_dir=base_dir)
    payload = json.loads(seed_file.read_text(encoding="utf-8"))
    train_seeds = [int(item) for item in payload.get("train_seeds", [])]
    eval_seeds = [int(item) for item in payload.get("eval_seeds", [])]
    num_train = int(data_cfg.get("num_train_seeds", len(train_seeds)))
    num_eval = int(data_cfg.get("num_eval_seeds", len(eval_seeds)))
    trainset = [PromptOptExample(seed=seed, split="train") for seed in train_seeds[:num_train]]
    valset = [PromptOptExample(seed=seed, split="eval") for seed in eval_seeds[:num_eval]]
    if not trainset or not valset:
        raise ValueError("prompt-opt requires non-empty train and eval seed sets")
    return trainset, valset


class CrafterPromptOptAdapter(GEPAAdapter[PromptOptExample, dict[str, Any], dict[str, Any]]):
    def __init__(
        self,
        *,
        container_url: str,
        inference_url: str,
        inference_api_key: str,
        request_model: str,
        rollout_cfg: dict[str, Any],
    ) -> None:
        self.container_url = container_url
        self.inference_url = inference_url
        self.inference_api_key = inference_api_key
        self.request_model = request_model
        self.rollout_cfg = rollout_cfg

    def evaluate(
        self,
        batch: list[PromptOptExample],
        candidate: dict[str, str],
        capture_traces: bool = False,
    ) -> EvaluationBatch[dict[str, Any], dict[str, Any]]:
        if not batch:
            return EvaluationBatch(outputs=[], scores=[], trajectories=[] if capture_traces else None)
        system_prompt = str(candidate["system_prompt"])
        seeds = [example.seed for example in batch]
        rollouts, summary = asyncio.run(
            collect_rollouts_concurrently_with_summary(
                container_url=self.container_url,
                inference_url=self.inference_url,
                model=self.request_model,
                api_key=self.inference_api_key,
                seeds=seeds,
                max_steps=int(self.rollout_cfg["max_steps"]),
                system_prompt=system_prompt,
                temperature=float(self.rollout_cfg["temperature"]),
                max_tokens=int(self.rollout_cfg["max_tokens"]),
                enable_thinking=bool(self.rollout_cfg["enable_thinking"]),
                thinking_budget_tokens=int(self.rollout_cfg["thinking_budget_tokens"]),
                policy_version="prompt-opt",
                target_action_batch_size=int(self.rollout_cfg["target_action_batch_size"]),
                min_action_batch_size=int(self.rollout_cfg["min_action_batch_size"]),
                request_timeout_seconds=float(self.rollout_cfg["request_timeout_seconds"]),
                max_concurrent_rollouts=int(self.rollout_cfg["max_concurrent_rollouts"]),
                trace_prefix="prompt_opt",
                rollout_concurrency=int(self.rollout_cfg["rollout_concurrency"]),
                rollout_semaphore_limit=int(self.rollout_cfg["rollout_semaphore_limit"]),
            )
        )
        outputs: list[dict[str, Any]] = []
        scores: list[float] = []
        trajectories: list[dict[str, Any]] = []
        objective_scores: list[dict[str, float]] = []
        for example, rollout in zip(batch, rollouts, strict=False):
            valid = (
                isinstance(rollout, dict)
                and not rollout.get("error")
                and is_rollout_payload(rollout)
                and str(rollout.get("success_status") or "").strip().lower() == "success"
            )
            outcome_reward = rollout_outcome_reward(rollout) if valid else 0.0
            score = _search_score(rollout) if valid else 0.0
            outputs.append(rollout)
            scores.append(score)
            objective_scores.append(
                {
                    "search_score": float(score),
                    "outcome_reward": float(outcome_reward),
                    "llm_call_count": float(rollout_llm_call_count(rollout) if valid else 0),
                    "achievement_count": float(len(_achievement_summary(rollout)) if valid else 0),
                }
            )
            if capture_traces:
                trajectories.append(
                    {
                        "seed": example.seed,
                        "split": example.split,
                        "score": outcome_reward,
                        "search_score": score,
                        "rollout": rollout,
                        "summary": summary,
                    }
                )
        return EvaluationBatch(
            outputs=outputs,
            scores=scores,
            trajectories=trajectories if capture_traces else None,
            objective_scores=objective_scores,
        )

    def make_reflective_dataset(
        self,
        candidate: dict[str, str],
        eval_batch: EvaluationBatch[dict[str, Any], dict[str, Any]],
        components_to_update: list[str],
    ) -> dict[str, list[dict[str, Any]]]:
        trajectories = eval_batch.trajectories or []
        ranked = sorted(trajectories, key=lambda item: float(item.get("score", 0.0)))
        records: list[dict[str, Any]] = []
        for entry in ranked[: min(6, len(ranked))]:
            rollout = entry.get("rollout")
            if not isinstance(rollout, dict):
                continue
            score = float(entry.get("score", 0.0))
            records.append(
                {
                    "Inputs": {
                        "seed": entry.get("seed"),
                        "observation": _first_user_observation(rollout),
                    },
                    "Generated Outputs": {
                        "actions": _actions_summary(rollout),
                        "assistant_response": _assistant_summary(rollout),
                        "reasoning": _reasoning_summary(rollout),
                        "reward": score,
                        "achievements": _achievement_summary(rollout),
                    },
                    "Feedback": _feedback_for_rollout(rollout, score),
                }
            )
        return {component: list(records) for component in components_to_update}


def _build_reflection_lm(
    *,
    requested_model: str,
    inference_url: str,
    inference_api_key: str,
    request_model: str,
    backend: str,
):
    resolved_backend = backend.strip().lower() or "auto"
    openai_available = bool(str(os.getenv("OPENAI_API_KEY") or "").strip())
    if resolved_backend == "openai" or (
        resolved_backend == "auto" and openai_available and requested_model.startswith("gpt-")
    ):
        return requested_model, "openai"

    def _reflection_lm(prompt: str | list[dict[str, Any]]) -> str:
        messages: list[dict[str, str]] = [
            {
                "role": "system",
                "content": (
                    "You rewrite Crafter system prompts for a tool-calling policy. "
                    "Preserve these hard requirements: the policy must use the "
                    "`crafter_interact` tool exactly once, must not answer with JSON "
                    "or a plain-text action list, and should usually request exactly "
                    "4 valid Crafter actions unless the episode is already done. "
                    "Return only the revised prompt text."
                ),
            }
        ]
        if isinstance(prompt, str):
            messages.append({"role": "user", "content": prompt})
        else:
            messages.extend(
                [
                    {
                        "role": str(item.get("role") or "user"),
                        "content": str(item.get("content") or ""),
                    }
                    for item in prompt
                    if isinstance(item, dict)
                ]
            )
        payload = create_chat_completion(
            model=request_model,
            messages=messages,
            max_tokens=1024,
            temperature=0.2,
            base_url=f"{inference_url.rstrip('/')}/v1",
            api_key=inference_api_key,
            timeout_seconds=300.0,
            extra_body=build_thinking_budget_request_overrides(
                enable_thinking=False,
                thinking_budget=0,
            ),
        )
        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices:
            return ""
        message = choices[0].get("message", {})
        return str(message.get("content") or "").strip()

    return _reflection_lm, "policy_inference"


def _summarize_eval(
    *,
    dataset: list[PromptOptExample],
    candidate: dict[str, str],
    adapter: CrafterPromptOptAdapter,
    name: str,
    output_dir: Path,
) -> dict[str, Any]:
    batch = adapter.evaluate(dataset, candidate, capture_traces=True)
    reward_scores = [
        float(score_map.get("outcome_reward", 0.0))
        for score_map in (batch.objective_scores or [])
    ]
    mean_reward = (sum(reward_scores) / len(reward_scores)) if reward_scores else 0.0
    summary = {
        "name": name,
        "num_rollouts": len(reward_scores),
        "mean_outcome_reward": mean_reward,
        "max_outcome_reward": max(reward_scores) if reward_scores else 0.0,
        "details": [
            {
                "seed": example.seed,
                "score": reward_score,
                "search_score": search_score,
                "rollout_id": str(output.get("rollout_id") or ""),
                "success_status": output.get("success_status"),
                "llm_call_count": rollout_llm_call_count(output),
            }
            for example, reward_score, search_score, output in zip(
                dataset,
                reward_scores,
                [float(score) for score in batch.scores],
                batch.outputs,
                strict=False,
            )
        ],
    }
    write_json(output_dir / f"{name}_summary.json", summary)
    _write_jsonl(
        output_dir / f"{name}_rollouts.jsonl",
        [
            output
            for output in batch.outputs
            if isinstance(output, dict)
        ],
    )
    return summary


def run_training(
    *,
    config_path: Path,
    output_dir: Path,
    container_url: str,
    inference_url: str,
    inference_api_key: str,
    request_model: str,
) -> dict[str, Any]:
    config = load_config(config_path)
    base_dir = config_path.parent
    timer = Timer()
    rollout_cfg = dict(config["rollout"])
    rollout_inference_url = _normalize_inference_url(inference_url)
    trainset, valset = _load_seed_splits(config, base_dir)
    seed_prompt = str(config["prompt"]["seed_prompt"]).strip()
    component_name = str(config["prompt"].get("component_name", "system_prompt")).strip() or "system_prompt"
    seed_candidate = {component_name: seed_prompt}
    reflection_model = str(config["optimizer"]["proposer_model"]).strip()
    reflection_lm, reflection_backend = _build_reflection_lm(
        requested_model=reflection_model,
        inference_url=inference_url,
        inference_api_key=inference_api_key,
        request_model=request_model,
        backend=str(config["optimizer"].get("reflection_backend", "auto")),
    )
    selector = cast(
        Literal["pareto", "current_best", "epsilon_greedy", "top_k_pareto"],
        str(config["search"].get("candidate_selection_strategy", "current_best")),
    )
    adapter = CrafterPromptOptAdapter(
        container_url=container_url,
        inference_url=rollout_inference_url,
        inference_api_key=inference_api_key,
        request_model=request_model,
        rollout_cfg=rollout_cfg,
    )
    run_dir = ensure_dir(output_dir / "gepa_run")
    result = optimize(
        seed_candidate=seed_candidate,
        trainset=trainset,
        valset=valset,
        adapter=adapter,
        reflection_lm=reflection_lm,
        max_metric_calls=int(config["search"]["max_metric_calls"]),
        reflection_minibatch_size=int(config["search"]["reflection_minibatch_size"]),
        candidate_selection_strategy=selector,
        run_dir=str(run_dir),
        seed=int(config["search"].get("seed", 0)),
        reflection_prompt_template=REFLECTION_PROMPT_TEMPLATE,
        raise_on_exception=True,
    )
    best_candidate = result.best_candidate
    if not isinstance(best_candidate, dict):
        raise TypeError("GEPA best_candidate must be a dict for prompt optimization")
    best_score = float(result.val_aggregate_scores[result.best_idx])
    bootstrap_score = float(result.val_aggregate_scores[0])
    base_eval = _summarize_eval(
        dataset=valset,
        candidate=seed_candidate,
        adapter=adapter,
        name="base_eval",
        output_dir=output_dir,
    )
    best_eval = _summarize_eval(
        dataset=valset,
        candidate=best_candidate,
        adapter=adapter,
        name="best_eval",
        output_dir=output_dir,
    )
    prompt_bundle = {
        "seed_candidate": seed_candidate,
        "best_candidate": best_candidate,
        "best_candidate_idx": int(result.best_idx),
        "candidates": result.candidates,
        "val_aggregate_scores": result.val_aggregate_scores,
        "total_metric_calls": result.total_metric_calls,
    }
    write_json(output_dir / "prompt_bundle.json", prompt_bundle)
    write_json(output_dir / "gepa_result.json", result.to_dict())
    run_config = {
        "track": TRACK_ID,
        "task": str(config["task"]["name"]),
        "base_model": str(config["policy"]["model"]),
        "optimizer_budget_usd": float(config["optimizer"]["budget_usd"]),
        "optimizer_models": list(config["optimizer"]["allowed_models"]),
        "rollout": rollout_cfg,
        "train_seeds": [example.seed for example in trainset],
        "eval_seeds": [example.seed for example in valset],
    }
    write_text(output_dir / "run_config.yaml", yaml.safe_dump(run_config, sort_keys=True))
    write_json(
        output_dir / "metadata.json",
        {
            "name": os.environ.get("NANOHORIZON_RECORD_NAME", "reference_baseline"),
            "track": TRACK_ID,
            "task": str(config["task"]["name"]),
            "base_model": str(config["policy"]["model"]),
            "optimizer_budget_usd": float(config["optimizer"]["budget_usd"]),
            "optimizer_models": list(config["optimizer"]["allowed_models"]),
            "created_at": now_utc_iso()[:10],
        },
    )
    metrics = {
        "track": TRACK_ID,
        "baseline": "gepa_crafter_prompt_optimization",
        "status": "success",
        "primary_score": float(best_eval["mean_outcome_reward"]),
        "bootstrap_score": float(base_eval["mean_outcome_reward"]),
        "best_gepa_val_score": best_score,
        "score_delta": float(best_eval["mean_outcome_reward"] - base_eval["mean_outcome_reward"]),
        "gepa_score_delta": float(best_score - bootstrap_score),
        "num_candidates": int(len(result.candidates)),
        "best_candidate_idx": int(result.best_idx),
        "total_metric_calls": int(result.total_metric_calls or 0),
        "elapsed_minutes": timer.elapsed_minutes,
        "policy_model": str(config["policy"]["model"]),
        "reflection_model": reflection_model,
        "request_model": request_model,
        "rollout_inference_url": rollout_inference_url,
        "reflection_backend": reflection_backend,
    }
    write_json(output_dir / "metrics.json", metrics)
    write_json(output_dir / "system_info.json", system_info())
    write_text(
        output_dir / "command.txt",
        "./scripts/run_crafter_prompt_opt_qwen35_4b_gpt54_budget.sh\n",
    )
    write_text(
        output_dir / "notes.md",
        (
            f"Bootstrap held-out reward: {base_eval['mean_outcome_reward']:.3f}\n"
            f"Best held-out reward: {best_eval['mean_outcome_reward']:.3f}\n"
            f"Hillclimb delta: {best_eval['mean_outcome_reward'] - base_eval['mean_outcome_reward']:.3f}\n"
        ),
    )
    return {
        "output_dir": str(output_dir),
        "best_prompt": str(best_candidate[component_name]),
        "metrics": metrics,
        "base_eval": base_eval,
        "best_eval": best_eval,
    }


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).expanduser().resolve()
    config = load_config(config_path)
    base_dir = config_path.parent
    default_output_dir = resolve_path(str(config["output"]["root_dir"]), base_dir=base_dir)
    output_dir = ensure_dir(args.output_dir or default_output_dir)
    result = run_training(
        config_path=config_path,
        output_dir=output_dir,
        container_url=str(args.container_url).strip(),
        inference_url=str(args.inference_url).strip(),
        inference_api_key=str(args.inference_api_key).strip(),
        request_model=str(args.request_model).strip(),
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
