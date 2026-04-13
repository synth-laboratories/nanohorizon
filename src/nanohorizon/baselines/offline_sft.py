"""Craftax offline FBC / SFT baseline.

Known issue:
- Scaling teacher volume alone is not enough. Current high-volume runs still
  tend to admit mostly `collect_wood` traces, even when the filter allowlist
  includes stronger achievements like `eat_cow`, `defeat_zombie`, and
  `defeat_skeleton`.

What submissions should focus on:
- Improve high-quality SFT data selection, not just total token volume.
- Bias the admitted dataset toward traces that unlock stronger achievements,
  rather than low-value or overly common positives.
- Treat data filtering and trace selection as a primary optimization surface
  for improving held-out score.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import platform
import signal
import sys
import time
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean
from typing import Any, Callable, cast

import httpx
import yaml
from nanohorizon.craftax_core.metadata import DEFAULT_ACTION_NAMES, PRIMARY_TOOL_NAME
from nanohorizon.shared.craftax_data import rollout_achievements

# modal is only needed when NANOHORIZON_TRAIN_ON_MODAL=1 or for the Modal
# entrypoint at the bottom of this file. Lazy-import to avoid failures when
# running as a plain training script inside Modal workers.
modal = None  # type: ignore[assignment]
def _import_modal():
    global modal
    if modal is None:
        import modal as _modal
        modal = _modal
    return modal

def _import_evaluate_model():
    from nanohorizon.shared.eval_model import evaluate_model
    return evaluate_model


@dataclass
class TrainingExecution:
    output_dir: str
    adapter_dir: str
    examples_seen: int
    optimizer_steps: int
    mean_loss: float
    backend: str
    total_tokens: int = 0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    mean_sequence_length: float = 0.0
    max_sequence_length: int = 0


def now_utc_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    text = config_path.read_text(encoding="utf-8")
    payload = json.loads(text) if config_path.suffix.lower() == ".json" else yaml.safe_load(text)
    if not isinstance(payload, dict):
        raise ValueError(f"config must decode to an object: {config_path}")
    return payload


def resolve_path(path: str | Path, *, base_dir: str | Path | None = None) -> Path:
    target = Path(path).expanduser()
    if target.is_absolute():
        return target.resolve()
    anchor = Path(base_dir).expanduser().resolve() if base_dir is not None else Path.cwd().resolve()
    return (anchor / target).resolve()


def ensure_dir(path: str | Path) -> Path:
    target = Path(path).expanduser().resolve()
    target.mkdir(parents=True, exist_ok=True)
    return target


def write_json(path: str | Path, payload: Any) -> None:
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: str | Path, text: str) -> None:
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw in Path(path).expanduser().resolve().read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"jsonl row must be an object: {path}")
        rows.append(payload)
    return rows


def filter_rows_for_training(
    rows: list[dict[str, Any]],
    *,
    require_trainable: bool,
    drop_invalid_parse: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not require_trainable and not drop_invalid_parse:
        return rows, {
            "input_rows": len(rows),
            "kept_rows": len(rows),
            "dropped_rows": 0,
            "require_trainable": False,
            "drop_invalid_parse": False,
            "dropped_non_trainable": 0,
            "dropped_invalid_parse": 0,
        }

    kept_rows: list[dict[str, Any]] = []
    dropped_non_trainable = 0
    dropped_invalid_parse = 0
    for row in rows:
        metadata = row.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}
        if require_trainable and not bool(metadata.get("trainable", True)):
            dropped_non_trainable += 1
            continue
        if drop_invalid_parse and bool(metadata.get("invalid_parse", False)):
            dropped_invalid_parse += 1
            continue
        kept_rows.append(row)

    summary = {
        "input_rows": len(rows),
        "kept_rows": len(kept_rows),
        "dropped_rows": len(rows) - len(kept_rows),
        "require_trainable": require_trainable,
        "drop_invalid_parse": drop_invalid_parse,
        "dropped_non_trainable": dropped_non_trainable,
        "dropped_invalid_parse": dropped_invalid_parse,
    }
    return kept_rows, summary


def system_info() -> dict[str, Any]:
    return {
        "timestamp_utc": now_utc_iso(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "hostname": platform.node(),
        "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES", ""),
        "runpod_pod_id": os.getenv("RUNPOD_POD_ID", ""),
    }


class Timer:
    def __init__(self) -> None:
        self.start = time.time()
        self.started_at = now_utc_iso()

    @property
    def elapsed_seconds(self) -> float:
        return time.time() - self.start

    @property
    def elapsed_minutes(self) -> float:
        return self.elapsed_seconds / 60.0

    @property
    def ended_at(self) -> str:
        return now_utc_iso()


CRAFTAX_ACTION_ENUM = list(DEFAULT_ACTION_NAMES)

CRAFTAX_INTERACT_TOOL = {
    "type": "function",
    "function": {
        "name": PRIMARY_TOOL_NAME,
        "description": "Choose the next short Craftax macro-action sequence.",
        "parameters": {
            "type": "object",
            "properties": {
                "actions_list": {
                    "type": "array",
                    "items": {"type": "string", "enum": CRAFTAX_ACTION_ENUM},
                    "minItems": 1,
                    "maxItems": 10,
                }
            },
            "required": ["actions_list"],
            "additionalProperties": False,
        },
    },
}


def _is_synthtunnel_url(url: str) -> bool:
    try:
        parsed = httpx.URL(url)
        hostname = parsed.host or ""
        path = parsed.path or ""
    except Exception:
        return False
    return hostname == "st.usesynth.ai" or hostname.endswith(".st.usesynth.ai") or "/s/rt_" in path


def _is_cloudflare_quick_tunnel_url(url: str) -> bool:
    try:
        hostname = httpx.URL(url).host or ""
    except Exception:
        return False
    return hostname.endswith(".trycloudflare.com")


def _container_headers(
    *,
    container_url: str,
    container_worker_token: str | None,
    environment_api_key: str | None,
) -> dict[str, str]:
    headers: dict[str, str] = {"Content-Type": "application/json"}
    if _is_synthtunnel_url(container_url):
        worker_token = (container_worker_token or "").strip()
        if not worker_token:
            raise ValueError(
                "container_worker_token is required for SynthTunnel container_url. "
                "Pass the worker token emitted by the tunnel helper."
            )
        headers["Authorization"] = f"Bearer {worker_token}"
        return headers
    env_key = (environment_api_key or "").strip()
    if env_key:
        headers["x-api-key"] = env_key
    return headers


def flatten_messages(messages: list[dict[str, Any]]) -> str:
    rendered: list[str] = []
    for message in messages:
        role = str(message.get("role") or "user").strip() or "user"
        content = message.get("content")
        if isinstance(content, list):
            content_text = "\n".join(
                str(item.get("text", "")) if isinstance(item, dict) else str(item)
                for item in content
            )
        else:
            content_text = str(content or "")
        rendered.append(f"{role}: {content_text}")
    return "\n".join(rendered).strip()


def rollout_turns(rollout: dict[str, Any]) -> list[dict[str, Any]]:
    artifact = rollout.get("artifact")
    if isinstance(artifact, list):
        for entry in artifact:
            if isinstance(entry, dict) and isinstance(entry.get("turns"), list):
                return [turn for turn in entry["turns"] if isinstance(turn, dict)]
    trace = rollout.get("trace")
    if isinstance(trace, dict):
        inference = trace.get("inference")
        if isinstance(inference, dict) and isinstance(inference.get("turns"), list):
            return [turn for turn in inference["turns"] if isinstance(turn, dict)]
    return []


def is_rollout_payload(rollout: dict[str, Any]) -> bool:
    if not isinstance(rollout, dict):
        return False
    reward_info = rollout.get("reward_info")
    trace = rollout.get("trace")
    success_status = rollout.get("success_status")
    return (
        isinstance(reward_info, dict)
        and isinstance(trace, dict)
        and isinstance(success_status, str)
        and str(success_status).strip().lower() == "success"
    )


def rollout_outcome_reward(rollout: dict[str, Any]) -> float:
    reward_info = rollout.get("reward_info")
    if isinstance(reward_info, dict):
        outcome_objectives = reward_info.get("outcome_objectives")
        if isinstance(outcome_objectives, dict):
            for key in ("unique_achievements", "reward"):
                try:
                    value = outcome_objectives.get(key)
                    if value is not None:
                        return float(value)
                except (TypeError, ValueError):
                    pass
        details = reward_info.get("details")
        if isinstance(details, dict):
            achievements = details.get("achievements")
            if isinstance(achievements, list):
                unique = {str(item).strip() for item in achievements if str(item).strip()}
                if unique:
                    return float(len(unique))
        try:
            return float(reward_info.get("outcome_reward", 0.0))
        except (TypeError, ValueError):
            return 0.0
    return 0.0


def rollout_llm_call_count(rollout: dict[str, Any]) -> int:
    metadata = rollout.get("metadata")
    if isinstance(metadata, dict):
        try:
            value = metadata.get("llm_call_count")
            if value is not None:
                return int(value)
        except (TypeError, ValueError):
            pass
    reward_info = rollout.get("reward_info")
    if isinstance(reward_info, dict):
        details = reward_info.get("details")
        if isinstance(details, dict):
            try:
                value = details.get("llm_call_count")
                if value is not None:
                    return int(value)
            except (TypeError, ValueError):
                return 0
    return 0


def build_rollout_request(
    *,
    inference_url: str,
    model: str,
    api_key: str,
    seed: int,
    max_steps: int,
    trace_correlation_id: str,
    system_prompt: str,
    temperature: float = 0.0,
    max_tokens: int = 180,
    enable_thinking: bool = False,
    thinking_budget_tokens: int = 0,
    policy_version: str = "bootstrap",
    target_action_batch_size: int = 8,
    min_action_batch_size: int = 5,
    timeout_s: int = 45,
) -> dict[str, Any]:
    return {
        "trace_correlation_id": trace_correlation_id,
        "trial_id": trace_correlation_id,
        "env": {
            "seed": int(seed),
            "config": {
                "max_steps": int(max_steps),
                "episode_max_steps": int(max_steps),
            },
        },
        "policy": {
            "config": {
                "model": model,
                "api_key": api_key,
                "inference_url": inference_url,
                "temperature": float(temperature),
                "max_tokens": int(max_tokens),
                "system_prompt": system_prompt,
                "enable_thinking": bool(enable_thinking),
                "thinking_budget_tokens": int(thinking_budget_tokens),
                "use_tools": True,
                "policy_version": str(policy_version),
                "route": "teacher" if "teacher" in str(policy_version).lower() else "student",
                "target_action_batch_size": int(target_action_batch_size),
                "min_action_batch_size": int(min_action_batch_size),
                "timeout_s": int(timeout_s),
            }
        },
    }


async def collect_rollouts_concurrently_with_summary(
    *,
    container_url: str,
    container_worker_token: str = "",
    environment_api_key: str = "",
    inference_url: str,
    model: str,
    api_key: str,
    seeds: list[int],
    max_steps: int,
    system_prompt: str,
    temperature: float,
    max_tokens: int,
    enable_thinking: bool,
    thinking_budget_tokens: int = 0,
    policy_version: str,
    target_action_batch_size: int,
    min_action_batch_size: int,
    request_timeout_seconds: float,
    max_concurrent_rollouts: int,
    trace_prefix: str,
    rollout_concurrency: int | None = None,
    rollout_semaphore_limit: int | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    worker_count = max(
        1,
        int(rollout_concurrency if rollout_concurrency is not None else max_concurrent_rollouts),
    )
    permit_limit = max(
        1,
        int(rollout_semaphore_limit if rollout_semaphore_limit is not None else max_concurrent_rollouts),
    )
    semaphore = asyncio.Semaphore(permit_limit)
    timeout = httpx.Timeout(float(request_timeout_seconds), connect=min(30.0, float(request_timeout_seconds)))
    container_base = str(container_url).rstrip("/")
    headers = _container_headers(
        container_url=container_base,
        container_worker_token=container_worker_token,
        environment_api_key=environment_api_key,
    )
    is_quick_tunnel = _is_cloudflare_quick_tunnel_url(container_base)
    if is_quick_tunnel:
        headers = {**headers, "Connection": "close"}
    rollout_queue: asyncio.Queue[tuple[int, int] | None] = asyncio.Queue()
    for index, seed in enumerate(seeds):
        rollout_queue.put_nowait((index, int(seed)))
    results: list[dict[str, Any] | None] = [None] * len(seeds)
    request_latencies_s: list[float] = []
    active_rollouts = 0
    high_watermark = 0
    requests_started = 0
    requests_finished = 0
    started_at = time.perf_counter()

    async def _run_one(client: httpx.AsyncClient, seed: int, index: int) -> dict[str, Any]:
        request_body = build_rollout_request(
            inference_url=inference_url,
            model=model,
            api_key=api_key,
            seed=seed,
            max_steps=max_steps,
            trace_correlation_id=f"{trace_prefix}_{index:05d}_{seed}",
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            enable_thinking=enable_thinking,
            thinking_budget_tokens=thinking_budget_tokens,
            policy_version=policy_version,
            target_action_batch_size=target_action_batch_size,
            min_action_batch_size=min_action_batch_size,
            timeout_s=max(1, math.ceil(request_timeout_seconds)),
        )
        try:
            response = await client.post(
                f"{container_base}/rollout",
                headers=headers,
                json=request_body,
                follow_redirects=False,
            )
            deadline = time.perf_counter() + float(request_timeout_seconds)
            while True:
                if response.status_code == 303:
                    location = response.headers.get("location", "").strip()
                    if not location:
                        raise RuntimeError("rollout redirect missing Location header")
                    result_url = str(response.request.url.join(location))
                    response = await client.get(result_url, headers=headers, follow_redirects=False)
                    continue
                if response.status_code == 408 and "__modal_function_call_id=" in str(response.request.url):
                    if time.perf_counter() >= deadline:
                        raise RuntimeError(f"modal result url timed out: {response.request.url}")
                    await asyncio.sleep(1.0)
                    response = await client.get(str(response.request.url), headers=headers, follow_redirects=False)
                    continue
                break
            response.raise_for_status()
            payload = response.json()
            if isinstance(payload, dict):
                payload.setdefault("trace_correlation_id", request_body["trace_correlation_id"])
                payload.setdefault("trial_id", request_body["trial_id"])
                payload.setdefault("_request_seed", seed)
                return payload
            return {
                "error": "rollout response was not an object",
                "seed": seed,
                "trace_correlation_id": request_body["trace_correlation_id"],
            }
        except Exception as exc:
            error_text = str(exc).strip() or f"{type(exc).__name__}: no detail"
            return {
                "error": error_text,
                "seed": seed,
                "trace_correlation_id": request_body["trace_correlation_id"],
            }

    async def _worker(client: httpx.AsyncClient) -> None:
        nonlocal active_rollouts, high_watermark, requests_started, requests_finished
        while True:
            item = await rollout_queue.get()
            if item is None:
                rollout_queue.task_done()
                return
            index, seed = item
            async with semaphore:
                active_rollouts += 1
                high_watermark = max(high_watermark, active_rollouts)
                requests_started += 1
                request_started_at = time.perf_counter()
                try:
                    results[index] = await _run_one(client, seed, index)
                    if progress_callback is not None and isinstance(results[index], dict):
                        completed_rollouts = [item for item in results if isinstance(item, dict)]
                        valid_rollouts = [
                            item
                            for item in completed_rollouts
                            if not item.get("error") and is_rollout_payload(item)
                        ]
                        rewards = [rollout_outcome_reward(item) for item in valid_rollouts]
                        progress_callback(
                            {
                                "stage": "teacher_rollout_progress",
                                "requested_rollouts": len(seeds),
                                "completed_rollouts": len(completed_rollouts),
                                "num_structured_rollouts": len(valid_rollouts),
                                "num_errors": len(completed_rollouts) - len(valid_rollouts),
                                "active_rollouts": active_rollouts,
                                "rollout_requests_started": requests_started,
                                "rollout_requests_finished": requests_finished,
                                "mean_outcome_reward": mean(rewards) if rewards else 0.0,
                                "max_outcome_reward": max(rewards) if rewards else 0.0,
                                "latest_rollout": results[index],
                            }
                        )
                finally:
                    request_latencies_s.append(time.perf_counter() - request_started_at)
                    requests_finished += 1
                    active_rollouts -= 1
            rollout_queue.task_done()

    client_kwargs: dict[str, Any] = {"timeout": timeout}
    if is_quick_tunnel:
        client_kwargs.update(
            {
                "limits": httpx.Limits(
                    max_connections=max(worker_count, permit_limit),
                    max_keepalive_connections=0,
                ),
                "headers": {"Connection": "close"},
                "http2": False,
            }
        )

    async with httpx.AsyncClient(**client_kwargs) as client:
        workers = [asyncio.create_task(_worker(client)) for _ in range(worker_count)]
        await rollout_queue.join()
        for _worker_task in workers:
            rollout_queue.put_nowait(None)
        await asyncio.gather(*workers)

    completed_rollouts = [item for item in results if isinstance(item, dict)]
    valid_rollouts = [
        item for item in completed_rollouts if not item.get("error") and is_rollout_payload(item)
    ]
    rewards = [rollout_outcome_reward(item) for item in valid_rollouts]
    elapsed_s = max(time.perf_counter() - started_at, 1e-9)
    summary = {
        "requested_rollouts": len(seeds),
        "completed_rollouts": len(completed_rollouts),
        "num_errors": len(completed_rollouts) - len(valid_rollouts),
        "num_structured_rollouts": len(valid_rollouts),
        "mean_outcome_reward": mean(rewards) if rewards else 0.0,
        "max_outcome_reward": max(rewards) if rewards else 0.0,
        "elapsed_s": elapsed_s,
        "rollouts_per_minute": len(valid_rollouts) / (elapsed_s / 60.0),
        "rollout_concurrency": worker_count,
        "rollout_semaphore_limit": permit_limit,
        "rollout_requests_started": requests_started,
        "rollout_requests_finished": requests_finished,
        "active_rollout_high_watermark": high_watermark,
        "mean_request_latency_s": mean(request_latencies_s) if request_latencies_s else 0.0,
        "max_request_latency_s": max(request_latencies_s) if request_latencies_s else 0.0,
    }
    normalized_results = [item if isinstance(item, dict) else {"error": "missing rollout result"} for item in results]
    return normalized_results, summary


def reward_quantile_threshold(
    rewards: list[float],
    *,
    quantile: float = 0.75,
    minimum_threshold: float = 0.0,
) -> float:
    if not rewards:
        return float(minimum_threshold)
    ordered = sorted(float(item) for item in rewards)
    q = min(max(float(quantile), 0.0), 1.0)
    idx = min(len(ordered) - 1, max(0, math.ceil(q * len(ordered)) - 1))
    return max(float(minimum_threshold), float(ordered[idx]))


def rollout_achievements(rollout: dict[str, Any]) -> list[str]:
    metadata = rollout.get("metadata")
    if isinstance(metadata, dict):
        achievements = metadata.get("achievements")
        if isinstance(achievements, list):
            return [str(item).strip() for item in achievements if str(item).strip()]
    reward_info = rollout.get("reward_info")
    if isinstance(reward_info, dict):
        details = reward_info.get("details")
        if isinstance(details, dict):
            achievements = details.get("achievements")
            if isinstance(achievements, list):
                return [str(item).strip() for item in achievements if str(item).strip()]
    return []


def rollout_unique_achievement_count(rollout: dict[str, Any]) -> int:
    return len(set(rollout_achievements(rollout)))


def _normalize_achievement_names(values: list[str] | tuple[str, ...] | set[str] | None) -> list[str]:
    if not values:
        return []
    normalized: list[str] = []
    seen: set[str] = set()
    for value in values:
        name = str(value).strip()
        if not name or name in seen:
            continue
        seen.add(name)
        normalized.append(name)
    return normalized


def _resolve_allowed_teacher_achievements(*, teacher_cfg: dict[str, Any]) -> list[str]:
    raw_env = str(os.getenv("NANOHORIZON_ALLOWED_TEACHER_ACHIEVEMENTS") or "").strip()
    if raw_env:
        return _normalize_achievement_names(raw_env.split(","))
    if str(os.getenv("NANOHORIZON_FILTER_COLLECT_WOOD") or "").strip() in {"1", "true", "True"}:
        return ["collect_wood"]
    configured = teacher_cfg.get("allowed_achievements")
    if isinstance(configured, list):
        return _normalize_achievement_names(
            [str(item) for item in configured if str(item).strip()]
        )
    if isinstance(configured, str):
        return _normalize_achievement_names(configured.split(","))
    return []


def _resolve_min_teacher_unique_achievements(*, teacher_cfg: dict[str, Any]) -> int:
    raw_env = str(os.getenv("NANOHORIZON_MIN_TEACHER_UNIQUE_ACHIEVEMENTS") or "").strip()
    if raw_env:
        return max(0, int(raw_env))
    configured = teacher_cfg.get("min_unique_achievements")
    if configured is not None:
        return max(0, int(configured))
    return 0


def _resolve_teacher_priority_achievements(*, teacher_cfg: dict[str, Any]) -> list[str]:
    raw_env = str(os.getenv("NANOHORIZON_PRIORITY_TEACHER_ACHIEVEMENTS") or "").strip()
    if raw_env:
        return _normalize_achievement_names(raw_env.split(","))
    configured = teacher_cfg.get("priority_achievements")
    if isinstance(configured, list):
        return _normalize_achievement_names(
            [str(item) for item in configured if str(item).strip()]
        )
    if isinstance(configured, str):
        return _normalize_achievement_names(configured.split(","))
    return []


def build_openai_sft_rows_from_rollouts(
    rollouts: list[dict[str, Any]],
    *,
    reward_threshold: float,
    allowed_achievements: list[str] | None = None,
    min_unique_achievements: int = 0,
    priority_achievements: list[str] | None = None,
) -> list[dict[str, Any]]:
    allowed = set(_normalize_achievement_names(allowed_achievements))
    priority = set(_normalize_achievement_names(priority_achievements))
    rows: list[dict[str, Any]] = []
    for rollout in rollouts:
        outcome_reward = rollout_outcome_reward(rollout)
        if outcome_reward < float(reward_threshold):
            continue
        achievements = rollout_achievements(rollout)
        unique_achievements = set(achievements)
        unique_count = len(unique_achievements)
        if unique_count < int(min_unique_achievements):
            continue
        if allowed and not (set(achievements) & allowed):
            continue
        trace_correlation_id = str(rollout.get("trace_correlation_id") or "")
        rollout_id = str(rollout.get("rollout_id") or "")
        priority_hits = len(unique_achievements & priority)
        for turn in rollout_turns(rollout):
            prompt_messages = turn.get("prompt_messages")
            assistant_text = str(turn.get("assistant_text") or "").strip()
            reasoning_text = str(turn.get("reasoning_text") or "").strip()
            if not isinstance(prompt_messages, list):
                continue
            safe_prompt_messages = [item for item in prompt_messages if isinstance(item, dict)]
            actions = turn.get("actions")
            action_list = (
                [str(item).strip().lower() for item in actions if str(item).strip()]
                if isinstance(actions, list)
                else []
            )
            if not action_list:
                continue
            if not assistant_text and not reasoning_text:
                continue
            rows.append(
                {
                    "tools": [CRAFTAX_INTERACT_TOOL],
                    "messages": [
                        *safe_prompt_messages,
                        {
                            "role": "assistant",
                            "content": "",
                            "reasoning_content": reasoning_text or assistant_text,
                            "tool_calls": [
                                {
                                    "id": f"call_{trace_correlation_id or rollout_id or turn.get('turn_index') or 0}",
                                    "type": "function",
                                    "function": {
                                        "name": PRIMARY_TOOL_NAME,
                                        "arguments": {"actions_list": action_list},
                                    },
                                }
                            ],
                        },
                    ],
                    "metadata": {
                        "rollout_id": rollout_id,
                        "trace_correlation_id": trace_correlation_id,
                        "turn_index": int(turn.get("turn_index") or 0),
                        "decision_reward": float(turn.get("decision_reward") or 0.0),
                        "return_to_go": float(turn.get("return_to_go") or 0.0),
                        "outcome_reward": float(outcome_reward),
                        "unique_achievement_count": int(unique_count),
                        "priority_achievement_hits": int(priority_hits),
                        "assistant_text": assistant_text,
                        "actions": action_list,
                        "achievements": achievements,
                        "invalid_parse": bool(turn.get("invalid_parse")),
                        "trainable": bool(turn.get("trainable", True)),
                    },
                }
            )
    return rows


def summarize_rollouts(rollouts: list[dict[str, Any]]) -> dict[str, Any]:
    valid_rollouts = [
        item for item in rollouts if isinstance(item, dict) and not item.get("error") and is_rollout_payload(item)
    ]
    rewards = [rollout_outcome_reward(item) for item in valid_rollouts]
    llm_calls = [rollout_llm_call_count(item) for item in valid_rollouts]
    return {
        "num_rollouts": len(valid_rollouts),
        "num_errors": len(rollouts) - len(valid_rollouts),
        "mean_outcome_reward": mean(rewards) if rewards else 0.0,
        "max_outcome_reward": max(rewards) if rewards else 0.0,
        "mean_llm_calls_per_rollout": mean(llm_calls) if llm_calls else 0.0,
    }


def build_sft_examples(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for row in rows:
        messages = row.get("messages")
        if isinstance(messages, list) and messages:
            metadata = row.get("metadata", {})
            tools = row.get("tools")
            assistant_message = messages[-1] if len(messages) > 0 and isinstance(messages[-1], dict) else {}
            assistant_content = str(assistant_message.get("content") or "")
            assistant_reasoning = str(assistant_message.get("reasoning_content") or "")
            response_text = assistant_content
            examples.append(
                {
                    "prompt": flatten_messages(messages[:-1]) if len(messages) > 1 else "",
                    "prompt_messages": messages[:-1] if len(messages) > 1 else [],
                    "messages": messages,
                    "response": response_text or assistant_reasoning,
                    "weight": 1.0,
                    "metadata": metadata if isinstance(metadata, dict) else {},
                    "tools": tools if isinstance(tools, list) else [],
                }
            )
            continue
        prompt = str(row.get("prompt") or "")
        response = str(row.get("response") or "")
        if prompt and response:
            examples.append(
                {
                    "prompt": prompt,
                    "response": response,
                    "weight": 1.0,
                    "metadata": row.get("metadata", {}),
                    "tools": row.get("tools", []),
                }
            )
    filtered_examples: list[dict[str, Any]] = []
    for example in examples:
        if str(example.get("response") or "").strip():
            filtered_examples.append(example)
            continue
        messages = example.get("messages")
        if not isinstance(messages, list) or not messages:
            continue
        assistant_message = messages[-1] if isinstance(messages[-1], dict) else {}
        if isinstance(assistant_message.get("tool_calls"), list) and assistant_message["tool_calls"]:
            filtered_examples.append(example)
    return filtered_examples


def _example_repeat_count(example: dict[str, Any]) -> int:
    metadata = example.get("metadata", {})
    if not isinstance(metadata, dict):
        return 1
    try:
        outcome_reward = float(metadata.get("outcome_reward", 0.0) or 0.0)
    except (TypeError, ValueError):
        outcome_reward = 0.0
    try:
        return_to_go = float(metadata.get("return_to_go", 0.0) or 0.0)
    except (TypeError, ValueError):
        return_to_go = 0.0
    repeat_count = 1
    if outcome_reward >= 2.0:
        repeat_count += 1
    if return_to_go >= 1.0:
        repeat_count += 1
    return min(repeat_count, 3)


def rebalance_sft_examples(
    examples: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    expanded: list[dict[str, Any]] = []
    repeat_histogram: dict[str, int] = {}
    for example in examples:
        repeat_count = _example_repeat_count(example)
        repeat_histogram[str(repeat_count)] = repeat_histogram.get(str(repeat_count), 0) + 1
        for _ in range(repeat_count):
            expanded.append(example)
    summary = {
        "input_examples": len(examples),
        "expanded_examples": len(expanded),
        "repeat_histogram": repeat_histogram,
    }
    return expanded, summary


DEFAULT_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def _load_text_only_causal_lm(*, base_model: str, device: str, use_cache: bool = True) -> Any:
    import torch
    from transformers import AutoModelForCausalLM

    model_kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "torch_dtype": torch.bfloat16 if device == "cuda" else torch.float32,
    }
    if device == "cuda":
        model_kwargs["device_map"] = "auto"

    try:
        model = AutoModelForCausalLM.from_pretrained(base_model, **model_kwargs)
    except Exception:
        from transformers import Qwen3_5ForCausalLM

        model = Qwen3_5ForCausalLM.from_pretrained(base_model, **model_kwargs)

    model.config.use_cache = use_cache
    return model


@dataclass
class TrainResult:
    output_dir: str
    examples_seen: int
    optimizer_steps: int
    mean_loss: float
    total_tokens: int = 0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    mean_sequence_length: float = 0.0
    max_sequence_length: int = 0


def _tokenize_pair(tokenizer: Any, prompt: str, response: str, max_length: int) -> dict[str, Any]:
    import torch

    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    response_ids = tokenizer(response, add_special_tokens=False)["input_ids"]
    eos_id = tokenizer.eos_token_id
    input_ids = prompt_ids + response_ids + ([eos_id] if eos_id is not None else [])
    labels = ([-100] * len(prompt_ids)) + response_ids + ([eos_id] if eos_id is not None else [])
    input_ids = input_ids[:max_length]
    labels = labels[:max_length]
    attention_mask = [1] * len(input_ids)
    return {
        "input_ids": torch.tensor([input_ids], dtype=torch.long),
        "labels": torch.tensor([labels], dtype=torch.long),
        "attention_mask": torch.tensor([attention_mask], dtype=torch.long),
    }


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
                parsed_arguments = json.loads(arguments)
            except Exception:
                parsed_arguments = arguments
            arguments = parsed_arguments
        if not isinstance(arguments, (dict, list, str, int, float, bool)) and arguments is not None:
            arguments = str(arguments)
        normalized.append(
            {
                "id": str(item_dict.get("id", f"call_{index}") or f"call_{index}"),
                "type": "function",
                "function": {"name": name, "arguments": arguments},
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


def _render_prompt(tokenizer: Any, prompt: Any) -> str:
    if isinstance(prompt, list):
        messages = _normalize_messages_for_chat_template([item for item in prompt if isinstance(item, dict)])
        if hasattr(tokenizer, "apply_chat_template"):
            rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            if isinstance(rendered, str):
                return rendered
        rendered_lines = []
        for message in messages:
            rendered_lines.append(f"<|{message.get('role', 'user')}|>\n{message.get('content', '')}")
        rendered_lines.append("<|assistant|>")
        return "\n".join(rendered_lines)
    return str(prompt or "")


def _render_messages(tokenizer: Any, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> str:
    safe_messages = _normalize_messages_for_chat_template([item for item in messages if isinstance(item, dict)])
    if hasattr(tokenizer, "apply_chat_template"):
        rendered = tokenizer.apply_chat_template(
            safe_messages,
            tools=tools or None,
            tokenize=False,
            add_generation_prompt=False,
        )
        if isinstance(rendered, str):
            return rendered
    return _render_prompt(tokenizer, safe_messages)


def _tokenize_messages_with_assistant_mask(
    tokenizer: Any,
    prompt_messages: list[dict[str, Any]],
    full_messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    max_length: int,
) -> dict[str, Any]:
    import torch

    normalized_prompt_messages = _normalize_messages_for_chat_template(prompt_messages)
    normalized_full_messages = _normalize_messages_for_chat_template(full_messages)
    prompt_text = tokenizer.apply_chat_template(
        normalized_prompt_messages,
        tools=tools or None,
        tokenize=False,
        add_generation_prompt=True,
    )
    full_text = _render_messages(tokenizer, normalized_full_messages, tools)
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


def release_cuda_memory() -> None:
    import gc

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            with suppress(Exception):
                torch.cuda.ipc_collect()
    except Exception:
        pass


def train_sft_with_trl(
    *,
    base_model: str,
    examples: list[dict[str, Any]],
    output_dir: str | Path,
    learning_rate: float,
    epochs: int,
    max_length: int,
    max_steps: int,
    lora_rank: int,
    per_device_train_batch_size: int = 1,
    gradient_accumulation_steps: int = 1,
) -> TrainResult:
    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import AutoTokenizer, Trainer, TrainingArguments

    if not examples:
        raise ValueError("no examples provided for training")

    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = _load_text_only_causal_lm(base_model=base_model, device=device, use_cache=False)

    rows: list[dict[str, list[int]]] = []
    for example in examples:
        prompt = str(example.get("prompt") or "").strip()
        prompt_messages = example.get("prompt_messages")
        full_messages = example.get("messages")
        tools = example.get("tools") if isinstance(example.get("tools"), list) else []
        response = str(example.get("response") or "").strip()
        if isinstance(prompt_messages, list) and isinstance(full_messages, list) and full_messages:
            safe_tools = [item for item in tools if isinstance(item, dict)] if isinstance(tools, list) else []
            tokenized = _tokenize_messages_with_assistant_mask(
                tokenizer,
                [item for item in prompt_messages if isinstance(item, dict)],
                [item for item in full_messages if isinstance(item, dict)],
                safe_tools,
                max_length,
            )
        else:
            if not response:
                continue
            tokenized = _tokenize_pair(tokenizer, _render_prompt(tokenizer, prompt_messages or prompt), response, max_length)
        rows.append(
            {
                "input_ids": tokenized["input_ids"][0].tolist(),
                "labels": tokenized["labels"][0].tolist(),
                "attention_mask": tokenized["attention_mask"][0].tolist(),
            }
        )
    # Compute token volume stats
    total_tokens = sum(len(r["input_ids"]) for r in rows)
    total_prompt_tokens = sum(sum(1 for lbl in r["labels"] if lbl == -100) for r in rows)
    total_completion_tokens = total_tokens - total_prompt_tokens
    seq_lengths = [len(r["input_ids"]) for r in rows]
    mean_seq_len = mean(seq_lengths) if seq_lengths else 0.0
    max_seq_len = max(seq_lengths) if seq_lengths else 0

    dataset = Dataset.from_list(rows)

    peft_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_rank * 2,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=DEFAULT_TARGET_MODULES,
    )
    model = get_peft_model(model, peft_config)
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    def data_collator(features: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        max_feature_len = max(len(feature["input_ids"]) for feature in features)
        pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id or 0
        input_ids = []
        labels = []
        attention_mask = []
        for feature in features:
            pad_len = max_feature_len - len(feature["input_ids"])
            input_ids.append(feature["input_ids"] + ([pad_id] * pad_len))
            labels.append(feature["labels"] + ([-100] * pad_len))
            attention_mask.append(feature["attention_mask"] + ([0] * pad_len))
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }

    training_args = TrainingArguments(
        output_dir=str(destination),
        learning_rate=learning_rate,
        num_train_epochs=float(max(1, epochs)),
        max_steps=max_steps,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        logging_steps=1,
        save_strategy="no",
        report_to=[],
        bf16=torch.cuda.is_available(),
        fp16=False,
        remove_unused_columns=False,
        dataloader_num_workers=0,
        gradient_checkpointing=True,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )
    train_output = trainer.train()
    trainer.save_model(str(destination))
    tokenizer.save_pretrained(destination)
    del trainer
    del model
    release_cuda_memory()

    mean_loss = float(getattr(train_output, "training_loss", 0.0) or 0.0)
    if math.isnan(mean_loss):
        mean_loss = 0.0
    return TrainResult(
        output_dir=str(destination),
        examples_seen=len(rows),
        optimizer_steps=int(max_steps if max_steps > 0 else len(rows)),
        mean_loss=mean_loss,
        total_tokens=total_tokens,
        total_prompt_tokens=total_prompt_tokens,
        total_completion_tokens=total_completion_tokens,
        mean_sequence_length=mean_seq_len,
        max_sequence_length=max_seq_len,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NanoHorizon offline Craftax training")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", default="")
    return parser.parse_args()


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
        "You are a Craftax teacher policy.\n"
        f"You may think for up to about {thinking_budget_tokens} tokens before answering.\n"
        f"{action_instruction}\n"
        "Use movement to explore when nothing useful is adjacent.\n"
        "Use 'do' only when facing a useful nearby object or resource.\n"
        "Read the recent action history and avoid repeating unproductive loops.\n"
        f"Use the provided `{PRIMARY_TOOL_NAME}` tool exactly once for the final answer.\n"
        "Do not return plain text actions or JSON.\n"
        "Your final assistant action must be a tool call with valid full-Craftax actions."
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


def _row_priority(row: dict[str, Any]) -> tuple[float, float, float, float, float, int]:
    metadata = row.get("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
    outcome_reward = float(metadata.get("outcome_reward", 0.0) or 0.0)
    unique_achievement_count = float(metadata.get("unique_achievement_count", 0.0) or 0.0)
    priority_achievement_hits = float(metadata.get("priority_achievement_hits", 0.0) or 0.0)
    return_to_go = float(metadata.get("return_to_go", 0.0) or 0.0)
    decision_reward = float(metadata.get("decision_reward", 0.0) or 0.0)
    turn_index = int(metadata.get("turn_index", 0) or 0)
    return (outcome_reward, unique_achievement_count, priority_achievement_hits, return_to_go, decision_reward, -turn_index)


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
    allowed_teacher_achievements = _resolve_allowed_teacher_achievements(teacher_cfg=teacher_cfg)
    min_unique_achievements = _resolve_min_teacher_unique_achievements(teacher_cfg=teacher_cfg)
    priority_achievements = _resolve_teacher_priority_achievements(teacher_cfg=teacher_cfg)
    container_url = str(
        os.getenv("NANOHORIZON_CRAFTAX_CONTAINER_URL")
        or os.getenv("NANOHORIZON_CRAFTAX_CONTAINER_URL")
        or teacher_cfg.get("container_url")
        or os.getenv("NANOHORIZON_CONTAINER_URL")
        or "direct://local"
    ).strip()
    container_worker_token = str(
        os.getenv("NANOHORIZON_CRAFTAX_CONTAINER_WORKER_TOKEN")
        or os.getenv("NANOHORIZON_CRAFTAX_CONTAINER_WORKER_TOKEN")
        or teacher_cfg.get("container_worker_token")
        or ""
    ).strip()
    dataset_path = output_dir / "generated_sft_data.jsonl"
    rollouts_path = output_dir / "teacher_rollouts.jsonl"
    partial_rollouts_path = output_dir / "teacher_rollouts.partial.jsonl"
    progress_path = output_dir / "teacher_generation_progress.json"
    rollout_summary_path = output_dir / "teacher_rollout_summary.json"
    dataset_path.write_text("", encoding="utf-8")
    rollouts_path.write_text("", encoding="utf-8")
    partial_rollouts_path.write_text("", encoding="utf-8")

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

    target_action_batch_size = int(teacher_cfg.get("target_action_batch_size", 4))
    min_action_batch_size = int(teacher_cfg.get("min_action_batch_size", 3))
    write_json(
        progress_path,
        {
            "stage": "teacher_rollout_collection_started",
            "requested_rollouts": len(seeds),
            "completed_rollouts": 0,
            "num_structured_rollouts": 0,
            "num_errors": 0,
            "min_unique_achievements": min_unique_achievements,
            "priority_achievements": priority_achievements,
        },
    )

    def _on_rollout_progress(progress: dict[str, Any]) -> None:
        latest_rollout = progress.get("latest_rollout")
        if isinstance(latest_rollout, dict):
            with partial_rollouts_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(latest_rollout, sort_keys=True) + "\n")
        snapshot = {
            "stage": progress.get("stage", "teacher_rollout_progress"),
            "requested_rollouts": int(progress.get("requested_rollouts", len(seeds))),
            "completed_rollouts": int(progress.get("completed_rollouts", 0)),
            "num_structured_rollouts": int(progress.get("num_structured_rollouts", 0)),
            "num_errors": int(progress.get("num_errors", 0)),
            "active_rollouts": int(progress.get("active_rollouts", 0)),
            "rollout_requests_started": int(progress.get("rollout_requests_started", 0)),
            "rollout_requests_finished": int(progress.get("rollout_requests_finished", 0)),
            "mean_outcome_reward": float(progress.get("mean_outcome_reward", 0.0)),
            "max_outcome_reward": float(progress.get("max_outcome_reward", 0.0)),
            "min_unique_achievements": min_unique_achievements,
            "priority_achievements": priority_achievements,
        }
        write_json(progress_path, snapshot)
        print(json.dumps(snapshot, sort_keys=True), flush=True)

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
            model=str(os.getenv("NANOHORIZON_TEACHER_MODEL") or teacher_cfg["teacher_model"]),
            api_key=str(os.getenv("NANOHORIZON_TEACHER_API_KEY") or ""),
            seeds=seeds,
            max_steps=max_steps,
            system_prompt=_rollout_system_prompt(
                thinking_budget_tokens=thinking_budget_tokens,
                target_action_batch_size=target_action_batch_size,
                min_action_batch_size=min_action_batch_size,
            ),
            temperature=float(teacher_cfg.get("temperature", 0.2)),
            max_tokens=max_tokens,
            enable_thinking=bool(teacher_cfg.get("enable_thinking", True)),
            thinking_budget_tokens=thinking_budget_tokens,
            policy_version="teacher-rollout",
            target_action_batch_size=target_action_batch_size,
            min_action_batch_size=min_action_batch_size,
            request_timeout_seconds=request_timeout_seconds,
            max_concurrent_rollouts=rollout_concurrency,
            trace_prefix="nanohorizon_teacher",
            rollout_concurrency=rollout_concurrency,
            rollout_semaphore_limit=rollout_semaphore_limit,
            progress_callback=_on_rollout_progress,
        )
    )
    with rollouts_path.open("w", encoding="utf-8") as handle:
        for rollout in rollouts:
            handle.write(json.dumps(rollout, sort_keys=True) + "\n")

    successful_rollouts = [
        rollout
        for rollout in rollouts
        if isinstance(rollout, dict)
        and not rollout.get("error")
        and is_rollout_payload(rollout)
    ]
    failed_rollout_details = [
        str(rollout.get("status_detail") or rollout.get("error") or "").strip()
        for rollout in rollouts
        if isinstance(rollout, dict) and not is_rollout_payload(rollout)
    ]
    rewards = [rollout_outcome_reward(rollout) for rollout in successful_rollouts]
    threshold = reward_quantile_threshold(
        rewards,
        quantile=rollout_quantile,
        minimum_threshold=min_reward,
    )
    openai_rows = build_openai_sft_rows_from_rollouts(
        successful_rollouts,
        reward_threshold=threshold,
        allowed_achievements=allowed_teacher_achievements,
        min_unique_achievements=min_unique_achievements,
        priority_achievements=priority_achievements,
    )
    ranked_rows = _filter_rows_by_priority(openai_rows, keep_count=len(openai_rows))
    with dataset_path.open("w", encoding="utf-8") as handle:
        for row in ranked_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    rollout_summary = {
        **rollout_collection_summary,
        "reward_threshold": threshold,
        "accepted_rows": len(ranked_rows),
        "allowed_achievements": allowed_teacher_achievements,
        "min_unique_achievements": min_unique_achievements,
        "priority_achievements": priority_achievements,
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
            "allowed_achievements": allowed_teacher_achievements,
            "min_unique_achievements": min_unique_achievements,
            "priority_achievements": priority_achievements,
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
    if not successful_rollouts:
        sample_failures = [detail for detail in failed_rollout_details if detail][:3]
        failure_suffix = f": {' | '.join(sample_failures)}" if sample_failures else ""
        raise RuntimeError(
            "teacher rollout collection produced no successful rollout payloads"
            f"{failure_suffix}"
        )
    print(
        json.dumps(
            {
                "stage": "teacher_generation_complete",
                "successful_rollouts": len(successful_rollouts),
                "generated_rows": len(ranked_rows),
                "reward_threshold": threshold,
                "allowed_achievements": allowed_teacher_achievements,
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
    raw_pgid = os.getenv("NANOHORIZON_LOCAL_TEACHER_PGID", "").strip()
    pid: int | None = None
    pgid: int | None = None
    if raw_pid:
        try:
            pid = int(raw_pid)
        except ValueError:
            pid = None
    if raw_pgid:
        try:
            pgid = int(raw_pgid)
        except ValueError:
            pgid = None
    if pid is None and pgid is None:
        return
    if pgid is not None:
        try:
            os.killpg(pgid, signal.SIGTERM)
        except ProcessLookupError:
            pgid = None
        except OSError:
            pgid = None
    elif pid is not None:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pid = None
        except OSError:
            pid = None
    if pid is None and pgid is None:
        os.environ.pop("NANOHORIZON_LOCAL_TEACHER_PID", None)
        os.environ.pop("NANOHORIZON_LOCAL_TEACHER_PGID", None)
        return
    for _ in range(20):
        alive = False
        if pgid is not None:
            try:
                os.killpg(pgid, 0)
                alive = True
            except ProcessLookupError:
                pgid = None
            except OSError:
                pgid = None
        elif pid is not None:
            try:
                os.kill(pid, 0)
                alive = True
            except ProcessLookupError:
                pid = None
            except OSError:
                pid = None
        if not alive:
            break
        time.sleep(1.0)
    os.environ.pop("NANOHORIZON_LOCAL_TEACHER_PID", None)
    os.environ.pop("NANOHORIZON_LOCAL_TEACHER_PGID", None)


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
            total_tokens=int(result.total_tokens),
            total_prompt_tokens=int(result.total_prompt_tokens),
            total_completion_tokens=int(result.total_completion_tokens),
            mean_sequence_length=float(result.mean_sequence_length),
            max_sequence_length=int(result.max_sequence_length),
        )

    import modal

    app_name = os.getenv("NANOHORIZON_MODAL_SFT_APP_NAME", "nanohorizon-craftax-sft")
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
        total_tokens=int(payload.get("total_tokens") or 0),
        total_prompt_tokens=int(payload.get("total_prompt_tokens") or 0),
        total_completion_tokens=int(payload.get("total_completion_tokens") or 0),
        mean_sequence_length=float(payload.get("mean_sequence_length") or 0.0),
        max_sequence_length=int(payload.get("max_sequence_length") or 0),
    )


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    config_dir = Path(args.config).expanduser().resolve().parent
    output_dir = ensure_dir(args.output_dir or config["output"]["root_dir"])
    timer = Timer()

    teacher_generation = config.get("teacher_generation", {})
    teacher_enabled = bool(teacher_generation.get("enabled", False))
    dataset_override = str(os.getenv("NANOHORIZON_DATASET_JSONL_OVERRIDE") or "").strip()
    dataset_path = resolve_path(
        dataset_override or config["data"]["dataset_jsonl"],
        base_dir=config_dir,
    )
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

    row_filter_summary = {
        "input_rows": 0,
        "kept_rows": 0,
        "dropped_rows": 0,
        "require_trainable": False,
        "drop_invalid_parse": False,
        "dropped_non_trainable": 0,
        "dropped_invalid_parse": 0,
    }
    dataset_rows = read_jsonl(dataset_path)
    dataset_rows, row_filter_summary = filter_rows_for_training(
        dataset_rows,
        require_trainable=bool(filter_cfg.get("require_trainable", False)),
        drop_invalid_parse=bool(filter_cfg.get("drop_invalid_parse", False)),
    )
    write_json(output_dir / "row_filter_summary.json", row_filter_summary)
    print(json.dumps({"stage": "row_filter_complete", **row_filter_summary}, sort_keys=True), flush=True)

    examples = build_sft_examples(dataset_rows)
    training_cfg = config.get("training", {})
    training_method = str(training_cfg.get("method", "trl_sft") or "trl_sft").strip()
    if training_method == "reward_reweighted_sft":
        examples, rebalance_summary = rebalance_sft_examples(examples)
        write_json(output_dir / "rebalance_summary.json", rebalance_summary)
        print(json.dumps({"stage": "rebalance_complete", **rebalance_summary}, sort_keys=True), flush=True)
    print(
        json.dumps(
            {
                "stage": "sft_examples_ready",
                "dataset_path": str(dataset_path),
                "num_examples": len(examples),
                "training_method": training_method,
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
        os.getenv("NANOHORIZON_CRAFTAX_CONTAINER_URL")
        or os.getenv("NANOHORIZON_CRAFTAX_CONTAINER_URL")
        or evaluation_cfg.get("container_url")
        or os.getenv("NANOHORIZON_CONTAINER_URL")
        or config.get("teacher_generation", {}).get("container_url")
        or "direct://local"
    ).strip()
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
        evaluate_model = _import_evaluate_model()
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
        base_freqs = base_eval_summary.get("achievement_frequencies", {})
        ft_freqs = finetuned_eval_summary.get("achievement_frequencies", {})
        all_achievements = sorted(set(list(base_freqs.keys()) + list(ft_freqs.keys())))
        achievement_deltas = {}
        for ach in all_achievements:
            base_f = float((base_freqs.get(ach) or {}).get("frequency", 0.0) or 0.0)
            ft_f = float((ft_freqs.get(ach) or {}).get("frequency", 0.0) or 0.0)
            achievement_deltas[ach] = {
                "base_frequency": base_f,
                "finetuned_frequency": ft_f,
                "delta": round(ft_f - base_f, 4),
            }
        comparison_summary = {
            "base_mean_outcome_reward": base_eval_summary["mean_outcome_reward"],
            "finetuned_mean_outcome_reward": finetuned_eval_summary["mean_outcome_reward"],
            "reward_delta": finetuned_eval_summary["mean_outcome_reward"] - base_eval_summary["mean_outcome_reward"],
            "base_num_eval_rollouts": base_eval_summary["num_eval_rollouts"],
            "finetuned_num_eval_rollouts": finetuned_eval_summary["num_eval_rollouts"],
            "achievement_deltas": achievement_deltas,
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
        "training_method": training_method,
        "started_at": timer.started_at,
        "ended_at": timer.ended_at,
        "examples_seen": training_result.examples_seen,
        "optimizer_steps": training_result.optimizer_steps,
        "mean_loss": training_result.mean_loss,
        "elapsed_minutes": timer.elapsed_minutes,
        "dataset_jsonl": str(dataset_path),
        "row_filter_summary": row_filter_summary,
        "teacher_generation_enabled": teacher_enabled,
        "teacher_model": teacher_generation.get("teacher_model", ""),
        "min_unique_achievements": min_unique_achievements,
        "priority_achievements": priority_achievements,
        "training_backend": training_result.backend,
        "adapter_dir": training_result.adapter_dir,
        "training_output_dir": training_result.output_dir,
        "submission_mean_outcome_reward": finetuned_eval_summary.get(
            "mean_outcome_reward_over_requested_rollouts",
            finetuned_eval_summary["mean_outcome_reward"],
        ),
        "submission_achievement_frequencies": finetuned_eval_summary.get("achievement_frequencies", {}),
        "finetuned_num_eval_rollouts": finetuned_eval_summary["num_eval_rollouts"],
        "finetuned_mean_outcome_reward": finetuned_eval_summary["mean_outcome_reward"],
        "finetuned_achievement_frequencies": finetuned_eval_summary.get("achievement_frequencies", {}),
        "base_num_eval_rollouts": base_eval_summary["num_eval_rollouts"],
        "base_mean_outcome_reward": base_eval_summary["mean_outcome_reward"],
        "base_achievement_frequencies": base_eval_summary.get("achievement_frequencies", {}),
        "reward_delta": comparison_summary["reward_delta"],
        "achievement_deltas": comparison_summary.get("achievement_deltas", {}),
        "sft_token_volume": {
            "total_tokens": training_result.total_tokens,
            "total_prompt_tokens": training_result.total_prompt_tokens,
            "total_completion_tokens": training_result.total_completion_tokens,
            "mean_sequence_length": training_result.mean_sequence_length,
            "max_sequence_length": training_result.max_sequence_length,
        },
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


# Modal SFT entrypoint collapsed from former modal_sft.py
REMOTE_SRC = Path("/root/nanohorizon/src")
if REMOTE_SRC.exists():
    sys.path.insert(0, str(REMOTE_SRC))

try:
    import modal as _modal_lib
    from nanohorizon.shared.modal_common import (
        ARTIFACT_DIR,
        GPU_OFFLINE,
        REMOTE_ROOT,
        training_image,
        volume_mounts,
    )
    APP_NAME = os.getenv("NANOHORIZON_MODAL_SFT_APP_NAME", "nanohorizon-craftax-sft")
    app = _modal_lib.App(APP_NAME)
    image = training_image()
    _MODAL_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    _MODAL_AVAILABLE = False
    ARTIFACT_DIR = ""  # type: ignore[assignment]


def _default_output_dir() -> str:
    stamp = now_utc_iso().replace(":", "").replace("+00:00", "Z")
    return f"{ARTIFACT_DIR}/offline_sft/{stamp}"


if _MODAL_AVAILABLE:

    @app.function(
        image=image,
        gpu=GPU_OFFLINE,
        timeout=60 * 60,
        volumes=volume_mounts(),
    )
    def train_sft(
        *,
        base_model: str,
        examples: list[dict[str, Any]],
        output_dir: str = "",
        learning_rate: float = 5.0e-5,
        epochs: int = 1,
        max_length: int = 3072,
        max_steps: int = 16,
        lora_rank: int = 16,
        per_device_train_batch_size: int = 1,
        gradient_accumulation_steps: int = 1,
    ) -> dict[str, Any]:
        os.chdir(REMOTE_ROOT)
        destination = ensure_dir(output_dir or _default_output_dir())
        result = train_sft_with_trl(
            base_model=base_model,
            examples=examples,
            output_dir=destination / "adapter",
            learning_rate=learning_rate,
            epochs=epochs,
            max_length=max_length,
            max_steps=max_steps,
            lora_rank=lora_rank,
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
        )
        payload = {
            "output_dir": str(destination),
            "adapter_dir": str(destination / "adapter"),
            "examples_seen": int(result.examples_seen),
            "optimizer_steps": int(result.optimizer_steps),
            "mean_loss": float(result.mean_loss),
            "base_model": base_model,
            "total_tokens": int(result.total_tokens),
            "total_prompt_tokens": int(result.total_prompt_tokens),
            "total_completion_tokens": int(result.total_completion_tokens),
            "mean_sequence_length": float(result.mean_sequence_length),
            "max_sequence_length": int(result.max_sequence_length),
        }
        write_json(destination / "modal_sft_result.json", payload)
        return payload

    @app.local_entrypoint()
    def modal_train_sft_main(
        base_model: str = "Qwen/Qwen3.5-4B",
        examples_jsonl: str = "",
        output_dir: str = "",
        learning_rate: float = 5.0e-5,
        epochs: int = 1,
        max_length: int = 3072,
        max_steps: int = 16,
        lora_rank: int = 16,
        per_device_train_batch_size: int = 1,
        gradient_accumulation_steps: int = 1,
    ) -> None:
        examples: list[dict[str, Any]] = []
        if examples_jsonl:
            for raw in Path(examples_jsonl).expanduser().resolve().read_text(encoding="utf-8").splitlines():
                line = raw.strip()
                if not line:
                    continue
                payload = json.loads(line)
                if isinstance(payload, dict):
                    examples.append(payload)
        result: object = train_sft.remote(
            base_model=base_model,
            examples=examples,
            output_dir=output_dir,
            learning_rate=learning_rate,
            epochs=epochs,
            max_length=max_length,
            max_steps=max_steps,
            lora_rank=lora_rank,
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
