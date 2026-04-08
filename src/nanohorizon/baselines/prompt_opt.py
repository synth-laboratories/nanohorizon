import argparse
import asyncio
import json
import math
import os
import platform
import re
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from statistics import mean
from typing import Any, Literal, cast

import httpx
import modal
import yaml
from gepa import EvaluationBatch, GEPAAdapter, optimize

from nanohorizon.craftax_core.rollout import run_rollout_request
from nanohorizon.craftax_core.metadata import PRIMARY_TOOL_NAME
from nanohorizon.custom_vllm.runtime import build_thinking_budget_request_overrides
from nanohorizon.shared.craftax_data import summarize_achievement_frequencies
from nanohorizon.shared.openai_compat import create_chat_completion


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
        and bool(str(success_status).strip())
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
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    normalized_container_url = str(container_url or "").strip()
    if not normalized_container_url or normalized_container_url.lower().startswith("direct://"):
        worker_count = max(
            1,
            int(rollout_concurrency if rollout_concurrency is not None else max_concurrent_rollouts),
        )
        permit_limit = max(
            1,
            int(rollout_semaphore_limit if rollout_semaphore_limit is not None else max_concurrent_rollouts),
        )
        semaphore = asyncio.Semaphore(permit_limit)
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

        async def _run_one_direct(seed: int, index: int) -> dict[str, Any]:
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
                payload = await asyncio.to_thread(run_rollout_request, request_body)
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

        async def _worker_direct() -> None:
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
                        results[index] = await _run_one_direct(seed, index)
                    finally:
                        request_latencies_s.append(time.perf_counter() - request_started_at)
                        requests_finished += 1
                        active_rollouts -= 1
                rollout_queue.task_done()

        workers = [asyncio.create_task(_worker_direct()) for _ in range(worker_count)]
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
        normalized_results = [
            item if isinstance(item, dict) else {"error": "missing rollout result"} for item in results
        ]
        return normalized_results, summary

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
    container_base = normalized_container_url.rstrip("/")
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

TRACK_ID = "prompt_opt_1usd_gpt54_family"
TODO_SCRATCHPAD_REQUIREMENTS = [
    "Keep a tiny private todo list with exactly three items before the tool call.",
    "Use a compact internal todo list or scratchpad with the ordered slots danger, target resource, and loop-avoidance fallback.",
    "The three items must track (1) the immediate danger or blocker, (2) the next tile, object, or resource target, and (3) the loop-break or fallback progress action.",
    "Refresh completed todo items every turn.",
    "If the policy repeats the same movement pattern without progress or new information, replace the stale target item instead of continuing the loop.",
    "Dispatch the 3-4 action batch end-to-end from the current first todo item; only spend actions on lower-priority items after the first item is satisfied, blocked, or unsafe.",
    "Do not reveal the todo list or scratchpad in the final answer.",
]


def todo_scratchpad_directive() -> str:
    return " ".join(TODO_SCRATCHPAD_REQUIREMENTS)


REFLECTION_PROMPT_TEMPLATE = f"""I provided an assistant with the following Craftax system prompt:
```
<curr_param>
```

The following are examples of task inputs, model outputs, and feedback:
```
<side_info>
```

Write a revised Craftax system prompt.

Hard requirements you must preserve:
- The policy must think if needed, then use the `craftax_interact` tool exactly once.
- The final answer must not be plain text actions, JSON, or prose outside the tool call.
- The prompt must preserve this todo-tool contract: {todo_scratchpad_directive()}.
- The prompt should ask for a short valid full-Craftax action batch unless the episode is already done.
- The prompt should prioritize early-game resource gathering and avoid repeated movement loops.

Return only the revised system prompt inside ``` blocks."""


@dataclass(frozen=True)
class PromptOptExample:
    seed: int
    split: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GEPA-based Craftax prompt optimization baseline")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--container-url", default=os.getenv("NANOHORIZON_PROMPT_OPT_CONTAINER_URL", "direct://local"))
    parser.add_argument("--inference-url", required=True)
    parser.add_argument(
        "--inference-api-key",
        default=(
            os.getenv("NANOHORIZON_INFERENCE_API_KEY")
            or os.getenv("OPENAI_API_KEY")
            or ""
        ),
    )
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
            f"Keep the tool-calling contract strict: think if needed, then use the `{PRIMARY_TOOL_NAME}` tool exactly once with a short valid full-Craftax action batch. Preserve this todo-tool contract: {todo_scratchpad_directive()} Strengthen instructions about gathering nearby resources, using `do` only when adjacent to a useful target, and avoiding repeated no-op movement loops."
        )
        return " ".join(parts)
    parts = [f"This rollout achieved reward {score:.2f} and failed to make progress."]
    if invalid_parses:
        parts.append(
            f"It produced {invalid_parses} invalid parse(s); make the prompt stricter about one tool call only with 1-4 valid full-Craftax actions."
        )
    if llm_calls <= 1:
        parts.append(
            "The model had very few decision opportunities, so the prompt should encourage a short but useful macro-action that approaches a tree or other gatherable resource immediately."
        )
    if action_summary:
        parts.append(f"Observed action sequence: {action_summary}.")
    parts.append(
        f"Emphasize early-game progression: move toward trees, use `do` when adjacent, avoid sleep or crafting unless the inventory and local state justify it, and break out of repeated movement loops. Add this todo-tool contract before the final action choice: {todo_scratchpad_directive()} The final answer must be one `{PRIMARY_TOOL_NAME}` tool call, not a plain-text action list or JSON blob."
    )
    return " ".join(parts)


def build_reflection_system_directive() -> str:
    return (
        "You rewrite Craftax system prompts for a tool-calling policy. "
        "Preserve the compact internal todo list or scratchpad covering danger, target resource, and loop-avoidance. "
        f"Preserve these hard requirements: the policy must use the `{PRIMARY_TOOL_NAME}` "
        "tool exactly once, must not answer with JSON or a plain-text action list, and must "
        f"preserve this todo-tool contract: {todo_scratchpad_directive()} Return only the "
        "revised prompt text."
    )


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


class CraftaxPromptOptAdapter(GEPAAdapter[PromptOptExample, dict[str, Any], dict[str, Any]]):
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
                "content": build_reflection_system_directive(),
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
    adapter: CraftaxPromptOptAdapter,
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
        "requested_num_rollouts": len(dataset),
        "num_rollouts": len(reward_scores),
        "mean_outcome_reward": mean_reward,
        "mean_outcome_reward_over_requested_rollouts": mean_reward,
        "max_outcome_reward": max(reward_scores) if reward_scores else 0.0,
        "achievement_frequencies": summarize_achievement_frequencies(
            [
                output
                for output in batch.outputs
                if isinstance(output, dict)
            ],
            denominator=len(dataset),
        ),
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


def _chat_base_url_from_rollout_inference_url(url: str) -> str:
    normalized = str(url or "").strip()
    suffix = "/chat/completions"
    if normalized.endswith(suffix):
        return normalized[: -len(suffix)]
    return normalized




# Reflexion baseline implementation moved to `nanohorizon.baselines.reflexion`.



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
    adapter = CraftaxPromptOptAdapter(
        container_url=container_url,
        inference_url=rollout_inference_url,
        inference_api_key=inference_api_key,
        request_model=request_model,
        rollout_cfg=rollout_cfg,
    )
    algorithm = str(config.get("optimizer", {}).get("algorithm", "gepa")).strip().lower() or "gepa"
    if algorithm == "reflexion":
        from nanohorizon.baselines.reflexion import run_reflexion_baseline

        return run_reflexion_baseline(
            config=config,
            output_dir=output_dir,
            timer=timer,
            adapter=adapter,
            rollout_cfg=rollout_cfg,
            trainset=trainset,
            valset=valset,
            seed_prompt=seed_prompt,
            component_name=component_name,
            rollout_inference_url=rollout_inference_url,
            inference_api_key=inference_api_key,
            request_model=request_model,
        )
    reflection_model = str(config["optimizer"]["proposer_model"]).strip()
    reflection_lm, reflection_backend = _build_reflection_lm(
        requested_model=reflection_model,
        inference_url=inference_url,
        inference_api_key=inference_api_key,
        request_model=request_model,
        backend=str(config["optimizer"].get("reflection_backend", "auto")),
    )
    search_cfg = cast(dict[str, Any], config.get("search") or {})
    selector = cast(
        Literal["pareto", "current_best", "epsilon_greedy", "top_k_pareto"],
        str(search_cfg.get("candidate_selection_strategy", "current_best")),
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
        "baseline": "gepa_craftax_prompt_optimization",
        "status": "success",
        "submission_mean_outcome_reward": float(
            best_eval.get(
                "mean_outcome_reward_over_requested_rollouts",
                best_eval["mean_outcome_reward"],
            )
        ),
        "submission_achievement_frequencies": best_eval.get("achievement_frequencies", {}),
        "primary_score": float(best_eval["mean_outcome_reward"]),
        "primary_achievement_frequencies": best_eval.get("achievement_frequencies", {}),
        "bootstrap_score": float(base_eval["mean_outcome_reward"]),
        "bootstrap_achievement_frequencies": base_eval.get("achievement_frequencies", {}),
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
        "./scripts/run_craftax_prompt_opt_qwen35_4b_gpt54_budget.sh\n",
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
    inference_api_key = str(args.inference_api_key).strip() or str(
        os.getenv("NANOHORIZON_INFERENCE_API_KEY") or os.getenv("OPENAI_API_KEY") or ""
    ).strip()
    result = run_training(
        config_path=config_path,
        output_dir=output_dir,
        container_url=str(args.container_url).strip(),
        inference_url=str(args.inference_url).strip(),
        inference_api_key=inference_api_key,
        request_model=str(args.request_model).strip(),
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()


# Modal entrypoint collapsed from former modal_prompt_opt.py
REMOTE_SRC = Path("/root/nanohorizon/src")
if REMOTE_SRC.exists():
    sys.path.insert(0, str(REMOTE_SRC))

from nanohorizon.custom_vllm.runtime import (
    build_thinking_budget_request_overrides,
    enable_thinking_budget_support,
)
from nanohorizon.shared.modal_common import (
    CRAFTAX_PACKAGES,
    COMMON_PACKAGES,
    GPU_PROMPT_OPT,
    OFFLINE_VENV_ROOT,
    PROJECT_ROOT,
    RECORDS_DIR,
    REMOTE_ROOT,
    VLLM_BASE_IMAGE,
    VLLM_IMAGE_PYTHON_VERSION,
    _cuda_base_image,
    prompt_image,
    volume_mounts,
)

APP_NAME = "nanohorizon-craftax-prompt-opt"
CRAFTAX_PORT = 8903
VLLM_PORT = 8000
DEFAULT_REQUEST_TIMEOUT_S = 60 * 20
DEFAULT_MODEL = "Qwen/Qwen3.5-4B"
DEFAULT_SERVED_MODEL_NAME = "qwen35-4b-prompt-opt"
DEFAULT_API_KEY = "nanohorizon-prompt-opt-key"
DEFAULT_MAX_MODEL_LEN = 8192

app = modal.App(APP_NAME)


def _default_output_dir() -> str:
    stamp = now_utc_iso().replace(":", "").replace("+00:00", "Z")
    return f"{RECORDS_DIR}/prompt_opt_1usd_gpt54_family/{stamp}_reference_baseline"


def _pythonpath_with_repo() -> str:
    repo_src = f"{REMOTE_ROOT}/src"
    existing = str(os.environ.get("PYTHONPATH") or "").strip()
    if not existing:
        return repo_src
    parts = [repo_src, *[item for item in existing.split(os.pathsep) if item]]
    deduped: list[str] = []
    for part in parts:
        if part not in deduped:
            deduped.append(part)
    return os.pathsep.join(deduped)


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


def _resolve_local_config_path(config: str) -> Path:
    raw = Path(config).expanduser()
    if raw.is_absolute():
        return raw.resolve()
    return (PROJECT_ROOT / raw).resolve()


def _resolve_remote_config_path(config: str) -> Path:
    raw = Path(config)
    if raw.is_absolute():
        return raw.resolve()
    return (Path(REMOTE_ROOT) / raw).resolve()


def _config_arg_for_remote(config_path: Path) -> str:
    try:
        return config_path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError as exc:
        raise RuntimeError(
            f"prompt-opt config must live under the repo root: {config_path}"
        ) from exc


def _prompt_craftax_image() -> modal.Image:
    return (
        _cuda_base_image()
        .pip_install(*COMMON_PACKAGES, *CRAFTAX_PACKAGES)
        .add_local_dir(
            (PROJECT_ROOT / "src").as_posix(), remote_path=f"{REMOTE_ROOT}/src", copy=True
        )
        .add_local_dir(
            (PROJECT_ROOT / "scripts").as_posix(), remote_path=f"{REMOTE_ROOT}/scripts", copy=True
        )
        .add_local_dir(
            (PROJECT_ROOT / "configs").as_posix(), remote_path=f"{REMOTE_ROOT}/configs", copy=True
        )
        .add_local_dir(
            (PROJECT_ROOT / "data").as_posix(), remote_path=f"{REMOTE_ROOT}/data", copy=True
        )
    )


def _prompt_vllm_image() -> modal.Image:
    teacher_venv = f"{OFFLINE_VENV_ROOT}/teacher"
    return (
        modal.Image.from_registry(VLLM_BASE_IMAGE, add_python=VLLM_IMAGE_PYTHON_VERSION)
        .entrypoint([])
        .apt_install("curl")
        .pip_install(*COMMON_PACKAGES)
        .run_commands(
            f"python -m venv {teacher_venv}",
            f"{teacher_venv}/bin/python -m pip install --upgrade pip",
            f"{teacher_venv}/bin/python -m pip install "
            "\"httpx>=0.28.1\" \"pyyaml>=6.0.2\" \"vllm>=0.10.0\"",
        )
        .add_local_dir(
            (PROJECT_ROOT / "src").as_posix(), remote_path=f"{REMOTE_ROOT}/src", copy=True
        )
        .add_local_dir(
            (PROJECT_ROOT / "scripts").as_posix(), remote_path=f"{REMOTE_ROOT}/scripts", copy=True
        )
        .add_local_dir(
            (PROJECT_ROOT / "configs").as_posix(), remote_path=f"{REMOTE_ROOT}/configs", copy=True
        )
        .add_local_dir(
            (PROJECT_ROOT / "data").as_posix(), remote_path=f"{REMOTE_ROOT}/data", copy=True
        )
        .add_local_file(
            (PROJECT_ROOT / "pyproject.toml").as_posix(),
            remote_path=f"{REMOTE_ROOT}/pyproject.toml",
            copy=True,
        )
    )


if REMOTE_SRC.exists():
    prompt_runner_image = modal.Image.debian_slim(python_version="3.11")
    craftax_image = modal.Image.debian_slim(python_version="3.11")
    inference_image = modal.Image.debian_slim(python_version="3.11")
else:
    prompt_runner_image = prompt_image()
    craftax_image = _prompt_craftax_image()
    inference_image = _prompt_vllm_image()
openai_secret = modal.Secret.from_dict(
    {
        key: value
        for key, value in {
            "OPENAI_API_KEY": str(os.getenv("OPENAI_API_KEY") or "").strip(),
            "OPENAI_BASE_URL": str(os.getenv("OPENAI_BASE_URL") or "").strip(),
        }.items()
        if value
    }
)


def _wait_for_health(url: str) -> dict[str, Any]:
    deadline = time.time() + DEFAULT_REQUEST_TIMEOUT_S
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            with httpx.Client(timeout=10.0, follow_redirects=True) as client:
                response = client.get(url)
            response.raise_for_status()
            body = response.json() if response.content else {}
            if isinstance(body, dict):
                return body
            return {"status": "ok", "body": body}
        except Exception as exc:
            last_error = exc
            time.sleep(1.0)
    raise RuntimeError(f"timed out waiting for {url}: {last_error!r}")


def _wait_for_task_info(base_url: str) -> dict[str, Any]:
    deadline = time.time() + DEFAULT_REQUEST_TIMEOUT_S
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            with httpx.Client(timeout=10.0, follow_redirects=True) as client:
                response = client.get(f"{base_url.rstrip('/')}/task_info")
            response.raise_for_status()
            payload = response.json()
            if isinstance(payload, dict):
                return payload
        except Exception as exc:
            last_error = exc
        time.sleep(1.0)
    raise RuntimeError(f"timed out waiting for {base_url}/task_info: {last_error!r}")


def _probe_inference_chat(*, inference_base_url: str, api_key: str, model: str) -> dict[str, Any]:
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Reply with OK."}],
        "max_tokens": 1,
        "temperature": 0.0,
        **build_thinking_budget_request_overrides(enable_thinking=False, thinking_budget=0),
    }
    with httpx.Client(timeout=600.0, follow_redirects=True) as client:
        response = client.post(
            f"{inference_base_url.rstrip('/')}/v1/chat/completions",
            headers=headers,
            json=payload,
        )
    response.raise_for_status()
    body = response.json()
    return {
        "status": "ok",
        "model": body.get("model"),
        "choices": len(body.get("choices") or []),
    }


def _probe_container_roundtrip(
    *,
    container_url: str,
    inference_url: str,
    api_key: str,
    request_model: str,
    seed_prompt: str,
) -> dict[str, Any]:
    rollouts, summary = asyncio.run(
        collect_rollouts_concurrently_with_summary(
            container_url=container_url,
            inference_url=inference_url,
            model=request_model,
            api_key=api_key,
            seeds=[0],
            max_steps=2,
            system_prompt=seed_prompt,
            temperature=0.0,
            max_tokens=512,
            enable_thinking=True,
            thinking_budget_tokens=128,
            policy_version="prompt-opt-preflight",
            target_action_batch_size=2,
            min_action_batch_size=1,
            request_timeout_seconds=600.0,
            max_concurrent_rollouts=1,
            trace_prefix="prompt_opt_preflight",
            rollout_concurrency=1,
            rollout_semaphore_limit=1,
        )
    )
    rollout = rollouts[0] if rollouts else {}
    return {
        "summary": summary,
        "success_status": rollout.get("success_status") if isinstance(rollout, dict) else None,
        "reward": rollout.get("reward_info", {}).get("outcome_reward")
        if isinstance(rollout, dict) and isinstance(rollout.get("reward_info"), dict)
        else None,
    }


@app.cls(
    image=craftax_image,
    timeout=60 * 60 * 4,
    min_containers=1,
    max_containers=1,
    scaledown_window=60 * 10,
    volumes=volume_mounts(),
)
@modal.concurrent(max_inputs=32)
class CraftaxService:
    @modal.web_server(port=CRAFTAX_PORT, startup_timeout=60 * 10)
    def serve(self) -> None:
        env = {
            **os.environ,
            "PYTHONPATH": _pythonpath_with_repo(),
            "NANOHORIZON_CRAFTAX_BIND_HOST": "0.0.0.0",
            "NANOHORIZON_CRAFTAX_BIND_PORT": str(CRAFTAX_PORT),
        }
        cmd = [sys.executable, "-m", "nanohorizon.craftax_core.http_shim"]
        print("Launching Craftax service:", " ".join(shlex.quote(x) for x in cmd), flush=True)
        subprocess.Popen(cmd, env=env)


@app.cls(
    image=inference_image,
    gpu=GPU_PROMPT_OPT,
    timeout=60 * 60 * 4,
    scaledown_window=60 * 10,
    volumes=volume_mounts(),
)
@modal.concurrent(max_inputs=16)
class PromptOptInferenceServer:
    model: str = modal.parameter(default=DEFAULT_MODEL)
    served_model_name: str = modal.parameter(default=DEFAULT_SERVED_MODEL_NAME)
    api_key: str = modal.parameter(default=DEFAULT_API_KEY)
    max_model_len: int = modal.parameter(default=DEFAULT_MAX_MODEL_LEN)

    @modal.web_server(port=VLLM_PORT, startup_timeout=60 * 25)
    def serve(self) -> None:
        vllm_bin = f"{OFFLINE_VENV_ROOT}/teacher/bin/vllm"
        runtime_env = {
            **os.environ,
            "PYTHONUNBUFFERED": "1",
            "PYTHONPATH": _pythonpath_with_repo(),
        }
        cmd = [
            vllm_bin,
            "serve",
            self.model.strip() or DEFAULT_MODEL,
            "--served-model-name",
            self.served_model_name.strip() or DEFAULT_SERVED_MODEL_NAME,
            "--host",
            "0.0.0.0",
            "--port",
            str(VLLM_PORT),
            "--max-model-len",
            str(max(1024, int(self.max_model_len))),
            "--max-num-seqs",
            "32",
            "--gpu-memory-utilization",
            "0.9",
            "--uvicorn-log-level",
            "info",
            "--enable-prefix-caching",
            "--reasoning-parser",
            "qwen3",
            "--enforce-eager",
            "--enable-auto-tool-choice",
            "--tool-call-parser",
            "qwen3_coder",
            "--api-key",
            self.api_key.strip() or DEFAULT_API_KEY,
        ]
        cmd, env = enable_thinking_budget_support(
            cmd=cmd,
            env=runtime_env,
            model_ref=self.model.strip() or DEFAULT_MODEL,
        )
        print("Launching prompt-opt vLLM:", " ".join(shlex.quote(x) for x in cmd), flush=True)
        subprocess.Popen(cmd, env=env)


@app.function(
    image=prompt_runner_image,
    timeout=60 * 60 * 3,
    volumes=volume_mounts(),
    secrets=[openai_secret],
)
def run(
    *,
    config: str,
    output_dir: str,
    container_url: str,
    inference_url: str,
    inference_api_key: str,
    request_model: str,
    bootstrap_info: dict[str, Any],
) -> dict[str, Any]:
    os.chdir(REMOTE_ROOT)
    destination = ensure_dir(output_dir or _default_output_dir())
    write_json(destination / "bootstrap_info.json", bootstrap_info)
    return run_training(
        config_path=_resolve_remote_config_path(config),
        output_dir=destination,
        container_url=container_url,
        inference_url=inference_url,
        inference_api_key=inference_api_key,
        request_model=request_model,
    )


@app.local_entrypoint()
def modal_main(
    config: str = "configs/craftax_prompt_opt_qwen35_4b_gpt54_budget.yaml",
    output_dir: str = "",
) -> None:
    config_path = _resolve_local_config_path(config)
    config_payload = load_config(config_path)
    remote_config_arg = _config_arg_for_remote(config_path)
    resolved_output_dir = output_dir or _default_output_dir()
    model = str(config_payload["policy"]["model"]).strip() or DEFAULT_MODEL
    inference_api_key = (
        str(config_payload["policy"].get("inference_api_key", "")).strip() or DEFAULT_API_KEY
    )
    served_model_name = (
        str(config_payload["policy"].get("served_model_name", "")).strip()
        or DEFAULT_SERVED_MODEL_NAME
    )
    max_model_len = int(config_payload["policy"].get("max_model_len", DEFAULT_MAX_MODEL_LEN))
    craftax = cast(Any, CraftaxService)()
    container_url = craftax.serve.get_web_url()
    inference = cast(Any, PromptOptInferenceServer)(
        model=model,
        served_model_name=served_model_name,
        api_key=inference_api_key,
        max_model_len=max_model_len,
    )
    inference_url = inference.serve.get_web_url()
    rollout_inference_url = _normalize_inference_url(inference_url)
    print(
        json.dumps(
            {
                "stage": "prompt_opt_bootstrap_urls",
                "container_url": container_url,
                "inference_url": inference_url,
                "rollout_inference_url": rollout_inference_url,
                "request_model": served_model_name,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    seed_prompt = str(config_payload["prompt"]["seed_prompt"]).strip()
    bootstrap_info = {
        "container_url": container_url,
        "inference_url": inference_url,
        "rollout_inference_url": rollout_inference_url,
        "request_model": served_model_name,
        "preflights": {
            "craftax_health": (print('{"stage":"preflight_craftax_health"}', flush=True) or _wait_for_health(f"{container_url.rstrip('/')}/health")),
            "craftax_task_info": (print('{"stage":"preflight_craftax_task_info"}', flush=True) or _wait_for_task_info(container_url)),
            "inference_health": (print('{"stage":"preflight_inference_health"}', flush=True) or _wait_for_health(f"{inference_url.rstrip('/')}/health")),
            "inference_chat": (print('{"stage":"preflight_inference_chat"}', flush=True) or _probe_inference_chat(
                inference_base_url=inference_url,
                api_key=inference_api_key,
                model=served_model_name,
            )),
            "craftax_roundtrip": (print('{"stage":"preflight_craftax_roundtrip"}', flush=True) or _probe_container_roundtrip(
                container_url=container_url,
                inference_url=rollout_inference_url,
                api_key=inference_api_key,
                request_model=served_model_name,
                seed_prompt=seed_prompt,
            )),
        },
    }
    result = run.remote(
        config=remote_config_arg,
        output_dir=resolved_output_dir,
        container_url=container_url,
        inference_url=inference_url,
        inference_api_key=inference_api_key,
        request_model=served_model_name,
        bootstrap_info=bootstrap_info,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
