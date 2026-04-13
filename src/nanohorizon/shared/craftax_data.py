from __future__ import annotations

import asyncio
import math
import time
from collections.abc import Callable
from statistics import mean
from typing import Any

import httpx

from nanohorizon.craftax_core.metadata import (
    DEFAULT_ACHIEVEMENT_NAMES,
    DEFAULT_ACTION_NAMES,
    PRIMARY_TOOL_NAME,
)
from nanohorizon.craftax_core.rollout import run_rollout_request

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
                    "items": {
                        "type": "string",
                        "enum": DEFAULT_ACTION_NAMES,
                    },
                    "minItems": 1,
                    "maxItems": 10,
                }
            },
            "required": ["actions_list"],
            "additionalProperties": False,
        },
    },
}

CRAFTAX_CORE_ACHIEVEMENTS = list(DEFAULT_ACHIEVEMENT_NAMES)


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


def _is_modal_edge_url(url: str) -> bool:
    try:
        hostname = httpx.URL(url).host or ""
    except Exception:
        return False
    return hostname.endswith(".modal.run") or hostname.endswith(".w.modal.host")


def _container_base_urls(raw_url: str) -> list[str]:
    urls = [item.strip().rstrip("/") for item in str(raw_url or "").split(",") if item.strip()]
    return urls or [""]


def _container_headers(
    *,
    container_url: str,
    container_worker_token: str | None,
    environment_api_key: str | None,
) -> dict[str, str]:
    headers: dict[str, str] = {"Content-Type": "application/json"}
    if (
        _is_synthtunnel_url(container_url)
        or _is_cloudflare_quick_tunnel_url(container_url)
        or _is_modal_edge_url(container_url)
    ):
        headers["Connection"] = "close"
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
                unique = {
                    str(item).strip()
                    for item in achievements
                    if str(item).strip()
                }
                if unique:
                    return float(len(unique))
            try:
                native_reward_total = details.get("native_env_reward_total")
                if native_reward_total is not None:
                    return float(native_reward_total)
            except (TypeError, ValueError):
                pass
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


def summarize_achievement_frequencies(
    rollouts: list[dict[str, Any]],
    *,
    achievement_names: list[str] | None = None,
    denominator: int | None = None,
) -> dict[str, dict[str, float | int]]:
    names = list(achievement_names or CRAFTAX_CORE_ACHIEVEMENTS)
    counts = {name: 0 for name in names}
    valid_rollouts = [
        item for item in rollouts if isinstance(item, dict) and not item.get("error") and is_rollout_payload(item)
    ]
    total = max(0, int(denominator if denominator is not None else len(valid_rollouts)))
    for rollout in valid_rollouts:
        achieved = set(rollout_achievements(rollout))
        for name in names:
            if name in achieved:
                counts[name] += 1
    return {
        name: {
            "count": int(counts[name]),
            "frequency": (float(counts[name]) / float(total)) if total else 0.0,
        }
        for name in names
    }


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
    media: dict[str, Any] | None = None,
    request_logprobs: bool = True,
) -> dict[str, Any]:
    payload = {
        "trace_correlation_id": trace_correlation_id,
        "trial_id": trace_correlation_id,
        "env": {
            "seed": int(seed),
            "config": {
                "env_kind": "full",
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
                "request_logprobs": bool(request_logprobs),
            }
        },
    }
    if media:
        payload["media"] = media
    return payload


async def collect_rollouts_concurrently(
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
    video_capture_rollout_index: int | None = None,
    video_capture_output_dir: str = "",
    video_capture_fps: int = 6,
    video_capture_tile_size: int = 16,
    video_capture_show_status_bars: bool = True,
    request_logprobs: bool = True,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> list[dict[str, Any]]:
    rollouts, _summary = await collect_rollouts_concurrently_with_summary(
        container_url=container_url,
        container_worker_token=container_worker_token,
        environment_api_key=environment_api_key,
        inference_url=inference_url,
        model=model,
        api_key=api_key,
        seeds=seeds,
        max_steps=max_steps,
        system_prompt=system_prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        enable_thinking=enable_thinking,
        thinking_budget_tokens=thinking_budget_tokens,
        policy_version=policy_version,
        target_action_batch_size=target_action_batch_size,
        min_action_batch_size=min_action_batch_size,
        request_timeout_seconds=request_timeout_seconds,
        max_concurrent_rollouts=max_concurrent_rollouts,
        trace_prefix=trace_prefix,
        video_capture_rollout_index=video_capture_rollout_index,
        video_capture_output_dir=video_capture_output_dir,
        video_capture_fps=video_capture_fps,
        video_capture_tile_size=video_capture_tile_size,
        video_capture_show_status_bars=video_capture_show_status_bars,
        request_logprobs=request_logprobs,
        progress_callback=progress_callback,
    )
    return rollouts


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
    video_capture_rollout_index: int | None = None,
    video_capture_output_dir: str = "",
    video_capture_fps: int = 6,
    video_capture_tile_size: int = 16,
    video_capture_show_status_bars: bool = True,
    rollout_concurrency: int | None = None,
    rollout_semaphore_limit: int | None = None,
    request_logprobs: bool = True,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not str(container_url or "").strip() or str(container_url).strip().lower().startswith("direct://"):
        started_at = time.perf_counter()
        rollout_queue: asyncio.Queue[tuple[int, int] | None] = asyncio.Queue()
        for index, seed in enumerate(seeds):
            rollout_queue.put_nowait((index, int(seed)))
        results: list[dict[str, Any] | None] = [None] * len(seeds)
        request_latencies_s: list[float] = []
        active_rollouts = 0
        high_watermark = 0
        requests_started = 0
        requests_finished = 0
        semaphore = asyncio.Semaphore(max(1, int(rollout_semaphore_limit or max_concurrent_rollouts)))

        async def _run_one_direct(seed: int, index: int) -> dict[str, Any]:
            media = None
            if video_capture_rollout_index is not None and index == int(video_capture_rollout_index):
                media = {
                    "capture_video": True,
                    "fps": int(video_capture_fps),
                }
                if str(video_capture_output_dir).strip():
                    media["output_dir"] = str(video_capture_output_dir).strip()
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
                media=media,
                request_logprobs=request_logprobs,
            )
            try:
                payload = await asyncio.to_thread(run_rollout_request, request_body)
                payload.setdefault("trace_correlation_id", request_body["trace_correlation_id"])
                payload.setdefault("trial_id", request_body["trial_id"])
                payload.setdefault("_request_seed", seed)
                return payload
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
                        latest = results[index]
                        if progress_callback is not None and isinstance(latest, dict):
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
                                    "latest_rollout": latest,
                                }
                            )
                rollout_queue.task_done()

        workers = [
            asyncio.create_task(_worker_direct())
            for _ in range(max(1, int(rollout_concurrency or max_concurrent_rollouts)))
        ]
        await rollout_queue.join()
        for _ in workers:
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
            "rollout_concurrency": max(1, int(rollout_concurrency or max_concurrent_rollouts)),
            "rollout_semaphore_limit": max(1, int(rollout_semaphore_limit or max_concurrent_rollouts)),
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

    container_bases = _container_base_urls(container_url)
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
    headers_by_base = {
        base: _container_headers(
            container_url=base,
            container_worker_token=container_worker_token,
            environment_api_key=environment_api_key,
        )
        for base in container_bases
    }
    proxy_edge_bases = {
        base
        for base in container_bases
        if _is_cloudflare_quick_tunnel_url(base) or _is_modal_edge_url(base)
    }
    for base in proxy_edge_bases:
        headers_by_base[base] = {**headers_by_base[base], "Connection": "close"}
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
        container_base = container_bases[index % len(container_bases)]
        headers = headers_by_base[container_base]
        media = None
        if video_capture_rollout_index is not None and index == int(video_capture_rollout_index):
            media = {
                "capture_video": True,
                "fps": int(video_capture_fps),
                "tile_size": int(video_capture_tile_size),
                "show_status_bars": bool(video_capture_show_status_bars),
            }
            if str(video_capture_output_dir).strip():
                media["output_dir"] = str(video_capture_output_dir).strip()
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
            media=media,
            request_logprobs=request_logprobs,
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
                    response = await client.get(
                        str(response.request.url),
                        headers=headers,
                        follow_redirects=False,
                    )
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
                    latest = results[index]
                    if progress_callback is not None and isinstance(latest, dict):
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
                                "latest_rollout": latest,
                            }
                        )
            rollout_queue.task_done()

    client_kwargs: dict[str, Any] = {"timeout": timeout}
    if proxy_edge_bases:
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


def build_openai_sft_rows_from_rollouts(
    rollouts: list[dict[str, Any]],
    *,
    reward_threshold: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for rollout in rollouts:
        outcome_reward = rollout_outcome_reward(rollout)
        if outcome_reward < float(reward_threshold):
            continue
        trace_correlation_id = str(rollout.get("trace_correlation_id") or "")
        rollout_id = str(rollout.get("rollout_id") or "")
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
                                        "arguments": {
                                            "actions_list": action_list,
                                        },
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
                        "assistant_text": assistant_text,
                        "actions": action_list,
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


def build_rlvr_examples(rollouts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    rewards = [rollout_outcome_reward(rollout) for rollout in rollouts]
    min_reward = min(rewards) if rewards else 0.0
    max_reward = max(rewards) if rewards else 1.0
    denom = max(max_reward - min_reward, 1e-6)
    for rollout in rollouts:
        outcome_reward = rollout_outcome_reward(rollout)
        normalized = 0.1 + 0.9 * ((outcome_reward - min_reward) / denom)
        for turn in rollout_turns(rollout):
            messages = turn.get("prompt_messages")
            assistant_text = str(turn.get("assistant_text") or "")
            if not isinstance(messages, list) or not assistant_text.strip():
                continue
            examples.append(
                {
                    "prompt": flatten_messages(messages),
                    "prompt_messages": messages,
                    "response": assistant_text,
                    "weight": float(turn.get("decision_reward", normalized) or normalized),
                    "metadata": {
                        "rollout_id": rollout.get("rollout_id"),
                        "trace_correlation_id": rollout.get("trace_correlation_id"),
                        "outcome_reward": outcome_reward,
                    },
                }
            )
    return examples
