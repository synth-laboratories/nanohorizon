#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any
from urllib import request

SYSTEM_PROMPT = (
    "You are a Craftax policy agent. Think carefully, then use the "
    "`craftax_interact` tool exactly once. Return exactly 5 valid full-Craftax "
    "actions unless the episode is already done. Use only the tool call as the "
    "final answer. Do not output JSON, prose, or a plain-text action list."
)
DEFAULT_ROLLOUT_COUNT = 10
DEFAULT_ROLLOUT_CONCURRENCY = 10
DEFAULT_REQUEST_TIMEOUT_SECONDS = 180.0
DEFAULT_POOL_API_TIMEOUT_SECONDS = 300.0


def _positive_int_env(name: str, default: int) -> int:
    raw = str(os.getenv(name) or "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return max(1, value)


def _rollout_seeds() -> list[int]:
    count = _positive_int_env("NANOHORIZON_ROLLOUTS", DEFAULT_ROLLOUT_COUNT)
    return [1100 + idx for idx in range(count)]


def _resolve_container_url() -> str:
    explicit = str(os.getenv("NANOHORIZON_CRAFTAX_CONTAINER_URL") or "").strip()
    allow_direct_local = str(os.getenv("NANOHORIZON_ALLOW_DIRECT_LOCAL") or "").strip() == "1"
    if not explicit:
        if allow_direct_local:
            return "direct://local"
        raise RuntimeError(
            "Craftax runtime resource was not selected. Start a local service "
            "inside the actor sandbox or deploy a Synth container-pool task, "
            "then set NANOHORIZON_CRAFTAX_CONTAINER_URL for this child command. "
            "For explicit same-actor local proof runs, set "
            "NANOHORIZON_ALLOW_DIRECT_LOCAL=1."
        )
    if explicit.startswith("direct://"):
        if allow_direct_local:
            return explicit
        raise RuntimeError("direct://local requires NANOHORIZON_ALLOW_DIRECT_LOCAL=1.")
    try:
        with request.urlopen(f"{explicit.rstrip('/')}/health", timeout=3.0) as response:
            if 200 <= int(response.status) < 300:
                return explicit
    except Exception as exc:
        raise RuntimeError(
            "Actor-managed Craftax HTTP resource did not pass /health at "
            f"{explicit!r}."
        ) from exc
    raise RuntimeError(
        "Actor-managed Craftax HTTP resource returned a non-2xx /health "
        f"response at {explicit!r}."
    )


def _load_inference_config() -> tuple[str, str, str]:
    openrouter_key = str(os.getenv("OPENROUTER_API_KEY") or "").strip()
    if openrouter_key:
        return (
            str(os.getenv("NANOHORIZON_INFERENCE_URL") or "https://openrouter.ai/api/v1/chat/completions"),
            str(os.getenv("NANOHORIZON_MODEL") or "x-ai/grok-4.1-fast"),
            openrouter_key,
        )
    base_url = str(os.getenv("OPENAI_BASE_URL") or "").strip().rstrip("/")
    direct = str(os.getenv("OPENAI_API_KEY") or "").strip()
    if direct:
        if base_url == "https://openrouter.ai/api/v1":
            return (
                str(os.getenv("NANOHORIZON_INFERENCE_URL") or f"{base_url}/chat/completions"),
                str(os.getenv("NANOHORIZON_MODEL") or "x-ai/grok-4.1-fast"),
                direct,
            )
        return (
            str(os.getenv("NANOHORIZON_INFERENCE_URL") or "https://api.openai.com/v1/chat/completions"),
            str(os.getenv("NANOHORIZON_MODEL") or "gpt-4.1-nano"),
            direct,
        )
    raise RuntimeError(
        "OPENAI_API_KEY or OPENROUTER_API_KEY must be materialized by the run orchestrator."
    )


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _media_output_dir() -> str:
    return str(os.getenv("NANOHORIZON_ROLLOUT_MEDIA_DIR") or Path("artifacts") / "rollout_media")


def _positive_float_env(name: str, default: float) -> float:
    raw = str(os.getenv(name) or "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    return max(1.0, value)


def _pool_fanout_enabled() -> bool:
    return str(os.getenv("NANOHORIZON_CRAFTAX_POOL_FANOUT") or "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _pool_image_ref() -> str:
    return str(os.getenv("NANOHORIZON_CRAFTAX_POOL_IMAGE_REF") or "synth-open-research-craftax:latest").strip()


def _backend_url() -> str | None:
    value = str(os.getenv("SYNTH_BACKEND_URL") or os.getenv("SYNTH_API_BASE_URL") or "").strip()
    return value or None


def _pool_api_timeout_seconds() -> float:
    request_timeout = _positive_float_env(
        "NANOHORIZON_REQUEST_TIMEOUT_SECONDS",
        DEFAULT_REQUEST_TIMEOUT_SECONDS,
    )
    default_timeout = max(DEFAULT_POOL_API_TIMEOUT_SECONDS, request_timeout + 120.0)
    return _positive_float_env("NANOHORIZON_POOL_API_TIMEOUT_SECONDS", default_timeout)


def _retry_pool_oom_sequentially() -> bool:
    return str(os.getenv("NANOHORIZON_CRAFTAX_RETRY_POOL_OOM_SEQUENTIALLY") or "1").strip().lower() not in {
        "0",
        "false",
        "no",
    }


def _build_rollout_request(
    *,
    inference_url: str,
    model: str,
    api_key: str,
    seed: int,
    index: int,
) -> dict[str, Any]:
    from nanohorizon.shared.craftax_data import build_rollout_request

    request_payload = build_rollout_request(
        inference_url=inference_url,
        model=model,
        api_key=api_key,
        seed=seed,
        max_steps=1,
        trace_correlation_id=f"nanohorizon_craftax_hello_world_{index:05d}_{seed}",
        system_prompt=SYSTEM_PROMPT,
        temperature=0.0,
        max_tokens=256,
        enable_thinking=False,
        thinking_budget_tokens=0,
        policy_version="hello_world_pool_fanout",
        target_action_batch_size=5,
        min_action_batch_size=5,
        timeout_s=int(_positive_float_env("NANOHORIZON_REQUEST_TIMEOUT_SECONDS", DEFAULT_REQUEST_TIMEOUT_SECONDS)),
        media={
            "capture_video": True,
            "fps": 6,
            "tile_size": 16,
            "write_mp4": False,
            "output_dir": "/tmp/craftax_rollout_media",
        },
        request_logprobs=False,
    )
    request_payload.pop("trial_id", None)
    request_payload.pop("media", None)
    return request_payload


def _create_pool(client: Any) -> tuple[str, dict[str, Any]]:
    explicit_pool_id = str(os.getenv("NANOHORIZON_CRAFTAX_POOL_ID") or "").strip()
    if explicit_pool_id:
        return explicit_pool_id, client.pools.get(explicit_pool_id)
    image_ref = _pool_image_ref()
    if not image_ref:
        raise RuntimeError("NANOHORIZON_CRAFTAX_POOL_IMAGE_REF is required for pool fanout.")
    request_payload = {
        "name": f"open-research-craftax-{int(time.time())}",
        "backend": "arbitrary",
        "provider": str(os.getenv("NANOHORIZON_CRAFTAX_POOL_PROVIDER") or "docker"),
        "runtime_kind": "image_ref",
        "interface_mode": "command_job",
        "image_ref": image_ref,
        "entrypoint": str(
            os.getenv("NANOHORIZON_CRAFTAX_POOL_ENTRYPOINT")
            or "python3 /work/workspace/craftax_pool_rollout_job.py"
        ),
        "input_path": "/tmp/rollout.json",
        "output_path": "/tmp/result.json",
        "workdir": "/work",
        "env_vars": {
            "PYTHONPATH": "/work/src",
            "JAX_DISABLE_JIT": "true",
            "JAX_PLATFORMS": "cpu",
            "JAX_PLATFORM_NAME": "cpu",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
            "XLA_FLAGS": "--xla_force_host_platform_device_count=1",
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
            "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.15",
        },
        "metadata": {
            "application_id": "craftax",
            "track": "craftax",
            "resource_id": "craftax_rollout_pool",
            "resource_owner": "open_research_actor",
        },
    }
    pool = client.pools.create(request_payload)
    pool_id = str(pool.get("id") or pool.get("pool_id") or "").strip()
    if not pool_id:
        raise RuntimeError(f"pool create response did not include id/pool_id: {pool}")
    return pool_id, pool


def _rollout_from_pool_result(result: dict[str, Any], *, seed: int) -> dict[str, Any]:
    rollout = result.get("rollout")
    if isinstance(rollout, dict):
        rollout.setdefault("_request_seed", seed)
        return rollout
    nested_result = result.get("result")
    if isinstance(nested_result, dict):
        nested_rollout = nested_result.get("rollout")
        if isinstance(nested_rollout, dict):
            nested_rollout.setdefault("_request_seed", seed)
            return nested_rollout
    execution_metadata = result.get("execution_metadata") if isinstance(result.get("execution_metadata"), dict) else {}
    if isinstance(nested_result, dict) and isinstance(nested_result.get("execution_metadata"), dict):
        execution_metadata = nested_result["execution_metadata"]
    error_message = result.get("error")
    if not error_message and isinstance(nested_result, dict):
        error_message = nested_result.get("error")
    if not error_message:
        error_message = execution_metadata.get("stderr")
    return {
        "error": str(error_message or "pool rollout did not return rollout payload"),
        "pool_return_code": execution_metadata.get("return_code"),
        "pool_stdout_preview": str(execution_metadata.get("stdout") or "")[:1000],
        "pool_stderr_preview": str(execution_metadata.get("stderr") or "")[:1000],
        "seed": seed,
        "trace_correlation_id": result.get("trace_correlation_id"),
    }


def _run_pool_fanout_eval() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from synth_ai import SynthClient
    from nanohorizon.shared.craftax_data import summarize_rollouts

    inference_url, model, api_key = _load_inference_config()
    seeds = _rollout_seeds()
    rollout_concurrency = min(
        len(seeds),
        _positive_int_env("NANOHORIZON_ROLLOUT_CONCURRENCY", DEFAULT_ROLLOUT_CONCURRENCY),
    )
    client = SynthClient(base_url=_backend_url(), timeout=_pool_api_timeout_seconds())
    pool_id, pool = _create_pool(client)
    started_at = time.perf_counter()
    results: list[dict[str, Any] | None] = [None] * len(seeds)
    requests_started = 0

    def _submit_one(index: int, seed: int) -> dict[str, Any]:
        request_payload = _build_rollout_request(
            inference_url=inference_url,
            model=model,
            api_key=api_key,
            seed=seed,
            index=index,
        )
        request_payload.update(
            {
                "pool_id": pool_id,
                "mode": "sync",
                "seed": int(seed),
                "metadata": {
                    "application_id": "craftax",
                    "track": "craftax",
                    "rollout_index": index,
                    "execution_topology": "synth_container_pool_fanout",
                },
            }
        )
        response = client.pools.rollouts.create(pool_id, request_payload)
        return _rollout_from_pool_result(response, seed=seed)

    with ThreadPoolExecutor(max_workers=max(1, rollout_concurrency)) as executor:
        future_to_index = {}
        for index, seed in enumerate(seeds):
            requests_started += 1
            future_to_index[executor.submit(_submit_one, index, int(seed))] = (index, int(seed))
        for future in as_completed(future_to_index):
            index, seed = future_to_index[future]
            try:
                results[index] = future.result()
            except Exception as exc:  # noqa: BLE001
                results[index] = {
                    "error": str(exc).strip() or f"{type(exc).__name__}: no detail",
                    "seed": seed,
                    "trace_correlation_id": f"nanohorizon_craftax_hello_world_{index:05d}_{seed}",
                }

    sequential_retries = 0
    if _retry_pool_oom_sequentially():
        for index, seed in enumerate(seeds):
            current = results[index]
            if not isinstance(current, dict) or not current.get("error"):
                continue
            if int(current.get("pool_return_code") or 0) != 137:
                continue
            sequential_retries += 1
            requests_started += 1
            try:
                results[index] = _submit_one(index, int(seed))
            except Exception as exc:  # noqa: BLE001
                results[index] = {
                    "error": str(exc).strip() or f"{type(exc).__name__}: no detail",
                    "seed": int(seed),
                    "trace_correlation_id": f"nanohorizon_craftax_hello_world_{index:05d}_{int(seed)}",
                }

    rollouts = [item if isinstance(item, dict) else {"error": "missing pool rollout result"} for item in results]
    summary = summarize_rollouts(rollouts)
    elapsed_s = max(time.perf_counter() - started_at, 1e-9)
    summary.update(
        {
            "benchmark": "nanohorizon_craftax_hello_world",
            "task": "craftax",
            "model": model,
            "requested_rollouts": len(seeds),
            "requested_total_llm_calls": len(seeds),
            "requested_max_steps_per_rollout": 1,
            "requested_llm_calls_per_rollout": 1,
            "requested_rollout_seeds": seeds,
            "requested_rollout_concurrency": rollout_concurrency,
            "selected_container_url": "",
            "execution_topology": "synth_container_pool_fanout",
            "pool_id": pool_id,
            "pool": pool,
            "rollout_concurrency": rollout_concurrency,
            "rollout_summary": {
                "requested_rollouts": len(seeds),
                "completed_rollouts": len(rollouts),
                "num_errors": summary["num_errors"],
                "num_structured_rollouts": summary["num_rollouts"],
                "elapsed_s": elapsed_s,
                "rollout_concurrency": rollout_concurrency,
                "rollout_semaphore_limit": rollout_concurrency,
                "rollout_requests_started": requests_started,
                "rollout_requests_finished": len(rollouts),
                "active_rollout_high_watermark": rollout_concurrency,
                "sequential_oom_retries": sequential_retries,
                "mean_outcome_reward": summary["mean_outcome_reward"],
                "max_outcome_reward": summary["max_outcome_reward"],
            },
        }
    )
    return rollouts, summary


async def _run_eval() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if _pool_fanout_enabled():
        return _run_pool_fanout_eval()

    inference_url, model, api_key = _load_inference_config()
    from nanohorizon.shared.craftax_data import (
        collect_rollouts_concurrently_with_summary,
        summarize_rollouts,
    )

    container_url = _resolve_container_url()
    seeds = _rollout_seeds()
    rollout_concurrency = min(
        len(seeds),
        _positive_int_env("NANOHORIZON_ROLLOUT_CONCURRENCY", DEFAULT_ROLLOUT_CONCURRENCY),
    )
    rollouts, rollout_summary = await collect_rollouts_concurrently_with_summary(
        container_url=container_url,
        container_worker_token="",
        environment_api_key="",
        inference_url=inference_url,
        model=model,
        api_key=api_key,
        seeds=seeds,
        max_steps=1,
        system_prompt=SYSTEM_PROMPT,
        temperature=0.0,
        max_tokens=256,
        enable_thinking=False,
        thinking_budget_tokens=0,
        policy_version="hello_world",
        target_action_batch_size=5,
        min_action_batch_size=5,
        request_timeout_seconds=_positive_float_env(
            "NANOHORIZON_REQUEST_TIMEOUT_SECONDS",
            DEFAULT_REQUEST_TIMEOUT_SECONDS,
        ),
        max_concurrent_rollouts=rollout_concurrency,
        trace_prefix="nanohorizon_craftax_hello_world",
        video_capture_output_dir=_media_output_dir(),
        video_capture_all_rollouts=True,
        rollout_concurrency=rollout_concurrency,
        rollout_semaphore_limit=rollout_concurrency,
        request_logprobs=False,
    )
    summary = summarize_rollouts(rollouts)
    summary.update(
        {
            "benchmark": "nanohorizon_craftax_hello_world",
            "task": "craftax",
            "model": model,
            "requested_rollouts": len(seeds),
            "requested_total_llm_calls": len(seeds),
            "requested_max_steps_per_rollout": 1,
            "requested_llm_calls_per_rollout": 1,
            "requested_rollout_seeds": seeds,
            "requested_rollout_concurrency": rollout_concurrency,
            "selected_container_url": container_url,
            "rollout_concurrency": int(rollout_summary.get("rollout_concurrency", rollout_concurrency)),
            "rollout_summary": rollout_summary,
        }
    )
    return rollouts, summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the NanoHorizon Craftax hello-world baseline worker.")
    parser.add_argument("--summary-output", required=True)
    parser.add_argument("--rollouts-output", required=True)
    args = parser.parse_args()

    rollouts, summary = asyncio.run(_run_eval())
    _write_json(Path(args.summary_output), summary)
    _write_jsonl(Path(args.rollouts_output), rollouts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
