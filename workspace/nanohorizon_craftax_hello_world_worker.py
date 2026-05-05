#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import os
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


async def _run_eval() -> tuple[list[dict[str, Any]], dict[str, Any]]:
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
        request_timeout_seconds=45.0,
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
