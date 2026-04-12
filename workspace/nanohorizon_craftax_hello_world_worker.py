#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
from typing import Any
from urllib import request


BASELINE_POLICY_PRESET = "baseline"
STATE_AWARE_POLICY_PRESET = "state_aware_macro_v1"

BASELINE_SYSTEM_PROMPT = (
    "You are a Craftax policy agent. Think carefully, then use the "
    "`craftax_interact` tool exactly once. Return exactly 5 valid full-Craftax "
    "actions unless the episode is already done. Use only the tool call as the "
    "final answer. Do not output JSON, prose, or a plain-text action list."
)

STATE_AWARE_SYSTEM_PROMPT = (
    "You are a Craftax policy agent. Think in short tactical macros, then use "
    "the `craftax_interact` tool exactly once. Keep a private three-item plan: "
    "(1) the immediate blocker or hazard, (2) the nearest useful resource or "
    "progression target, and (3) the fallback action that breaks a loop if "
    "progress stalls. Prefer the smallest batch that still makes visible "
    "progress: return 3 actions for danger or recovery cases, otherwise return "
    "4 actions. Move toward a useful resource or prerequisite if one is visible, "
    "use `do` only when adjacent to a useful target, avoid sleep/crafting/"
    "inventory-only actions unless the state clearly supports them, and do not "
    "repeat the same movement pattern without a new reason. Use only the tool "
    "call as the final answer. Do not output JSON, prose, or a plain-text action "
    "list."
)

POLICY_PRESETS: dict[str, dict[str, Any]] = {
    BASELINE_POLICY_PRESET: {
        "system_prompt": BASELINE_SYSTEM_PROMPT,
        "target_action_batch_size": 5,
        "min_action_batch_size": 5,
    },
    STATE_AWARE_POLICY_PRESET: {
        "system_prompt": STATE_AWARE_SYSTEM_PROMPT,
        "target_action_batch_size": 4,
        "min_action_batch_size": 3,
    },
}
DEFAULT_SEEDS = [1100 + idx for idx in range(10)]


def _resolve_container_url() -> str:
    explicit = str(os.getenv("NANOHORIZON_CRAFTAX_CONTAINER_URL") or "").strip()
    candidates = [explicit, "http://127.0.0.1:8913", "direct://local"]
    for candidate in candidates:
        if not candidate:
            continue
        if candidate.startswith("direct://"):
            return candidate
        try:
            with request.urlopen(f"{candidate.rstrip('/')}/health", timeout=3.0) as response:
                if 200 <= int(response.status) < 300:
                    return candidate
        except Exception:
            continue
    return "direct://local"


def _load_openai_api_key() -> str:
    direct = str(os.getenv("OPENAI_API_KEY") or "").strip()
    if direct:
        return direct
    candidate_paths = [
        Path("/Users/joshpurtell/Documents/GitHub/synth-ai/.env"),
        Path.home() / "Documents" / "GitHub" / "synth-ai" / ".env",
    ]
    for path in candidate_paths:
        if not path.exists():
            continue
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or not line.startswith("OPENAI_API_KEY="):
                continue
            _, _, value = line.partition("=")
            value = value.strip().strip("'").strip('"')
            if value:
                os.environ["OPENAI_API_KEY"] = value
                return value
    raise RuntimeError("OPENAI_API_KEY is required for the NanoHorizon Craftax hello-world baseline.")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _resolve_seed_slice() -> list[int]:
    raw = str(os.getenv("NANOHORIZON_CRAFTAX_WORKER_SEEDS") or "").strip()
    if not raw:
        return list(DEFAULT_SEEDS)
    seeds: list[int] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        seeds.append(int(token))
    return seeds or list(DEFAULT_SEEDS)


def _resolve_concurrency(default: int) -> int:
    raw = str(os.getenv("NANOHORIZON_CRAFTAX_WORKER_CONCURRENCY") or "").strip()
    if not raw:
        return int(default)
    return max(1, int(raw))


def _resolve_policy_preset() -> tuple[str, dict[str, Any]]:
    preset_name = str(os.getenv("NANOHORIZON_CRAFTAX_WORKER_POLICY") or "").strip() or STATE_AWARE_POLICY_PRESET
    preset = POLICY_PRESETS.get(preset_name, POLICY_PRESETS[STATE_AWARE_POLICY_PRESET])
    if preset_name not in POLICY_PRESETS:
        preset_name = STATE_AWARE_POLICY_PRESET
    return preset_name, preset


async def _run_eval() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    _load_openai_api_key()
    from nanohorizon.shared.craftax_data import (
        collect_rollouts_concurrently_with_summary,
        summarize_rollouts,
    )

    policy_preset_name, policy_preset = _resolve_policy_preset()
    seeds = _resolve_seed_slice()
    concurrency = _resolve_concurrency(default=10)
    container_url = _resolve_container_url()
    target_action_batch_size = int(policy_preset["target_action_batch_size"])
    min_action_batch_size = int(policy_preset["min_action_batch_size"])
    rollouts, rollout_summary = await collect_rollouts_concurrently_with_summary(
        container_url=container_url,
        container_worker_token="",
        environment_api_key="",
        inference_url="https://api.openai.com/v1/chat/completions",
        model="gpt-4.1-nano",
        api_key=os.environ["OPENAI_API_KEY"],
        seeds=seeds,
        max_steps=1,
        system_prompt=str(policy_preset["system_prompt"]),
        temperature=0.0,
        max_tokens=256,
        enable_thinking=False,
        thinking_budget_tokens=0,
        policy_version=policy_preset_name,
        target_action_batch_size=target_action_batch_size,
        min_action_batch_size=min_action_batch_size,
        request_timeout_seconds=45.0,
        max_concurrent_rollouts=concurrency,
        trace_prefix="nanohorizon_craftax_hello_world",
        rollout_concurrency=concurrency,
        rollout_semaphore_limit=concurrency,
        request_logprobs=False,
    )
    summary = summarize_rollouts(rollouts)
    summary.update(
        {
            "benchmark": "nanohorizon_craftax_hello_world",
            "task": "craftax",
            "model": "gpt-4.1-nano",
            "requested_rollouts": len(seeds),
            "requested_total_llm_calls": len(seeds),
            "requested_max_steps_per_rollout": 1,
            "requested_llm_calls_per_rollout": 1,
            "requested_rollout_seeds": list(seeds),
            "requested_rollout_concurrency": concurrency,
            "requested_max_concurrent_rollouts": concurrency,
            "requested_policy_preset": policy_preset_name,
            "requested_target_action_batch_size": target_action_batch_size,
            "requested_min_action_batch_size": min_action_batch_size,
            "selected_container_url": container_url,
            "rollout_concurrency": int(rollout_summary.get("rollout_concurrency", 10)),
            "rollout_summary": rollout_summary,
        }
    )
    return rollouts, summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the NanoHorizon Craftax hello-world worker.")
    parser.add_argument("--summary-output", required=True)
    parser.add_argument("--rollouts-output", required=True)
    args = parser.parse_args()

    rollouts, summary = asyncio.run(_run_eval())
    _write_json(Path(args.summary_output), summary)
    _write_jsonl(Path(args.rollouts_output), rollouts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
