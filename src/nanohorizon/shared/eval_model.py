from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
from typing import Any

from nanohorizon.craftax_core.metadata import craftax_system_prompt
from nanohorizon.shared.common import ensure_dir, write_json, write_text
from nanohorizon.shared.craftax_data import (
    CRAFTAX_CORE_ACHIEVEMENTS,
    collect_rollouts_concurrently,
    is_rollout_payload,
    rollout_achievements,
    rollout_llm_call_count,
    rollout_outcome_reward,
    summarize_achievement_frequencies,
)
from nanohorizon.shared.train_lora import release_cuda_memory
from nanohorizon.shared.vllm_eval import LocalVLLMEvalConfig, local_vllm_server


def reward_heuristic(user_observation: str, assistant_text: str) -> float:
    del user_observation, assistant_text
    return 0.0


def evaluate_model(
    *,
    base_model: str,
    output_dir: str | Path,
    container_url: str,
    seed_start: int,
    num_rollouts: int,
    max_steps: int,
    max_concurrent_rollouts: int,
    max_length: int,
    max_new_tokens: int,
    adapter_dir: str | Path | None = None,
    summary_name: str = "eval_summary.json",
    thinking_budget_tokens: int = 512,
    enable_thinking: bool = False,
    enforce_eager: bool = False,
    temperature: float = 0.0,
    system_prompt: str = "",
    inference_url: str = "",
    inference_api_key: str = "",
    request_model: str = "",
    request_timeout_seconds: float = 300.0,
    video_capture_rollout_index: int | None = None,
    video_capture_output_dir: str = "",
    video_capture_fps: int = 6,
    video_capture_tile_size: int = 16,
    video_capture_show_status_bars: bool = True,
) -> dict[str, Any]:
    out_dir = ensure_dir(output_dir)
    release_cuda_memory()
    resolved_container_url = str(
        container_url
        or os.getenv("NANOHORIZON_CRAFTAX_CONTAINER_URL")
        or os.getenv("NANOHORIZON_CRAFTAX_CONTAINER_URL")
        or os.getenv("NANOHORIZON_CONTAINER_URL")
        or "direct://local"
    ).strip()
    resolved_container_worker_token = str(
        os.getenv("NANOHORIZON_CRAFTAX_CONTAINER_WORKER_TOKEN")
        or os.getenv("NANOHORIZON_CRAFTAX_CONTAINER_WORKER_TOKEN")
        or ""
    ).strip()
    if not resolved_container_worker_token:
        resolved_container_worker_token = str(
        os.getenv("NANOHORIZON_CRAFTAX_CONTAINER_WORKER_TOKEN")
        or ""
        ).strip()
    resolved_inference_url = str(inference_url or "").strip()
    resolved_inference_api_key = str(
        inference_api_key or os.getenv("NANOHORIZON_EVAL_API_KEY", "")
    ).strip()
    lora_name = ""
    lora_path = ""
    resolved_request_model = str(request_model or "").strip() or base_model
    max_lora_rank = 16
    if adapter_dir is not None:
        from nanohorizon.shared.vllm_eval import infer_lora_rank

        lora_name = "policy-lora"
        lora_path = str(Path(adapter_dir).expanduser().resolve())
        if not resolved_inference_url:
            resolved_request_model = lora_name
        max_lora_rank = infer_lora_rank(lora_path)

    seeds = [int(seed_start) + idx for idx in range(max(1, int(num_rollouts)))]
    resolved_system_prompt = craftax_system_prompt(
        system_prompt or "You are a Craftax policy.",
        thinking_budget_tokens=thinking_budget_tokens,
    )
    if resolved_inference_url:
        rollouts = asyncio.run(
            collect_rollouts_concurrently(
                container_url=resolved_container_url,
                container_worker_token=resolved_container_worker_token,
                inference_url=resolved_inference_url,
                model=resolved_request_model,
                api_key=resolved_inference_api_key,
                seeds=seeds,
                max_steps=max_steps,
                system_prompt=resolved_system_prompt,
                temperature=float(temperature),
                max_tokens=max_new_tokens,
                enable_thinking=enable_thinking,
                thinking_budget_tokens=thinking_budget_tokens,
                policy_version="finetuned-eval" if adapter_dir is not None else "base-eval",
                target_action_batch_size=8,
                min_action_batch_size=5,
                request_timeout_seconds=float(request_timeout_seconds),
                max_concurrent_rollouts=max_concurrent_rollouts,
                trace_prefix=summary_name.removesuffix(".json"),
                video_capture_rollout_index=video_capture_rollout_index,
                video_capture_output_dir=video_capture_output_dir,
                video_capture_fps=video_capture_fps,
                video_capture_tile_size=video_capture_tile_size,
                video_capture_show_status_bars=video_capture_show_status_bars,
            )
        )
    else:
        config = LocalVLLMEvalConfig(
            model=base_model,
            served_model_name=base_model,
            lora_name=lora_name,
            lora_path=lora_path,
            max_lora_rank=max_lora_rank,
            max_model_len=max_length,
            max_new_tokens=max_new_tokens,
            enable_thinking=enable_thinking,
            enforce_eager=enforce_eager,
        )
        with local_vllm_server(
            config=config,
            log_path=out_dir / f"{summary_name.removesuffix('.json')}_vllm_eval_server.log",
        ) as server:
            rollouts = asyncio.run(
                collect_rollouts_concurrently(
                    container_url=resolved_container_url,
                    container_worker_token=resolved_container_worker_token,
                    inference_url=f"{str(server['base_url']).rstrip('/')}/chat/completions",
                    model=resolved_request_model,
                    api_key=resolved_inference_api_key,
                    seeds=seeds,
                    max_steps=max_steps,
                    system_prompt=resolved_system_prompt,
                    temperature=float(temperature),
                    max_tokens=max_new_tokens,
                    enable_thinking=enable_thinking,
                    thinking_budget_tokens=thinking_budget_tokens,
                    policy_version="finetuned-eval" if adapter_dir is not None else "base-eval",
                    target_action_batch_size=8,
                    min_action_batch_size=5,
                    request_timeout_seconds=float(request_timeout_seconds),
                    max_concurrent_rollouts=max_concurrent_rollouts,
                    trace_prefix=summary_name.removesuffix(".json"),
                    video_capture_rollout_index=video_capture_rollout_index,
                    video_capture_output_dir=video_capture_output_dir,
                    video_capture_fps=video_capture_fps,
                    video_capture_tile_size=video_capture_tile_size,
                    video_capture_show_status_bars=video_capture_show_status_bars,
                )
            )

    valid_rollouts = [
        item
        for item in rollouts
        if isinstance(item, dict) and not item.get("error") and is_rollout_payload(item)
    ]
    requested_rollout_count = len(rollouts)
    rewards = [rollout_outcome_reward(item) for item in valid_rollouts]
    llm_calls = [rollout_llm_call_count(item) for item in valid_rollouts]
    achievement_frequencies = summarize_achievement_frequencies(
        rollouts,
        denominator=requested_rollout_count,
    )
    result = {
        "requested_num_eval_rollouts": requested_rollout_count,
        "num_eval_rollouts": len(valid_rollouts),
        "num_rollout_errors": len(rollouts) - len(valid_rollouts),
        "mean_outcome_reward": (sum(rewards) / len(rewards)) if rewards else 0.0,
        "mean_unique_achievements": (sum(rewards) / len(rewards)) if rewards else 0.0,
        "mean_outcome_reward_over_requested_rollouts": (
            sum(rewards) / float(requested_rollout_count)
        ) if requested_rollout_count else 0.0,
        "mean_unique_achievements_over_requested_rollouts": (
            sum(rewards) / float(requested_rollout_count)
        ) if requested_rollout_count else 0.0,
        "max_outcome_reward": max(rewards) if rewards else 0.0,
        "mean_llm_calls_per_rollout": (sum(llm_calls) / len(llm_calls)) if llm_calls else 0.0,
        "achievement_names": list(CRAFTAX_CORE_ACHIEVEMENTS),
        "achievement_frequencies": achievement_frequencies,
        "inference_backend": "remote_openai_compat" if resolved_inference_url else "vllm",
        "inference_url": resolved_inference_url,
        "request_model": resolved_request_model,
        "enable_thinking": bool(enable_thinking),
        "thinking_budget_tokens": int(thinking_budget_tokens),
        "details": [
            {
                "seed": int(item.get("_request_seed") or 0),
                "rollout_id": str(item.get("rollout_id") or ""),
                "trace_correlation_id": str(item.get("trace_correlation_id") or ""),
                "outcome_reward": rollout_outcome_reward(item),
                "llm_call_count": rollout_llm_call_count(item),
                "achievements": rollout_achievements(item),
                "unique_achievement_count": len(rollout_achievements(item)),
                "success_status": item.get("success_status"),
                "error": item.get("error"),
            }
            for item in rollouts
        ],
    }
    write_json(out_dir / summary_name, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a Craftax model with concurrent real rollouts.")
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter-dir", default="")
    parser.add_argument("--container-url", default="")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed-start", type=int, default=10_000)
    parser.add_argument("--num-rollouts", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=48)
    parser.add_argument("--max-concurrent-rollouts", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--thinking-budget-tokens", type=int, default=512)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--inference-url", default="")
    parser.add_argument("--inference-api-key", default="")
    parser.add_argument("--request-model", default="")
    parser.add_argument("--request-timeout-seconds", type=float, default=300.0)
    parser.add_argument("--video-capture-rollout-index", type=int, default=-1)
    parser.add_argument("--video-capture-output-dir", default="")
    parser.add_argument("--video-capture-fps", type=int, default=6)
    parser.add_argument("--video-capture-tile-size", type=int, default=16)
    parser.add_argument("--video-capture-hide-status-bars", action="store_true")
    args = parser.parse_args()

    result = evaluate_model(
        base_model=args.base_model,
        adapter_dir=Path(args.adapter_dir).expanduser().resolve() if args.adapter_dir else None,
        container_url=args.container_url,
        output_dir=args.output_dir,
        seed_start=args.seed_start,
        num_rollouts=args.num_rollouts,
        max_steps=args.max_steps,
        max_concurrent_rollouts=args.max_concurrent_rollouts,
        max_length=args.max_length,
        max_new_tokens=args.max_new_tokens,
        thinking_budget_tokens=args.thinking_budget_tokens,
        enable_thinking=args.enable_thinking,
        enforce_eager=args.enforce_eager,
        inference_url=args.inference_url,
        inference_api_key=args.inference_api_key,
        request_model=args.request_model,
        request_timeout_seconds=args.request_timeout_seconds,
        video_capture_rollout_index=(
            args.video_capture_rollout_index if args.video_capture_rollout_index >= 0 else None
        ),
        video_capture_output_dir=args.video_capture_output_dir,
        video_capture_fps=args.video_capture_fps,
        video_capture_tile_size=args.video_capture_tile_size,
        video_capture_show_status_bars=not args.video_capture_hide_status_bars,
    )
    write_text(
        Path(args.output_dir).expanduser().resolve() / "eval_command.txt",
        " ".join(
            [
                "python",
                "-m",
                "nanohorizon.shared.eval_model",
                "--base-model",
                args.base_model,
                "--container-url",
                args.container_url,
                "--output-dir",
                str(Path(args.output_dir).expanduser().resolve()),
            ]
            + (
                ["--adapter-dir", str(Path(args.adapter_dir).expanduser().resolve())]
                if args.adapter_dir
                else []
            )
        )
        + "\n",
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
