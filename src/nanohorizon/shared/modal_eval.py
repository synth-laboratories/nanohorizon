from __future__ import annotations

# ruff: noqa: E402
import json
import os
import sys
from pathlib import Path
from typing import Any

import modal

REMOTE_SRC = Path("/root/nanohorizon/src")
if REMOTE_SRC.exists():
    sys.path.insert(0, str(REMOTE_SRC))

from nanohorizon.shared.common import ensure_dir, now_utc_iso, write_json
from nanohorizon.shared.eval_model import evaluate_model
from nanohorizon.shared.modal_common import (
    ARTIFACT_DIR,
    GPU_EVAL,
    REMOTE_ROOT,
    offline_image,
    volume_mounts,
)

app = modal.App("nanohorizon-craftax-eval")
image = offline_image()


def _default_output_dir() -> str:
    stamp = now_utc_iso().replace(":", "").replace("+00:00", "Z")
    return f"{ARTIFACT_DIR}/eval/{stamp}"


@app.function(
    image=image,
    gpu=GPU_EVAL,
    timeout=60 * 60,
    volumes=volume_mounts(),
)
def run_eval(
    *,
    base_model: str,
    adapter_dir: str = "",
    container_url: str = "",
    container_worker_token: str = "",
    system_prompt: str = "",
    output_dir: str = "",
    seed_start: int = 10_000,
    num_rollouts: int = 8,
    max_steps: int = 48,
    max_concurrent_rollouts: int = 8,
    max_length: int = 4096,
    max_new_tokens: int = 1024,
    thinking_budget_tokens: int = 512,
    enable_thinking: bool = False,
    enforce_eager: bool = False,
    video_capture_rollout_index: int = -1,
    video_capture_output_dir: str = "",
    video_capture_fps: int = 6,
    video_capture_tile_size: int = 16,
    video_capture_show_status_bars: bool = True,
    summary_name: str = "eval_summary.json",
) -> dict[str, Any]:
    os.chdir(REMOTE_ROOT)
    destination = ensure_dir(output_dir or _default_output_dir())
    if container_url:
        os.environ["NANOHORIZON_CRAFTAX_CONTAINER_URL"] = container_url
        os.environ["NANOHORIZON_CRAFTAX_CONTAINER_URL"] = container_url
    if container_worker_token:
        os.environ["NANOHORIZON_CRAFTAX_CONTAINER_WORKER_TOKEN"] = container_worker_token
        os.environ["NANOHORIZON_CRAFTAX_CONTAINER_WORKER_TOKEN"] = container_worker_token
    summary = evaluate_model(
        base_model=base_model,
        adapter_dir=Path(adapter_dir).resolve() if adapter_dir else None,
        container_url=container_url,
        system_prompt=system_prompt,
        output_dir=destination,
        seed_start=seed_start,
        num_rollouts=num_rollouts,
        max_steps=max_steps,
        max_concurrent_rollouts=max_concurrent_rollouts,
        max_length=max_length,
        max_new_tokens=max_new_tokens,
        thinking_budget_tokens=thinking_budget_tokens,
        enable_thinking=enable_thinking,
        enforce_eager=enforce_eager,
        video_capture_rollout_index=(
            video_capture_rollout_index if int(video_capture_rollout_index) >= 0 else None
        ),
        video_capture_output_dir=video_capture_output_dir,
        video_capture_fps=video_capture_fps,
        video_capture_tile_size=video_capture_tile_size,
        video_capture_show_status_bars=video_capture_show_status_bars,
        summary_name=summary_name,
    )
    payload = {
        "base_model": base_model,
        "adapter_dir": adapter_dir,
        "container_url": container_url,
        "system_prompt": system_prompt,
        "output_dir": str(destination),
        "summary": summary,
    }
    write_json(destination / "modal_eval_result.json", payload)
    return payload


@app.local_entrypoint()
def main(
    base_model: str = "Qwen/Qwen3.5-4B",
    adapter_dir: str = "",
    container_url: str = "",
    container_worker_token: str = "",
    system_prompt: str = "",
    output_dir: str = "",
    seed_start: int = 10_000,
    num_rollouts: int = 8,
    max_steps: int = 48,
    max_concurrent_rollouts: int = 8,
    max_length: int = 4096,
    max_new_tokens: int = 1024,
    thinking_budget_tokens: int = 512,
    enable_thinking: bool = False,
    enforce_eager: bool = False,
    video_capture_rollout_index: int = -1,
    video_capture_output_dir: str = "",
    video_capture_fps: int = 6,
    video_capture_tile_size: int = 16,
    video_capture_show_status_bars: bool = True,
) -> None:
    result: object = run_eval.remote(
        base_model=base_model,
        adapter_dir=adapter_dir,
        container_url=container_url,
        container_worker_token=container_worker_token,
        system_prompt=system_prompt,
        output_dir=output_dir,
        seed_start=seed_start,
        num_rollouts=num_rollouts,
        max_steps=max_steps,
        max_concurrent_rollouts=max_concurrent_rollouts,
        max_length=max_length,
        max_new_tokens=max_new_tokens,
        thinking_budget_tokens=thinking_budget_tokens,
        enable_thinking=enable_thinking,
        enforce_eager=enforce_eager,
        video_capture_rollout_index=video_capture_rollout_index,
        video_capture_output_dir=video_capture_output_dir,
        video_capture_fps=video_capture_fps,
        video_capture_tile_size=video_capture_tile_size,
        video_capture_show_status_bars=video_capture_show_status_bars,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
