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

from nanohorizon.common import ensure_dir, now_utc_iso, write_json
from nanohorizon.modal_common import (
    ARTIFACT_DIR,
    GPU_OFFLINE,
    REMOTE_ROOT,
    training_image,
    volume_mounts,
)
from nanohorizon.train_lora import train_sft_with_trl

APP_NAME = os.getenv("NANOHORIZON_MODAL_SFT_APP_NAME", "nanohorizon-crafter-sft")

app = modal.App(APP_NAME)
image = training_image()


def _default_output_dir() -> str:
    stamp = now_utc_iso().replace(":", "").replace("+00:00", "Z")
    return f"{ARTIFACT_DIR}/offline_sft/{stamp}"


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
    }
    write_json(destination / "modal_sft_result.json", payload)
    return payload


@app.local_entrypoint()
def main(
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
