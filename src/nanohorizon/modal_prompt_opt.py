from __future__ import annotations

# ruff: noqa: E402
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

import modal

REMOTE_SRC = Path("/root/nanohorizon/src")
if REMOTE_SRC.exists():
    sys.path.insert(0, str(REMOTE_SRC))

from nanohorizon.common import ensure_dir, now_utc_iso
from nanohorizon.modal_common import (
    ARTIFACT_DIR,
    GPU_PROMPT_OPT,
    REMOTE_ROOT,
    prompt_image,
    volume_mounts,
)

app = modal.App("nanohorizon-crafter-prompt-opt")
image = prompt_image()


def _default_output_dir() -> str:
    stamp = now_utc_iso().replace(":", "").replace("+00:00", "Z")
    return f"{ARTIFACT_DIR}/prompt_opt/{stamp}"


@app.function(
    image=image,
    gpu=GPU_PROMPT_OPT,
    timeout=60 * 30,
    volumes=volume_mounts(),
)
def run(
    *,
    config: str = "configs/crafter_prompt_opt_qwen35_4b_gpt54_budget.yaml",
    output_dir: str = "",
) -> dict[str, object]:
    os.chdir(REMOTE_ROOT)
    destination = ensure_dir(output_dir or _default_output_dir())
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{REMOTE_ROOT}/src"
    cmd = [
        sys.executable,
        "-m",
        "nanohorizon.baselines.prompt_opt",
        "--config",
        f"{REMOTE_ROOT}/{config}",
        "--output-dir",
        str(destination),
    ]
    print("Remote prompt-opt command:", " ".join(shlex.quote(part) for part in cmd), flush=True)
    subprocess.check_call(cmd, env=env)
    metrics_path = destination / "metrics.json"
    payload = {
        "output_dir": str(destination),
        "metrics": json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {},
    }
    return payload


@app.local_entrypoint()
def main(
    config: str = "configs/crafter_prompt_opt_qwen35_4b_gpt54_budget.yaml",
    output_dir: str = "",
) -> None:
    result = run.remote(config=config, output_dir=output_dir)
    print(json.dumps(result, indent=2, sort_keys=True))
