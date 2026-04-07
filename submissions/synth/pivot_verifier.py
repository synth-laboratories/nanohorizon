from __future__ import annotations

# ruff: noqa: E402
import json
import os
import shlex
import sys
from datetime import UTC, datetime
from pathlib import Path

import modal

REMOTE_ROOT = Path("/root/nanohorizon")
REMOTE_SRC = REMOTE_ROOT / "src"
_THIS_FILE = Path(__file__).resolve()
if len(_THIS_FILE.parents) >= 3:
    LOCAL_ROOT = _THIS_FILE.parents[2]
elif REMOTE_ROOT.exists():
    LOCAL_ROOT = REMOTE_ROOT
else:
    LOCAL_ROOT = Path.cwd().resolve()
if str(LOCAL_ROOT) not in sys.path:
    sys.path.insert(0, str(LOCAL_ROOT))
LOCAL_SRC = LOCAL_ROOT / "src"
if str(LOCAL_SRC) not in sys.path:
    sys.path.insert(0, str(LOCAL_SRC))
if REMOTE_ROOT.exists():
    sys.path.insert(0, str(REMOTE_ROOT))
if REMOTE_SRC.exists():
    sys.path.insert(0, str(REMOTE_SRC))

from nanohorizon.baselines import pivot_verifier as pivot_verifier_baseline
from nanohorizon.shared.common import ensure_dir, load_config, resolve_path, write_json
from nanohorizon.shared.modal_common import (
    GPU_OFFLINE,
    PROJECT_ROOT,
    RECORDS_DIR,
    offline_image,
    volume_mounts,
)
from nanohorizon.shared.modal_common import (
    REMOTE_ROOT as SHARED_REMOTE_ROOT,
)

APP_NAME = os.getenv("NANOHORIZON_MODAL_PIVOT_VERIFIER_APP_NAME", "nanohorizon-craftax-pivot-verifier").strip() or "nanohorizon-craftax-pivot-verifier"
image = offline_image().add_local_dir(
    (PROJECT_ROOT / "submissions").as_posix(),
    remote_path=f"{SHARED_REMOTE_ROOT}/submissions",
    copy=True,
)
app = modal.App(APP_NAME)


def _default_output_dir(config_payload: dict[str, object]) -> str:
    task_cfg = config_payload.get("task") if isinstance(config_payload.get("task"), dict) else {}
    track = str((task_cfg or {}).get("track") or pivot_verifier_baseline.DEFAULT_TRACK_ID)
    method = str((task_cfg or {}).get("method_name") or pivot_verifier_baseline.DEFAULT_METHOD_NAME)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{RECORDS_DIR}/{track}/{stamp}_{method}"


@app.function(
    image=image,
    gpu=GPU_OFFLINE,
    timeout=60 * 60 * 4,
    volumes=volume_mounts(),
)
def run(
    *,
    config: str = "configs/pivot_verifier_qwen35_4b_spct.yaml",
    output_dir: str = "",
    skip_train: bool = False,
    skip_eval: bool = False,
) -> dict[str, object]:
    os.chdir(str(REMOTE_ROOT))
    config_path = resolve_path(config, base_dir=REMOTE_ROOT)
    config_payload = load_config(config_path)
    destination = ensure_dir(output_dir or _default_output_dir(config_payload))
    command = " ".join(
        shlex.quote(part)
        for part in [
            "modal",
            "run",
            "submissions/synth/pivot_verifier.py",
            "--config",
            str(config),
            *(["--skip-train"] if skip_train else []),
            *(["--skip-eval"] if skip_eval else []),
        ]
    )
    return pivot_verifier_baseline.run_pipeline(
        config=config_payload,
        output_root=destination,
        command=command,
        skip_train=skip_train,
        skip_eval=skip_eval,
    )


@app.local_entrypoint()
def main(
    config: str = "configs/pivot_verifier_qwen35_4b_spct.yaml",
    output_dir: str = "",
    skip_train: bool = False,
    skip_eval: bool = False,
    local_result_path: str = "",
) -> None:
    result = run.remote(
        config=config,
        output_dir=output_dir,
        skip_train=skip_train,
        skip_eval=skip_eval,
    )
    if str(local_result_path or "").strip():
        write_json(local_result_path, result)
    print(json.dumps(result, indent=2, sort_keys=True))
