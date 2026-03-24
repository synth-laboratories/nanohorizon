from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import modal

PACKAGE_ROOT = Path("/root/nanohorizon/src")
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from nanohorizon.shared.modal_common import ARTIFACT_DIR, PROJECT_ROOT, RECORDS_DIR, volume_mounts

APP_NAME = "nanohorizon-classic"
REMOTE_ROOT = "/root/nanohorizon"
JAX_CACHE_DIR = f"{ARTIFACT_DIR}/classic/jax_cache"


def _resolve_modal_gpu(raw_value: str) -> str:
    value = str(raw_value or "").strip().upper()
    if value in {"T4", "L4", "A10G", "A100-40GB", "L40S"}:
        return value
    return "T4"


GPU_CLASSIC = _resolve_modal_gpu(os.getenv("NANOHORIZON_MODAL_GPU_CLASSIC", "L4"))


def _prepare_craftax_texture_cache() -> None:
    os.environ.pop("CRAFTAX_RELOAD_TEXTURES", None)

    import craftax.craftax.constants as craftax_constants
    import craftax.craftax_classic.constants as craftax_classic_constants

    print(
        "Craftax texture cache:",
        craftax_constants.TEXTURE_CACHE_FILE,
        os.path.exists(craftax_constants.TEXTURE_CACHE_FILE),
    )
    print(
        "Craftax-Classic texture cache:",
        craftax_classic_constants.TEXTURE_CACHE_FILE,
        os.path.exists(craftax_classic_constants.TEXTURE_CACHE_FILE),
    )


def _classic_runtime_env() -> dict[str, str]:
    env = {
        "JAX_COMPILATION_CACHE_DIR": JAX_CACHE_DIR,
        "JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES": "xla_gpu_per_fusion_autotune_cache_dir",
        "JAX_DEFAULT_MATMUL_PRECISION": "tensorfloat32",
        "XLA_FLAGS": "--xla_gpu_triton_gemm_any=true --xla_gpu_enable_latency_hiding_scheduler=true",
        "TF_CPP_MIN_LOG_LEVEL": "2",
    }
    existing_xla_flags = os.environ.get("XLA_FLAGS", "").strip()
    if existing_xla_flags:
        env["XLA_FLAGS"] = f"{existing_xla_flags} {env['XLA_FLAGS']}".strip()
    return env

classic_image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git", "curl", "build-essential", "libgl1", "libglib2.0-0")
    .pip_install(
        "httpx>=0.28.1",
        "pyyaml>=6.0.2",
        "chex>=0.1.90",
        "craftax",
        "distrax>=0.1.5",
        "flax>=0.10.0",
        "gymnax>=0.0.8",
        "numpy>=2.0.0",
        "optax>=0.2.4",
        "orbax-checkpoint==0.5.0",
        "jax[cuda12]",
    )
    .add_local_dir((PROJECT_ROOT / "src").as_posix(), remote_path=f"{REMOTE_ROOT}/src", copy=True)
    .add_local_dir((PROJECT_ROOT / "scripts").as_posix(), remote_path=f"{REMOTE_ROOT}/scripts", copy=True)
    .add_local_dir((PROJECT_ROOT / "configs").as_posix(), remote_path=f"{REMOTE_ROOT}/configs", copy=True)
    .add_local_file((PROJECT_ROOT / "pyproject.toml").as_posix(), remote_path=f"{REMOTE_ROOT}/pyproject.toml", copy=True)
    .add_local_file((PROJECT_ROOT / "README.md").as_posix(), remote_path=f"{REMOTE_ROOT}/README.md", copy=True)
    .run_function(_prepare_craftax_texture_cache, env={"JAX_PLATFORMS": "cpu"})
)

app = modal.App(APP_NAME)


@app.function(
    image=classic_image,
    gpu=GPU_CLASSIC,
    timeout=60 * 60 * 2,
    volumes=volume_mounts(),
)
def run_classic_baseline(
    config: str = "configs/classic_craftax_1m_random_init.yaml",
    output_dir: str = "",
    mode: str = "train",
) -> dict[str, str]:
    import subprocess
    import sys

    os.chdir(REMOTE_ROOT)
    target_dir = output_dir.strip() or f"{RECORDS_DIR}/classic/manual_reference_baseline"
    Path(JAX_CACHE_DIR).mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "nanohorizon.baselines.classic",
        "--config",
        config,
        "--output-dir",
        target_dir,
    ]
    normalized_mode = mode.strip().lower()
    if normalized_mode == "eval-only":
        cmd.append("--eval-only")
    else:
        cmd.append("--train")
    env = dict(os.environ)
    env.update(_classic_runtime_env())
    env.pop("CRAFTAX_RELOAD_TEXTURES", None)
    env["PYTHONPATH"] = f"{PACKAGE_ROOT}:{env.get('PYTHONPATH', '')}".rstrip(":")
    process = subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stderr, text=True, env=env)
    returncode = process.wait()
    if returncode != 0:
        raise RuntimeError(f"classic baseline failed with exit code {returncode}")
    return {
        "status": "ok",
        "gpu": GPU_CLASSIC,
        "config": config,
        "output_dir": target_dir,
        "mode": normalized_mode,
        "stdout_tail": "streamed to Modal logs",
    }


@app.local_entrypoint()
def modal_main(
    config: str = "configs/classic_craftax_1m_random_init.yaml",
    output_dir: str = "",
    mode: str = "train",
) -> None:
    result = run_classic_baseline.remote(config=config, output_dir=output_dir, mode=mode)
    print(json.dumps(result, indent=2, sort_keys=True))
