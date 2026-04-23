from __future__ import annotations

import os
from collections.abc import Iterable
from pathlib import Path, PurePosixPath
from typing import TypeAlias

import modal

PROJECT_ROOT = Path(__file__).resolve().parents[3]
REMOTE_ROOT = "/root/nanohorizon"
HF_CACHE_DIR = "/root/.cache/huggingface"
VLLM_CACHE_DIR = "/root/.cache/vllm"
VLLM_COMPILE_CACHE_DIR = f"{VLLM_CACHE_DIR}/torch_compile_cache"
TRITON_CACHE_DIR = "/root/.triton"
CRAFTAX_CACHE_DIR = "/vol/craftax-cache"
JAX_CACHE_DIR = f"{CRAFTAX_CACHE_DIR}/jax"
ARTIFACT_DIR = "/vol/artifacts"
RECORDS_DIR = "/vol/records"
OFFLINE_VENV_ROOT = "/opt/nanohorizon-offline-venvs"
VLLM_BASE_IMAGE = "vllm/vllm-openai:latest"
VLLM_IMAGE_PYTHON_VERSION = "3.12"
TRANSFORMERS_VERSION = os.getenv("NANOHORIZON_TRANSFORMERS_VERSION", "4.57.6").strip() or "4.57.6"
TRAIN_TRANSFORMERS_SPEC = (
    os.getenv("NANOHORIZON_TRAIN_TRANSFORMERS_SPEC", "transformers @ git+https://github.com/huggingface/transformers.git@main").strip()
    or "transformers @ git+https://github.com/huggingface/transformers.git@main"
)
PIP_DEFAULT_TIMEOUT = str(int(os.getenv("NANOHORIZON_MODAL_PIP_TIMEOUT_SECONDS", "1800") or "1800"))
PIP_RETRIES = str(int(os.getenv("NANOHORIZON_MODAL_PIP_RETRIES", "8") or "8"))

HF_CACHE_VOLUME = modal.Volume.from_name("nanohorizon-hf-cache", create_if_missing=True)
VLLM_CACHE_VOLUME = modal.Volume.from_name("nanohorizon-vllm-cache", create_if_missing=True)
TRITON_CACHE_VOLUME = modal.Volume.from_name("nanohorizon-triton-cache", create_if_missing=True)
CRAFTAX_CACHE_VOLUME = modal.Volume.from_name("nanohorizon-craftax-cache", create_if_missing=True)
ARTIFACT_VOLUME = modal.Volume.from_name("nanohorizon-artifacts", create_if_missing=True)
RECORDS_VOLUME = modal.Volume.from_name("nanohorizon-records", create_if_missing=True)

VolumeMounts: TypeAlias = dict[str | PurePosixPath, modal.Volume | modal.CloudBucketMount]

def _resolve_modal_gpu(raw_value: str) -> str:
    value = str(raw_value or "").strip()
    normalized = value.upper()
    if normalized in {"A100", "A100-40GB"}:
        return "A100-40GB"
    if normalized == "A100-80GB":
        return "A100-80GB"
    if normalized == "A10G":
        return "A10G"
    if normalized == "L4":
        return "L4"
    if normalized == "L40S":
        return "L40S"
    if normalized == "T4":
        return "T4"
    return value


GPU_OFFLINE = _resolve_modal_gpu(os.getenv("NANOHORIZON_MODAL_GPU_OFFLINE", "A10G"))
GPU_RLVR = _resolve_modal_gpu(os.getenv("NANOHORIZON_MODAL_GPU_RLVR", "A100-40GB"))
GPU_PROMPT_OPT = _resolve_modal_gpu(os.getenv("NANOHORIZON_MODAL_GPU_PROMPT_OPT", "L4"))
GPU_EVAL = _resolve_modal_gpu(os.getenv("NANOHORIZON_MODAL_GPU_EVAL", "L4"))
GPU_TEACHER = _resolve_modal_gpu(os.getenv("NANOHORIZON_MODAL_GPU_TEACHER", "A100-40GB"))

COMMON_PACKAGES = [
    "fastapi>=0.115.0",
    "httpx>=0.28.1",
    "pillow>=11.3.0",
    "pyyaml>=6.0.2",
    "uvicorn>=0.32.0",
]

CRAFTAX_PACKAGES = [
    "chex>=0.1.90",
    "craftax",
    "distrax>=0.1.5",
    "flax>=0.10.0",
    "gymnax>=0.0.8",
    "imageio>=2.37.0",
    "jax[cuda12]",
    "optax>=0.2.4",
    "orbax-checkpoint==0.5.0",
]

NLE_PACKAGES = [
    "gymnasium>=1.0.0",
    "imageio>=2.37.0",
    "imageio-ffmpeg>=0.6.0",
    "nle>=1.1.0",
    "numpy>=2.0.0",
    "pillow>=11.3.0",
]

TRAIN_PACKAGES = [
    "accelerate>=1.10.0",
    "datasets>=4.1.0",
    "peft>=0.15.0",
    "torch>=2.6.0",
    "trl>=0.28.0",
    TRAIN_TRANSFORMERS_SPEC,
]

PROMPT_PACKAGES = [
    *COMMON_PACKAGES,
    "gepa>=0.1.1",
    "litellm>=1.79.0",
]


def _dedupe(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item not in seen:
            out.append(item)
            seen.add(item)
    return out


def _attach_repo(image: modal.Image) -> modal.Image:
    return (
        image.add_local_dir((PROJECT_ROOT / "src").as_posix(), remote_path=f"{REMOTE_ROOT}/src", copy=True)
        .add_local_dir((PROJECT_ROOT / "scripts").as_posix(), remote_path=f"{REMOTE_ROOT}/scripts", copy=True)
        .add_local_dir((PROJECT_ROOT / "configs").as_posix(), remote_path=f"{REMOTE_ROOT}/configs", copy=True)
        .add_local_dir((PROJECT_ROOT / "data").as_posix(), remote_path=f"{REMOTE_ROOT}/data", copy=True)
        .add_local_file(
            (PROJECT_ROOT / "pyproject.toml").as_posix(),
            remote_path=f"{REMOTE_ROOT}/pyproject.toml",
            copy=True,
        )
        .add_local_file((PROJECT_ROOT / "README.md").as_posix(), remote_path=f"{REMOTE_ROOT}/README.md", copy=True)
    )


def _image_env() -> dict[str, str]:
    return {
        "HF_HOME": HF_CACHE_DIR,
        "TRITON_CACHE_DIR": TRITON_CACHE_DIR,
        "JAX_COMPILATION_CACHE_DIR": JAX_CACHE_DIR,
        "JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES": "xla_gpu_per_fusion_autotune_cache_dir",
        "NANOHORIZON_CRAFTAX_CACHE_DIR": CRAFTAX_CACHE_DIR,
        "PIP_DEFAULT_TIMEOUT": PIP_DEFAULT_TIMEOUT,
        "PIP_RETRIES": PIP_RETRIES,
        "PIP_DISABLE_PIP_VERSION_CHECK": "1",
        "PIP_PROGRESS_BAR": "off",
        "TF_CPP_MIN_LOG_LEVEL": "2",
    }


def _install_common_stack(image: modal.Image) -> modal.Image:
    return image.pip_install(*COMMON_PACKAGES)


def _install_craftax_stack(image: modal.Image) -> modal.Image:
    return image.pip_install(*CRAFTAX_PACKAGES)


def _install_nle_stack(image: modal.Image) -> modal.Image:
    return image.pip_install(*NLE_PACKAGES)


def _install_training_stack(image: modal.Image) -> modal.Image:
    return image.pip_install(*TRAIN_PACKAGES)


def _cuda_base_image() -> modal.Image:
    return (
        modal.Image.from_registry(
            "nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04",
            add_python="3.11",
        )
        .env(_image_env())
        .apt_install(
            "git",
            "curl",
            "build-essential",
            "ninja-build",
            "libgl1",
            "libglib2.0-0",
        )
    )


def base_runtime_image() -> modal.Image:
    image = _install_common_stack(_cuda_base_image())
    return _attach_repo(image)


def training_image(*extra_packages: str) -> modal.Image:
    image = _install_training_stack(_install_common_stack(_cuda_base_image()))
    extra = _dedupe(extra_packages)
    if extra:
        image = image.pip_install(*extra)
    return _attach_repo(image)


def prompt_image(*extra_packages: str) -> modal.Image:
    packages = _dedupe([*PROMPT_PACKAGES, *extra_packages])
    image = modal.Image.debian_slim(python_version="3.11").pip_install(*packages)
    return _attach_repo(image)


def craftax_runtime_image(*extra_packages: str) -> modal.Image:
    image = _install_craftax_stack(_install_common_stack(_cuda_base_image()))
    extra = _dedupe(extra_packages)
    if extra:
        image = image.pip_install(*extra)
    return _attach_repo(image)


def nle_runtime_image(*extra_packages: str) -> modal.Image:
    image = _install_nle_stack(_install_common_stack(_cuda_base_image()))
    extra = _dedupe(extra_packages)
    if extra:
        image = image.pip_install(*extra)
    return _attach_repo(image)


def offline_image() -> modal.Image:
    teacher_venv = f"{OFFLINE_VENV_ROOT}/teacher"
    image = (
        _install_training_stack(
            _install_craftax_stack(
                _install_common_stack(_cuda_base_image())
            )
        )
    )
    image = image.run_commands(
        f"python -m venv {teacher_venv}",
        f"{teacher_venv}/bin/python -m pip install --upgrade pip",
        f"{teacher_venv}/bin/python -m pip install "
        f"\"httpx>=0.28.1\" \"pyyaml>=6.0.2\" \"vllm>=0.10.0\" \"transformers=={TRANSFORMERS_VERSION}\"",
    )
    return _attach_repo(image)


def offline_worker_image() -> modal.Image:
    image = (
        _install_training_stack(
            _install_craftax_stack(
                _install_common_stack(_cuda_base_image())
            )
        )
    )
    return _attach_repo(image)


def rlvr_vllm_image(*extra_packages: str) -> modal.Image:
    packages = _dedupe(
        [
            *COMMON_PACKAGES,
            "numpy>=2.0.0",
            "openai>=1.109.1",
            "transformers==4.57.6",
            *extra_packages,
        ]
    )
    image = (
        modal.Image.from_registry(VLLM_BASE_IMAGE, add_python=VLLM_IMAGE_PYTHON_VERSION)
        .entrypoint([])
        .env(_image_env())
        .apt_install("curl")
        .pip_install(*packages)
    )
    return _attach_repo(image)


def volume_mounts(extra: VolumeMounts | None = None) -> VolumeMounts:
    mounts: VolumeMounts = {
        HF_CACHE_DIR: HF_CACHE_VOLUME,
        VLLM_CACHE_DIR: VLLM_CACHE_VOLUME,
        TRITON_CACHE_DIR: TRITON_CACHE_VOLUME,
        CRAFTAX_CACHE_DIR: CRAFTAX_CACHE_VOLUME,
        ARTIFACT_DIR: ARTIFACT_VOLUME,
        RECORDS_DIR: RECORDS_VOLUME,
    }
    if extra:
        mounts.update(extra)
    return mounts
