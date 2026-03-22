from __future__ import annotations

import os
from collections.abc import Iterable
from pathlib import Path, PurePosixPath
from typing import TypeAlias

import modal

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REMOTE_ROOT = "/root/nanohorizon"
HF_CACHE_DIR = "/root/.cache/huggingface"
VLLM_CACHE_DIR = "/root/.cache/vllm"
VLLM_COMPILE_CACHE_DIR = f"{VLLM_CACHE_DIR}/torch_compile_cache"
TRITON_CACHE_DIR = "/root/.triton"
ARTIFACT_DIR = "/vol/artifacts"
RECORDS_DIR = "/vol/records"
OFFLINE_VENV_ROOT = "/opt/nanohorizon-offline-venvs"
VLLM_BASE_IMAGE = "vllm/vllm-openai:latest"
VLLM_IMAGE_PYTHON_VERSION = "3.12"

HF_CACHE_VOLUME = modal.Volume.from_name("nanohorizon-hf-cache", create_if_missing=True)
VLLM_CACHE_VOLUME = modal.Volume.from_name("nanohorizon-vllm-cache", create_if_missing=True)
TRITON_CACHE_VOLUME = modal.Volume.from_name("nanohorizon-triton-cache", create_if_missing=True)
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
GPU_TEACHER = _resolve_modal_gpu(os.getenv("NANOHORIZON_MODAL_GPU_TEACHER", "A10G"))

COMMON_PACKAGES = [
    "httpx>=0.28.1",
    "pyyaml>=6.0.2",
]

TRAIN_PACKAGES = [
    *COMMON_PACKAGES,
    "accelerate>=1.10.0",
    "datasets>=4.1.0",
    "peft>=0.15.0",
    "torch>=2.6.0",
    "trl>=0.28.0",
    "transformers @ git+https://github.com/huggingface/transformers.git@main",
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
        image.add_local_dir((PROJECT_ROOT / "src").as_posix(), remote_path=f"{REMOTE_ROOT}/src")
        .add_local_dir((PROJECT_ROOT / "scripts").as_posix(), remote_path=f"{REMOTE_ROOT}/scripts")
        .add_local_dir((PROJECT_ROOT / "configs").as_posix(), remote_path=f"{REMOTE_ROOT}/configs")
        .add_local_dir((PROJECT_ROOT / "data").as_posix(), remote_path=f"{REMOTE_ROOT}/data")
        .add_local_dir(
            (PROJECT_ROOT / "containers" / "crafter_rs" / "src").as_posix(),
            remote_path=f"{REMOTE_ROOT}/containers/crafter_rs/src",
        )
        .add_local_file(
            (PROJECT_ROOT / "containers" / "crafter_rs" / "Cargo.toml").as_posix(),
            remote_path=f"{REMOTE_ROOT}/containers/crafter_rs/Cargo.toml",
        )
        .add_local_file(
            (PROJECT_ROOT / "containers" / "crafter_rs" / "Cargo.lock").as_posix(),
            remote_path=f"{REMOTE_ROOT}/containers/crafter_rs/Cargo.lock",
        )
        .add_local_file(
            (PROJECT_ROOT / "containers" / "crafter_rs" / "README.md").as_posix(),
            remote_path=f"{REMOTE_ROOT}/containers/crafter_rs/README.md",
        )
        .add_local_file((PROJECT_ROOT / "pyproject.toml").as_posix(), remote_path=f"{REMOTE_ROOT}/pyproject.toml")
        .add_local_file((PROJECT_ROOT / "README.md").as_posix(), remote_path=f"{REMOTE_ROOT}/README.md")
    )


def _cuda_base_image() -> modal.Image:
    return (
        modal.Image.from_registry(
            "nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04",
            add_python="3.11",
        )
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
    image = _cuda_base_image().pip_install(*COMMON_PACKAGES)
    return _attach_repo(image)


def training_image(*extra_packages: str) -> modal.Image:
    packages = _dedupe([*TRAIN_PACKAGES, *extra_packages])
    image = _cuda_base_image().pip_install(*packages)
    return _attach_repo(image)


def prompt_image(*extra_packages: str) -> modal.Image:
    packages = _dedupe([*PROMPT_PACKAGES, *extra_packages])
    image = modal.Image.debian_slim(python_version="3.11").pip_install(*packages)
    return _attach_repo(image)


def crafter_runtime_image(*extra_packages: str) -> modal.Image:
    packages = _dedupe([*COMMON_PACKAGES, *extra_packages])
    image = (
        _cuda_base_image()
        .pip_install(*packages)
        .run_commands(
            "curl https://sh.rustup.rs -sSf | sh -s -- -y --profile minimal --default-toolchain stable",
            "/root/.cargo/bin/cargo --version",
        )
    )
    return _attach_repo(image)


def offline_image() -> modal.Image:
    teacher_venv = f"{OFFLINE_VENV_ROOT}/teacher"
    image = (
        _cuda_base_image()
        .pip_install(*TRAIN_PACKAGES)
        .run_commands(
            f"python -m venv {teacher_venv}",
            f"{teacher_venv}/bin/python -m pip install --upgrade pip",
            f"{teacher_venv}/bin/python -m pip install "
            "\"httpx>=0.28.1\" \"pyyaml>=6.0.2\" \"vllm>=0.10.0\"",
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
        .apt_install("curl")
        .pip_install(*packages)
    )
    return _attach_repo(image)


def volume_mounts(extra: VolumeMounts | None = None) -> VolumeMounts:
    mounts: VolumeMounts = {
        HF_CACHE_DIR: HF_CACHE_VOLUME,
        VLLM_CACHE_DIR: VLLM_CACHE_VOLUME,
        TRITON_CACHE_DIR: TRITON_CACHE_VOLUME,
        ARTIFACT_DIR: ARTIFACT_VOLUME,
        RECORDS_DIR: RECORDS_VOLUME,
    }
    if extra:
        mounts.update(extra)
    return mounts
