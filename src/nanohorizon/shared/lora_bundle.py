from __future__ import annotations

import io
import tarfile
import tempfile
from pathlib import Path


def build_lora_bundle(adapter_dir: Path) -> tuple[bytes, list[str]]:
    resolved_dir = Path(adapter_dir).expanduser().resolve()
    if not resolved_dir.is_dir():
        raise FileNotFoundError(f"adapter_dir does not exist: {resolved_dir}")
    file_paths = sorted(path for path in resolved_dir.rglob("*") if path.is_file())
    if not file_paths:
        raise FileNotFoundError(f"adapter_dir contains no files: {resolved_dir}")
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for path in file_paths:
            archive.add(str(path), arcname=str(path.relative_to(resolved_dir)))
    return buffer.getvalue(), [str(path.relative_to(resolved_dir)) for path in file_paths]


def extract_lora_bundle(*, bundle_bytes: bytes, dest_root: Path, bundle_name: str) -> Path:
    resolved_root = Path(dest_root).expanduser().resolve()
    resolved_root.mkdir(parents=True, exist_ok=True)
    target_dir = Path(
        tempfile.mkdtemp(prefix=f"{_safe_name(bundle_name)}-", dir=str(resolved_root))
    ).resolve()
    with tarfile.open(fileobj=io.BytesIO(bundle_bytes), mode="r:gz") as archive:
        for member in archive.getmembers():
            _validate_member(member_name=member.name, target_dir=target_dir)
        archive.extractall(path=str(target_dir))
    return target_dir


def _safe_name(value: str) -> str:
    cleaned = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in str(value or "adapter"))
    cleaned = cleaned.strip("._")
    return cleaned or "adapter"


def _validate_member(*, member_name: str, target_dir: Path) -> None:
    destination = (target_dir / member_name).resolve()
    if target_dir not in destination.parents and destination != target_dir:
        raise ValueError(f"unsafe bundle member path: {member_name}")
