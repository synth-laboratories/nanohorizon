from __future__ import annotations

from dataclasses import dataclass, field
import gzip
import hashlib
import lzma
import pickle
from pathlib import Path
from typing import Any, Literal, Mapping

import numpy as np

try:
    import jax
except Exception:  # pragma: no cover - optional dependency
    jax = None


Compression = Literal["none", "gzip", "lzma"]


@dataclass(frozen=True)
class Checkpoint:
    version: int
    seed: int
    episode_index: int
    step_index: int
    next_rng: Any
    state: Any
    params: Any = None
    action_history: tuple[int, ...] = ()
    label: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


class CheckpointCodec:
    @staticmethod
    def dumps(checkpoint: Checkpoint, *, compression: Compression = "gzip") -> bytes:
        payload = pickle.dumps(checkpoint, protocol=5)
        if compression == "none":
            return payload
        if compression == "gzip":
            return gzip.compress(payload, compresslevel=6)
        if compression == "lzma":
            return lzma.compress(payload, preset=3)
        raise ValueError(f"unknown compression mode: {compression!r}")

    @staticmethod
    def loads(data: bytes, *, compression: Compression = "gzip") -> Checkpoint:
        if compression == "none":
            payload = data
        elif compression == "gzip":
            payload = gzip.decompress(data)
        elif compression == "lzma":
            payload = lzma.decompress(data)
        else:
            raise ValueError(f"unknown compression mode: {compression!r}")
        loaded = pickle.loads(payload)
        if not isinstance(loaded, Checkpoint):
            raise TypeError("checkpoint payload did not decode to Checkpoint")
        return loaded

    @staticmethod
    def save(checkpoint: Checkpoint, path: str | Path, *, compression: Compression = "gzip") -> Path:
        target = Path(path).expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(CheckpointCodec.dumps(checkpoint, compression=compression))
        return target

    @staticmethod
    def load(path: str | Path, *, compression: Compression = "gzip") -> Checkpoint:
        return CheckpointCodec.loads(Path(path).expanduser().resolve().read_bytes(), compression=compression)


def _leaf_bytes(leaf: Any) -> bytes:
    if isinstance(leaf, np.ndarray):
        header = f"array|{leaf.dtype.str}|{leaf.shape}".encode("utf-8")
        return header + b"\0" + leaf.tobytes(order="C")
    if jax is not None and isinstance(leaf, getattr(jax, "Array", ())):
        arr = np.asarray(leaf)
        header = f"array|{arr.dtype.str}|{arr.shape}".encode("utf-8")
        return header + b"\0" + arr.tobytes(order="C")
    return pickle.dumps(leaf, protocol=5)


def state_digest(tree: Any) -> str:
    if jax is None:
        return hashlib.sha256(pickle.dumps(tree, protocol=5)).hexdigest()
    leaves, treedef = jax.tree_util.tree_flatten(tree)
    digest = hashlib.sha256(pickle.dumps(treedef, protocol=5))
    for leaf in leaves:
        digest.update(_leaf_bytes(leaf))
    return digest.hexdigest()

