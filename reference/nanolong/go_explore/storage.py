from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .models import ArtifactRef


class LocalArtifactStore:
    def __init__(self, root: Path) -> None:
        self.root = root.expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def _write_bytes(
        self,
        *,
        artifact_type: str,
        artifact_id: str,
        body: bytes,
        suffix: str,
        metadata: dict[str, Any] | None = None,
    ) -> ArtifactRef:
        artifact_dir = self.root / artifact_type
        artifact_dir.mkdir(parents=True, exist_ok=True)
        path = artifact_dir / f"{artifact_id}{suffix}"
        path.write_bytes(body)
        return ArtifactRef(
            artifact_type=artifact_type,
            path=str(path),
            size_bytes=len(body),
            sha256=hashlib.sha256(body).hexdigest(),
            metadata=dict(metadata or {}),
        )

    def write_json(
        self,
        *,
        artifact_type: str,
        artifact_id: str,
        payload: Any,
        metadata: dict[str, Any] | None = None,
    ) -> ArtifactRef:
        body = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")
        return self._write_bytes(
            artifact_type=artifact_type,
            artifact_id=artifact_id,
            body=body,
            suffix=".json",
            metadata=metadata,
        )

    def write_text(
        self,
        *,
        artifact_type: str,
        artifact_id: str,
        text: str,
        metadata: dict[str, Any] | None = None,
    ) -> ArtifactRef:
        return self._write_bytes(
            artifact_type=artifact_type,
            artifact_id=artifact_id,
            body=text.encode("utf-8"),
            suffix=".txt",
            metadata=metadata,
        )
