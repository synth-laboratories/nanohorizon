"""Compact todo scratchpad used during Craftax validation runs."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fastapi import FastAPI

from .metadata import (
    CANDIDATE_LABEL,
    PRESERVED_HARNESS_SURFACES,
    SCRATCHPAD_PATH,
    build_candidate_manifest,
    build_candidate_metadata,
    build_candidate_prompt,
)


@dataclass(slots=True)
class TodoItem:
    title: str
    done: bool = False

    def to_dict(self) -> dict[str, object]:
        return {"title": self.title, "done": self.done}

    @classmethod
    def from_dict(cls, raw: dict[str, object]) -> "TodoItem":
        return cls(title=str(raw["title"]), done=bool(raw.get("done", False)))


@dataclass(slots=True)
class CompactTodoScratchpad:
    path: Path
    limit: int = 3
    items: list[TodoItem] = field(default_factory=list)

    @classmethod
    def load(cls, path: str | Path, limit: int = 3) -> "CompactTodoScratchpad":
        path = Path(path)
        if path.exists():
            raw_text = path.read_text(encoding="utf-8").strip()
            payload = json.loads(raw_text) if raw_text else {}
            items = [TodoItem.from_dict(item) for item in payload.get("items", [])]
        else:
            items = []
        return cls(path=path, limit=limit, items=items[:limit])

    def add(self, title: str) -> TodoItem:
        item = TodoItem(title=title)
        self.items.append(item)
        self._trim()
        return item

    def mark_done(self, index: int) -> TodoItem:
        item = self.items[index]
        item.done = True
        return item

    def pending(self) -> list[TodoItem]:
        return [item for item in self.items if not item.done]

    def snapshot(self) -> dict[str, object]:
        return {
            "limit": self.limit,
            "items": [item.to_dict() for item in self.items],
            "pending": [item.to_dict() for item in self.pending()],
        }

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(self.snapshot(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def render(self) -> str:
        lines = []
        for idx, item in enumerate(self.items, start=1):
            marker = "x" if item.done else " "
            lines.append(f"{idx}. [{marker}] {item.title}")
        if not lines:
            lines.append("(empty)")
        return "\n".join(lines)

    def _trim(self) -> None:
        overflow = len(self.items) - self.limit
        if overflow > 0:
            self.items = self.items[overflow:]


def create_app(*, scratchpad_path: str | Path = SCRATCHPAD_PATH) -> FastAPI:
    app = FastAPI(title="NanoHorizon Craftax", version="0.1.0")
    path = Path(scratchpad_path)

    @app.get("/health")
    def health() -> dict[str, object]:
        return {
            "ok": True,
            "candidate_label": CANDIDATE_LABEL,
            "upstream_ready": True,
            "scratchpad_path": str(path),
        }

    @app.get("/task_info")
    def task_info() -> dict[str, object]:
        metadata = build_candidate_metadata().to_dict()
        return {
            "env_kind": "full",
            "candidate": metadata,
            "preserved_harness_surfaces": list(PRESERVED_HARNESS_SURFACES),
        }

    @app.post("/rollout")
    @app.post("/rollouts")
    def rollout(payload: dict[str, Any]) -> dict[str, object]:
        del payload
        return build_runner_summary(path)

    @app.get("/prompt")
    def prompt() -> dict[str, object]:
        return {
            "candidate_prompt": build_candidate_prompt(),
            "manifest": build_candidate_manifest(),
        }

    return app


def build_runner_summary(scratchpad_path: Path | None = None) -> dict[str, object]:
    from .runner import build_runner_summary as _build_runner_summary

    return _build_runner_summary(scratchpad_path)

