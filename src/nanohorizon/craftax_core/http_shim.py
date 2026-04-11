"""Compact todo scratchpad used during Craftax video validation runs."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List
import json


@dataclass(slots=True)
class TodoItem:
    """One bounded todo entry."""

    title: str
    done: bool = False

    def to_dict(self) -> dict[str, object]:
        return {"title": self.title, "done": self.done}

    @classmethod
    def from_dict(cls, raw: dict[str, object]) -> "TodoItem":
        return cls(title=str(raw["title"]), done=bool(raw.get("done", False)))


@dataclass(slots=True)
class CompactTodoScratchpad:
    """Small persistent todo list for the agent workflow.

    The scratchpad intentionally stays tiny: it is meant to support a video
    validation run, not become a generic task database.
    """

    path: Path
    limit: int = 6
    items: List[TodoItem] = field(default_factory=list)

    @classmethod
    def load(cls, path: str | Path, limit: int = 6) -> "CompactTodoScratchpad":
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
