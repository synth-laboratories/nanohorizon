"""Compact persisted todo/scratchpad support for NanoHorizon runs.

The shared-history candidate needs a lightweight way to keep subgoals visible
while the agent iterates on a run. This module keeps the surface narrow:

- `TodoItem` models one subgoal with stable serialization.
- `TodoBoard` stores the ordered list, supports status transitions, and
  round-trips cleanly through dictionaries.

The implementation deliberately avoids framework dependencies so it can be
reused in tooling, tests, or future harness hooks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Iterable, Literal, Mapping, cast


TodoStatus = Literal["todo", "doing", "done"]


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _coerce_text(value: Any) -> str:
    text = str(value or "").strip()
    return text


def _coerce_metadata(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    return {}


def _coerce_status(value: Any) -> TodoStatus:
    status = _coerce_text(value) or "todo"
    if status not in {"todo", "doing", "done"}:
        status = "todo"
    return cast(TodoStatus, status)


@dataclass(slots=True)
class TodoItem:
    """One persisted subgoal in the shared history scratchpad."""

    title: str
    status: TodoStatus = "todo"
    item_id: str | None = None
    owner: str | None = None
    created_at: str = field(default_factory=_utc_now_iso)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "title": self.title,
            "status": self.status,
            "owner": self.owner,
            "created_at": self.created_at,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "TodoItem":
        return cls(
            title=_coerce_text(raw.get("title")) or "",
            status=_coerce_status(raw.get("status")),
            item_id=_coerce_text(raw.get("item_id")) or None,
            owner=_coerce_text(raw.get("owner")) or None,
            created_at=_coerce_text(raw.get("created_at")) or _utc_now_iso(),
            metadata=_coerce_metadata(raw.get("metadata")),
        )


@dataclass(slots=True)
class TodoBoard:
    """Ordered todo list that can be serialized into run state."""

    board_id: str = "shared-history"
    items: list[TodoItem] = field(default_factory=list)

    def add_item(
        self,
        title: str,
        *,
        owner: str | None = None,
        status: TodoStatus = "todo",
        metadata: Mapping[str, Any] | None = None,
    ) -> TodoItem:
        item_id = f"{self.board_id}:item_{len(self.items) + 1:04d}"
        item = TodoItem(
            title=_coerce_text(title),
            owner=_coerce_text(owner) or None,
            status=_coerce_status(status),
            item_id=item_id,
            metadata=_coerce_metadata(metadata),
        )
        self.items.append(item)
        return item

    def open_items(self) -> list[TodoItem]:
        return [item for item in self.items if item.status != "done"]

    def next_item(self) -> TodoItem | None:
        for item in self.items:
            if item.status != "done":
                return item
        return None

    def mark_doing(self, item_id: str) -> TodoItem:
        item = self._require_item(item_id)
        item.status = "doing"
        return item

    def mark_done(self, item_id: str) -> TodoItem:
        item = self._require_item(item_id)
        item.status = "done"
        return item

    def _require_item(self, item_id: str) -> TodoItem:
        needle = _coerce_text(item_id)
        for item in self.items:
            if item.item_id == needle:
                return item
        raise KeyError(f"unknown todo item: {item_id}")

    def to_dict(self) -> dict[str, Any]:
        items = [item.to_dict() for item in self.items]
        return {
            "board_id": self.board_id,
            "items": items,
            "project_todo": items,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "TodoBoard":
        items: list[TodoItem] = []
        rows = raw.get("items")
        if not isinstance(rows, list) or not rows:
            rows = raw.get("project_todo")
        if isinstance(rows, list):
            for row in rows:
                if isinstance(row, Mapping):
                    items.append(TodoItem.from_dict(row))
        return cls(
            board_id=_coerce_text(raw.get("board_id")) or "shared-history",
            items=items,
        )

    @classmethod
    def from_titles(
        cls,
        titles: Iterable[str],
        *,
        board_id: str = "shared-history",
    ) -> "TodoBoard":
        board = cls(board_id=board_id)
        for title in titles:
            board.add_item(title)
        return board
