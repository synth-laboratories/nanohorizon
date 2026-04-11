"""Compact todo scratchpad helpers for Craftax-style prompting.

The goal is to keep a short, inspectable subgoal list that can be rendered into
agent context or logged alongside an experiment without introducing a heavier
planner dependency.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence


@dataclass(frozen=True, slots=True)
class TodoItem:
    """One line in a compact scratchpad."""

    text: str
    done: bool = False

    def render(self) -> str:
        state = "x" if self.done else " "
        return f"- [{state}] {self.text}"


@dataclass(slots=True)
class TodoScratchpad:
    """A small ordered todo list for agent self-management."""

    title: str
    items: list[TodoItem] = field(default_factory=list)

    def add(self, text: str, *, done: bool = False) -> None:
        cleaned = text.strip()
        if not cleaned:
            raise ValueError("todo text must not be empty")
        self.items.append(TodoItem(cleaned, done))

    def mark_done(self, text: str) -> bool:
        cleaned = text.strip()
        for index, item in enumerate(self.items):
            if item.text == cleaned:
                self.items[index] = TodoItem(item.text, True)
                return True
        return False

    def render(self, *, max_items: int | None = None) -> str:
        lines = [f"## {self.title}"]
        visible_items = self.items if max_items is None else self.items[:max_items]
        lines.extend(item.render() for item in visible_items)
        if max_items is not None and len(self.items) > max_items:
            lines.append(f"- [...] {len(self.items) - max_items} more")
        return "\n".join(lines)


def compact_todo_block(title: str, items: Sequence[str], *, max_items: int = 5) -> str:
    """Render a bounded markdown todo block from raw item text."""

    scratchpad = TodoScratchpad(title=title)
    for item in items:
        scratchpad.add(item)
    return scratchpad.render(max_items=max_items)
