"""Compact todo scratchpad contract for the Craftax candidate."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from nanohorizon.baselines.prompt_opt import (
    CANDIDATE_LABEL,
    TODO_SCRATCHPAD_REQUIREMENTS,
    build_reflection_prompt as build_contract_reflection_prompt,
    build_seed_prompt as build_contract_seed_prompt,
)


DEFAULT_CANDIDATE_LABEL = CANDIDATE_LABEL
DEFAULT_OBJECTIVE = (
    "Improve the Craftax approach with a compact todo/scratchpad "
    "contract that stays fresh across turns."
)
DEFAULT_TODO_ITEMS = (
    "Confirm the task constraints and keep the change narrow.",
    "Render a compact server-pushed todo scratchpad.",
    "Surface the scratchpad through task info and the runner.",
)
SCRATCHPAD_MODE = "compact-three-item"


@dataclass(frozen=True, slots=True)
class TodoItem:
    """Single scratchpad item with a completion flag."""

    text: str
    done: bool = False

    def render(self) -> str:
        marker = "x" if self.done else " "
        return f"- [{marker}] {self.text}"


@dataclass(frozen=True, slots=True)
class CraftaxCandidateMetadata:
    """Minimal metadata bundle for the Server Push E2E candidate."""

    candidate_label: str = DEFAULT_CANDIDATE_LABEL
    objective: str = DEFAULT_OBJECTIVE
    scratchpad_mode: str = SCRATCHPAD_MODE
    todo_items: tuple[TodoItem, ...] = tuple(
        TodoItem(text=item) for item in DEFAULT_TODO_ITEMS
    )

    def render_todo_scratchpad(self) -> str:
        lines = ["## Todo Scratchpad", ""]
        lines.extend(item.render() for item in self.todo_items)
        return "\n".join(lines)

    def to_task_info(self) -> dict[str, object]:
        return {
            "candidate_label": self.candidate_label,
            "objective": self.objective,
            "scratchpad_mode": self.scratchpad_mode,
            "scratchpad_requirements": list(TODO_SCRATCHPAD_REQUIREMENTS),
            "todo_item_count": len(self.todo_items),
            "todo_items": [item.text for item in self.todo_items],
            "todo_scratchpad": self.render_todo_scratchpad(),
        }


def refresh_todo_items(
    todo_items: Iterable[str],
    *,
    completed_items: Iterable[str] = (),
    next_action: str | None = None,
) -> tuple[str, ...]:
    """Keep the scratchpad compact and server-pushed."""

    completed = {item.strip() for item in completed_items if item and item.strip()}
    live_items = [item for item in todo_items if item not in completed]
    if next_action:
        live_items = [item for item in live_items if item != next_action]
        live_items.append(next_action)
    return tuple(live_items[-3:])


def build_server_push_e2e_metadata(
    candidate_label: str = DEFAULT_CANDIDATE_LABEL,
    objective: str = DEFAULT_OBJECTIVE,
    todo_items: Iterable[str] = DEFAULT_TODO_ITEMS,
) -> CraftaxCandidateMetadata:
    return CraftaxCandidateMetadata(
        candidate_label=candidate_label,
        objective=objective,
        todo_items=tuple(TodoItem(text=item) for item in todo_items),
    )


def build_seed_prompt(metadata: CraftaxCandidateMetadata | None = None) -> str:
    _ = metadata or build_server_push_e2e_metadata()
    return build_contract_seed_prompt()


def build_reflection_prompt(metadata: CraftaxCandidateMetadata | None = None) -> str:
    _ = metadata or build_server_push_e2e_metadata()
    return build_contract_reflection_prompt()
