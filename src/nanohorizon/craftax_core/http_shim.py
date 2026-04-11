from __future__ import annotations

from dataclasses import asdict
from typing import Sequence

from .metadata import CandidateMetadata, TodoItem, build_todo_summary, normalize_todos

TODO_TOOL_NAME = "todo_scratchpad"


def build_todo_tool_schema() -> dict[str, object]:
    return {
        "type": "function",
        "function": {
            "name": TODO_TOOL_NAME,
            "description": "Maintain a compact scratchpad of subgoals while solving the task.",
            "parameters": {
                "type": "object",
                "properties": {
                    "items": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Ordered TODO entries to keep in memory.",
                    }
                },
                "required": ["items"],
                "additionalProperties": False,
            },
        },
    }


def build_request_payload(
    goal: str,
    *,
    todo_items: Sequence[TodoItem | str] = (),
    context: str = "",
    metadata: CandidateMetadata | None = None,
) -> dict[str, object]:
    todos = normalize_todos(todo_items)
    meta = metadata or CandidateMetadata(candidate_label="Daytona E2E Run 3")
    system_parts = [
        "You are a Craftax agent using a compact Todo Tool scratchpad.",
        "Keep the todo list short, current, and action-oriented.",
        "Update the list when subgoals are completed or reprioritized.",
        "Prefer one active target at a time; replace stale items when progress stalls.",
    ]
    if context.strip():
        system_parts.append(f"Context: {context.strip()}")

    messages = [
        {"role": "system", "content": "\n".join(system_parts)},
        {
            "role": "user",
            "content": f"Goal: {goal.strip()}\n\n{build_todo_summary(todos, meta.max_todos)}",
        },
    ]
    return {
        "candidate": meta.to_dict(),
        "messages": messages,
        "tools": [build_todo_tool_schema()],
        "scratchpad": [asdict(item) for item in todos[: meta.max_todos]],
    }
