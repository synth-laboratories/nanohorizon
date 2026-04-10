from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, MutableMapping


def normalize_resource_state(resource_state: Mapping[str, Any], *, max_items: int = 6) -> dict[str, str]:
    """Compact arbitrary resource state into a stable string map."""
    items = sorted(resource_state.items(), key=lambda item: item[0])[:max_items]
    normalized: dict[str, str] = {}
    for key, value in items:
        normalized[str(key)] = _stringify_value(value)
    return normalized


def _stringify_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "none"
    if isinstance(value, (int, float, str)):
        return str(value)
    if isinstance(value, Mapping):
        inner = ", ".join(f"{k}={_stringify_value(v)}" for k, v in sorted(value.items(), key=lambda item: item[0]))
        return "{" + inner + "}"
    if isinstance(value, (list, tuple, set, frozenset)):
        return "[" + ", ".join(_stringify_value(v) for v in value) + "]"
    return repr(value)


@dataclass(frozen=True)
class CraftaxCandidateMetadata:
    candidate_label: str = "Test Candidate"
    strategy: str = "custom"
    memory_capacity: int = 4
    prompt_summary: str = "Compact working-memory buffer for Craftax continuity."


@dataclass(frozen=True)
class CraftaxStepRecord:
    step_index: int
    subgoal: str
    resource_state: dict[str, str]
    action_plan: str = ""
    outcome: str = ""
    observation: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def render(self) -> str:
        parts = [
            f"step={self.step_index}",
            f"subgoal={self.subgoal}",
            f"resources={self._render_resources()}",
        ]
        if self.action_plan:
            parts.append(f"action={self.action_plan}")
        if self.outcome:
            parts.append(f"outcome={self.outcome}")
        return " | ".join(parts)

    def _render_resources(self) -> str:
        if not self.resource_state:
            return "{}"
        return "{" + ", ".join(f"{key}={value}" for key, value in self.resource_state.items()) + "}"


class WorkingMemoryBuffer:
    def __init__(self, capacity: int = 4) -> None:
        if capacity < 1:
            raise ValueError("capacity must be at least 1")
        self.capacity = capacity
        self._items: deque[CraftaxStepRecord] = deque()
        self._next_step_index = 1

    def push(
        self,
        *,
        subgoal: str,
        resource_state: Mapping[str, Any],
        action_plan: str = "",
        outcome: str = "",
        observation: str = "",
    ) -> CraftaxStepRecord:
        item = CraftaxStepRecord(
            step_index=self._next_step_index,
            subgoal=subgoal.strip() or "unspecified",
            resource_state=normalize_resource_state(resource_state),
            action_plan=action_plan.strip(),
            outcome=outcome.strip(),
            observation=observation.strip(),
        )
        self._next_step_index += 1
        self._items.append(item)
        while len(self._items) > self.capacity:
            self._items.popleft()
        return item

    def latest(self) -> CraftaxStepRecord | None:
        return self._items[-1] if self._items else None

    def snapshot(self) -> list[dict[str, Any]]:
        return [item.to_dict() for item in self._items]

    def render(self) -> str:
        if not self._items:
            return "Working memory: empty"
        lines = [f"Working memory ({len(self._items)}/{self.capacity})"]
        for item in self._items:
            lines.append(f"- {item.render()}")
        return "\n".join(lines)

