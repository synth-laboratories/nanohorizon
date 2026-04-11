"""Static Craftax metadata for the NanoHorizon Todo Tool candidate."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class CraftaxSurface:
    """Stable surface that downstream tooling can inspect."""

    path: str
    contract: str


@dataclass(frozen=True)
class TodoItem:
    """One compact subgoal in the scratchpad."""

    key: str
    text: str


TODO_TOOL_STRATEGY = (
    "Use a compact todo scratchpad to keep subgoals visible while preserving the "
    "existing Craftax contract."
)

CRAFTAX_SURFACES: Tuple[CraftaxSurface, ...] = (
    CraftaxSurface(path="docs/task-craftax.md", contract="task statement and strategy notes"),
    CraftaxSurface(path="src/nanohorizon/craftax_core/http_shim.py", contract="rollout/task-info HTTP shim"),
    CraftaxSurface(path="src/nanohorizon/craftax_core/runner.py", contract="candidate runner and prompt assembly"),
    CraftaxSurface(path="src/nanohorizon/craftax_core/metadata.py", contract="stable surface metadata and todo items"),
    CraftaxSurface(path="scripts/run_craftax_model_eval.sh", contract="uv-backed evaluation entrypoint"),
)


def build_default_todo_items() -> Tuple[TodoItem, ...]:
    """Return the smallest honest todo list for the candidate."""

    return (
        TodoItem(key="inspect", text="Inspect the task and preserve the stable Craftax surfaces."),
        TodoItem(key="scratchpad", text="Render a compact todo board that the agent can update while it works."),
        TodoItem(key="verify", text="Run the local verifier and record the result before marking the branch ready."),
        TodoItem(key="handoff", text="Document the behavior and keep the commit and PR reviewable."),
    )
