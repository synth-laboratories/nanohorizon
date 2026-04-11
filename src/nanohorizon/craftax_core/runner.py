from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .http_shim import build_request_payload
from .metadata import CandidateMetadata, TodoItem


def build_candidate_prompt(goal: str, todo_items: list[str] | None = None, context: str = "") -> str:
    payload = build_request_payload(
        goal,
        todo_items=todo_items or (),
        context=context,
        metadata=CandidateMetadata(candidate_label="Daytona E2E Run 3"),
    )
    system_message = payload["messages"][0]["content"]
    user_message = payload["messages"][1]["content"]
    return f"{system_message}\n\n{user_message}"


def _load_todos(values: list[str]) -> list[TodoItem]:
    return [TodoItem(text=value) for value in values if value.strip()]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Craftax Todo Tool candidate runner")
    parser.add_argument("--goal", default="Complete the task with a compact todo scratchpad.")
    parser.add_argument("--context", default="")
    parser.add_argument("--todo", action="append", default=[], help="Add a todo entry.")
    parser.add_argument("--json", action="store_true", help="Emit the full request payload as JSON.")
    parser.add_argument("--smoke", action="store_true", help="Run a built-in smoke example.")
    parser.add_argument("--output", type=Path, default=None, help="Optional path to write JSON output.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.smoke:
        args.goal = "Demonstrate Todo Tool support in the Craftax harness."
        args.todo = ["track subgoals", "update the scratchpad", "keep the prompt compact"]
        args.context = "Smoke test"

    payload = build_request_payload(
        args.goal,
        todo_items=_load_todos(args.todo),
        context=args.context,
        metadata=CandidateMetadata(candidate_label="Daytona E2E Run 3"),
    )

    data = payload if args.json or args.output else {"prompt": build_candidate_prompt(args.goal, args.todo, args.context)}
    rendered = json.dumps(data, indent=2, sort_keys=True)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    else:
        sys.stdout.write(rendered + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
