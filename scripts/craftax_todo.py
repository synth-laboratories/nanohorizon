#!/usr/bin/env python3
"""Manage a compact Craftax scratchpad in Markdown."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


DEFAULT_PATH = Path("craftax_todo.md")
DEFAULT_ITEMS = (
    "State the current hypothesis in one line.",
    "Run the smallest verifier that can falsify it.",
    "Record the result and handoff evidence.",
)


@dataclass
class TodoBoard:
    items: list[tuple[bool, str]]

    @classmethod
    def from_text(cls, text: str) -> "TodoBoard":
        items: list[tuple[bool, str]] = []
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("- [ ] "):
                items.append((False, stripped[6:]))
            elif stripped.startswith("- [x] "):
                items.append((True, stripped[6:]))
        return cls(items=items)

    def render(self) -> str:
        lines = ["# Craftax Todo", ""]
        if not self.items:
            lines.append("- [ ] Add the first task.")
        else:
            for done, text in self.items:
                mark = "x" if done else " "
                lines.append(f"- [{mark}] {text}")
        lines.append("")
        return "\n".join(lines)

    def with_defaults(self) -> "TodoBoard":
        if self.items:
            return self
        return TodoBoard(items=[(False, item) for item in DEFAULT_ITEMS])

    def mark_done(self, index: int) -> None:
        done, text = self.items[index]
        self.items[index] = (True, text)

    def add(self, text: str) -> None:
        self.items.append((False, text))


def read_board(path: Path) -> TodoBoard:
    if not path.exists():
        return TodoBoard(items=[]).with_defaults()
    return TodoBoard.from_text(path.read_text()).with_defaults()


def write_board(path: Path, board: TodoBoard) -> None:
    path.write_text(board.render())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_path_argument(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument("--path", type=Path, default=DEFAULT_PATH)

    show_parser = subparsers.add_parser("show", help="print the board")
    add_path_argument(show_parser)

    init_parser = subparsers.add_parser("init", help="create the default board if needed")
    add_path_argument(init_parser)

    add_parser = subparsers.add_parser("add", help="append a todo item")
    add_path_argument(add_parser)
    add_parser.add_argument("text")

    done_parser = subparsers.add_parser("done", help="mark an item complete")
    add_path_argument(done_parser)
    done_parser.add_argument("index", type=int)

    return parser


def cmd_show(path: Path) -> None:
    print(read_board(path).render(), end="")


def cmd_init(path: Path) -> None:
    if path.exists():
        return
    write_board(path, TodoBoard(items=[]).with_defaults())


def cmd_add(path: Path, text: str) -> None:
    board = read_board(path)
    board.add(text)
    write_board(path, board)


def cmd_done(path: Path, index: int) -> None:
    board = read_board(path)
    if index < 1 or index > len(board.items):
        raise SystemExit(f"index out of range: {index}")
    board.mark_done(index - 1)
    write_board(path, board)


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "show":
        cmd_show(args.path)
    elif args.command == "init":
        cmd_init(args.path)
    elif args.command == "add":
        cmd_add(args.path, args.text)
    elif args.command == "done":
        cmd_done(args.path, args.index)
    else:
        raise SystemExit(f"unknown command: {args.command}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
