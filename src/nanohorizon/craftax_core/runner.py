"""CLI entrypoint for the Craftax candidate harness."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .http_shim import CompactTodoScratchpad
from .metadata import SCRATCHPAD_PATH, build_candidate_metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="nanohorizon-craftax")
    parser.add_argument(
        "--scratchpad",
        type=Path,
        default=SCRATCHPAD_PATH,
        help="Path to the compact todo scratchpad file.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=3,
        help="Maximum number of todo items kept in the scratchpad.",
    )

    subparsers = parser.add_subparsers(dest="command")

    add_parser = subparsers.add_parser("add", help="Append a todo item.")
    add_parser.add_argument("title", help="Todo item title.")

    done_parser = subparsers.add_parser("done", help="Mark a todo item done.")
    done_parser.add_argument("index", type=int, help="1-based todo index.")

    subparsers.add_parser("list", help="Render the current scratchpad.")
    subparsers.add_parser("metadata", help="Print candidate metadata as JSON.")

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "metadata":
        print(json.dumps(build_candidate_metadata().to_dict(), indent=2, sort_keys=True))
        return 0

    scratchpad = CompactTodoScratchpad.load(args.scratchpad, limit=args.limit)

    if args.command == "add":
        scratchpad.add(args.title)
        scratchpad.save()
        print(scratchpad.render())
        return 0

    if args.command == "done":
        scratchpad.mark_done(args.index - 1)
        scratchpad.save()
        print(scratchpad.render())
        return 0

    if args.command == "list" or args.command is None:
        print(json.dumps(build_candidate_metadata().to_dict(), indent=2, sort_keys=True))
        print(scratchpad.render())
        return 0

    parser.error(f"unknown command: {args.command}")
    return 2


def build_runner_summary(scratchpad_path: Path | None = None) -> dict[str, object]:
    candidate = build_candidate_metadata().to_dict()
    effective_path = scratchpad_path or SCRATCHPAD_PATH
    scratchpad_present = effective_path.exists()
    scratchpad = (
        CompactTodoScratchpad.load(effective_path, limit=candidate["scratchpad_limit"])
        if scratchpad_present
        else CompactTodoScratchpad(path=effective_path, limit=candidate["scratchpad_limit"])
    )
    return {
        "candidate": candidate,
        "scratchpad_path": str(effective_path),
        "scratchpad_present": scratchpad_present,
        "scratchpad": scratchpad.snapshot(),
        "payload": {
            "candidate": candidate,
            "scratchpad_present": scratchpad_present,
            "scratchpad": scratchpad.snapshot(),
        },
    }


if __name__ == "__main__":
    raise SystemExit(main())

