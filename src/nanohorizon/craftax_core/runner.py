"""Command-line entrypoint for the Craftax candidate scaffold."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

if __package__ in {None, ""}:  # pragma: no cover - direct file execution fallback
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from nanohorizon.craftax_core.http_shim import (  # noqa: E402
    build_health_payload,
    build_rollout_payload,
    build_task_info,
)
from nanohorizon.craftax_core.metadata import default_metadata  # noqa: E402


def _render_text() -> str:
    metadata = default_metadata()
    lines = [
        f"Candidate: {metadata.candidate_label}",
        f"Strategy: {metadata.primary_strategy}",
        f"Objective: {metadata.objective}",
        "",
        metadata.todo_block(),
        "",
        "Health:",
        json.dumps(build_health_payload(metadata), indent=2, sort_keys=True),
        "",
        "Task info:",
        json.dumps(build_task_info(metadata), indent=2, sort_keys=True),
        "",
        "Rollout:",
        json.dumps(build_rollout_payload(metadata), indent=2, sort_keys=True),
    ]
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--format",
        choices=("json", "text"),
        default="json",
        help="Output format for the scaffold payload.",
    )
    parser.add_argument(
        "--emit",
        choices=("health", "task-info", "rollout", "all"),
        default="all",
        help="Select which payload to emit in json mode.",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    metadata = default_metadata()

    if args.format == "text":
        print(_render_text())
        return 0

    payload: dict[str, object]
    if args.emit == "health":
        payload = build_health_payload(metadata)
    elif args.emit == "task-info":
        payload = build_task_info(metadata)
    elif args.emit == "rollout":
        payload = build_rollout_payload(metadata)
    else:
        payload = {
            "health": build_health_payload(metadata),
            "task_info": build_task_info(metadata),
            "rollout": build_rollout_payload(metadata),
        }
    json.dump(payload, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(main())
