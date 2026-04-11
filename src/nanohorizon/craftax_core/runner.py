"""Craftax runner that renders the Todo Tool scratchpad."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from typing import Sequence

from .http_shim import CraftaxHTTPShim
from .metadata import CRAFTAX_SURFACES, TODO_TOOL_STRATEGY, build_default_todo_items


def render_todo_board() -> str:
    items = build_default_todo_items()
    lines = ["# NanoHorizon Todo Board", "", TODO_TOOL_STRATEGY, ""]
    for item in items:
        lines.append(f"- [{item.key}] {item.text}")
    return "\n".join(lines)


def build_candidate_summary(shim: CraftaxHTTPShim | None = None) -> dict:
    shim = shim or CraftaxHTTPShim()
    return {
        "candidate_label": shim.candidate_label,
        "health": shim.health(),
        "task_info": shim.task_info(),
        "todo_board": render_todo_board(),
        "stable_surfaces": [asdict(surface) for surface in CRAFTAX_SURFACES],
    }


class CraftaxRunner:
    """Minimal runner facade used by the evaluation script."""

    def __init__(self, candidate_label: str = "Daytona E2E Run 3") -> None:
        self._shim = CraftaxHTTPShim(candidate_label=candidate_label)

    @property
    def shim(self) -> CraftaxHTTPShim:
        return self._shim

    def render(self) -> str:
        return render_todo_board()

    def summary(self) -> dict:
        return build_candidate_summary(self._shim)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render the NanoHorizon Craftax todo board.")
    parser.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="Output format for the candidate summary.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    runner = CraftaxRunner()
    if args.format == "json":
        print(json.dumps(runner.summary(), indent=2, sort_keys=True))
    else:
        print(runner.render())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
