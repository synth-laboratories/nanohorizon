"""Runner entrypoint for the Server Push E2E candidate."""

from __future__ import annotations

import argparse
import json
from typing import Sequence

from .http_shim import build_task_info
from .metadata import build_server_push_e2e_metadata


def build_runner_output(candidate_label: str = "Server Push E2E") -> dict[str, object]:
    metadata = build_server_push_e2e_metadata(candidate_label=candidate_label)
    return {
        "runner": "nanohorizon.craftax_core.runner",
        "payload": build_task_info(metadata),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the Server Push E2E candidate.")
    parser.add_argument(
        "--candidate-label",
        default="Server Push E2E",
        help="Candidate label used in task info.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON instead of a human-readable summary.",
    )
    args = parser.parse_args(argv)

    output = build_runner_output(args.candidate_label)
    if args.json:
        print(json.dumps(output, indent=2, sort_keys=True))
    else:
        task_info = output["payload"]["task_info"]
        print(task_info["todo_scratchpad"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

