"""CLI entrypoint for the Craftax prompt-shaping smoke path."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import sys

from .http_shim import render_prompt_turn
from .metadata import build_candidate_manifest, build_candidate_prompt


def _load_payload_from_stdin() -> dict[str, Any]:
    raw = sys.stdin.read().strip()
    if not raw:
        return {}
    return json.loads(raw)


def build_prompt_context_from_json(payload: Mapping[str, Any]) -> dict[str, Any]:
    return render_prompt_turn(
        payload.get("observation"),
        payload.get("history", ()),
        metadata=payload.get("metadata"),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/craftax_prompt_opt_qwen35_4b_full_auto_e2e.yaml",
        help="Prompt-opt candidate config to display alongside the smoke output.",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="Optional output directory to include in the smoke summary.",
    )
    parser.add_argument(
        "--write",
        type=Path,
        default=None,
        help="Optional file path to write the JSON summary to.",
    )
    parser.add_argument(
        "--prompt-out",
        type=Path,
        default=None,
        help="Optional file path to write the candidate prompt text to.",
    )
    args = parser.parse_args(argv)

    config_path = Path(args.config).expanduser().resolve()
    payload = _load_payload_from_stdin()
    summary = {
        "config_path": str(config_path),
        "output_dir": args.output_dir,
        "prompt_turn": build_prompt_context_from_json(payload),
    }
    if args.write is not None:
        args.write.parent.mkdir(parents=True, exist_ok=True)
        args.write.write_text(
            json.dumps(build_candidate_manifest(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    if args.prompt_out is not None:
        args.prompt_out.parent.mkdir(parents=True, exist_ok=True)
        args.prompt_out.write_text(build_candidate_prompt() + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
