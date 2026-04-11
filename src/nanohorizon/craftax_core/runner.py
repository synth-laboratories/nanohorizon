"""CLI entrypoint for the Craftax prompt-shaping smoke path."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

from nanohorizon.baselines.prompt_opt import candidate_config, load_config, todo_scratchpad_directive

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
        default="",
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

    config_path = Path(args.config).expanduser().resolve() if args.config else None
    loaded_config: dict[str, Any]
    if config_path is not None:
        loaded_config = load_config(config_path)
        if loaded_config != candidate_config():
            raise ValueError("loaded config does not round-trip the prompt-opt candidate")
        if loaded_config["prompt"]["todo_contract"] != todo_scratchpad_directive():
            raise ValueError("todo contract mismatch")
    else:
        loaded_config = {}

    payload = _load_payload_from_stdin()
    effective_output_dir = Path(args.output_dir or loaded_config.get("output", {}).get("root_dir", ""))
    if effective_output_dir:
        effective_output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "config_path": str(config_path) if config_path is not None else None,
        "effective_output_dir": str(effective_output_dir) if effective_output_dir else None,
        "candidate_manifest": build_candidate_manifest(),
        "loaded_config": loaded_config,
        "prompt_turn": build_prompt_context_from_json(payload),
        "verification": ["config_roundtrip_smoke", "prompt_render_smoke"],
    }
    if effective_output_dir:
        (effective_output_dir / "smoke_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (effective_output_dir / "candidate_prompt.txt").write_text(
            build_candidate_prompt() + "\n",
            encoding="utf-8",
        )
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
