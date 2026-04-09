from __future__ import annotations

import argparse
import json
from pathlib import Path

from .full_config import FullGoExploreConfig
from .full_service import FullCrafterGoExploreService


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run full local Go-Explore against the checkpoint-capable Crafter container."
    )
    parser.add_argument("--container-url", default="http://127.0.0.1:8903")
    parser.add_argument(
        "--inference-url",
        default="https://openrouter.ai/api/v1/chat/completions",
    )
    parser.add_argument("--policy-model", default="openai/gpt-4.1-mini")
    parser.add_argument("--api-key-env", default="OPENROUTER_API_KEY")
    parser.add_argument("--prompt-text", default=FullGoExploreConfig().prompt_text)
    parser.add_argument("--seed", action="append", dest="seeds", type=int, default=None)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--fresh-queries", type=int, default=2)
    parser.add_argument("--resumed-queries", type=int, default=2)
    parser.add_argument("--segment-steps", type=int, default=64)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--output-json", default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config = FullGoExploreConfig(
        container_url=args.container_url,
        inference_url=args.inference_url,
        policy_model=args.policy_model,
        api_key_env=args.api_key_env,
        prompt_text=args.prompt_text,
        seed_ids=args.seeds or [11, 29],
        max_iterations=args.iterations,
        fresh_queries_per_iteration=args.fresh_queries,
        resumed_queries_per_iteration=args.resumed_queries,
        segment_steps=args.segment_steps,
        output_dir=(
            Path(args.output_dir).expanduser().resolve()
            if args.output_dir
            else FullGoExploreConfig().output_dir
        ),
    )
    result = FullCrafterGoExploreService().run(config)
    payload = result.to_dict()
    rendered = json.dumps(payload, indent=2)
    if args.output_json:
        target = Path(args.output_json).expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(rendered, encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
