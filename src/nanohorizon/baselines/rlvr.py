from __future__ import annotations

import argparse
from pathlib import Path

from nanohorizon.common import Timer, ensure_dir, load_config, read_jsonl, system_info, write_json, write_text
from nanohorizon.crafter_data import build_rlvr_examples
from nanohorizon.train_lora import train_weighted_lora


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NanoHorizon Crafter RLVR baseline")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    output_dir = ensure_dir(args.output_dir or config["output"]["root_dir"])
    timer = Timer()

    examples = build_rlvr_examples(read_jsonl(config["data"]["rollout_jsonl"]))
    result = train_weighted_lora(
        base_model=config["model"]["model"],
        examples=examples,
        output_dir=output_dir / "adapter",
        learning_rate=float(config["training"]["learning_rate"]),
        epochs=int(config["training"]["epochs"]),
        max_length=int(config["training"]["max_length"]),
        max_steps=int(config["training"]["max_steps"]),
        lora_rank=int(config["training"]["lora_rank"]),
    )

    metrics = {
        "track": "rlvr_20min_2xa100_40gb",
        "baseline": "reward_weighted_lora",
        "examples_seen": result.examples_seen,
        "optimizer_steps": result.optimizer_steps,
        "mean_loss": result.mean_loss,
        "elapsed_minutes": timer.elapsed_minutes,
    }
    write_json(output_dir / "metrics.json", metrics)
    write_json(output_dir / "system_info.json", system_info())
    write_text(output_dir / "command.txt", f"python -m nanohorizon.baselines.rlvr --config {Path(args.config).resolve()}\n")


if __name__ == "__main__":
    main()
