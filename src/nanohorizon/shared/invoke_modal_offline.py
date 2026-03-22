from __future__ import annotations

import argparse
import json
import os

import modal

APP_NAME = os.getenv("NANOHORIZON_MODAL_OFFLINE_APP_NAME", "nanohorizon-crafter-offline")
FUNCTION_NAME = "run"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Invoke the deployed NanoHorizon offline Modal app")
    parser.add_argument(
        "--config",
        default="configs/crafter_offline_reference.yaml",
    )
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--teacher-model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--teacher-api-key", default="")
    parser.add_argument("--teacher-enforce-eager", type=int, default=0)
    parser.add_argument("--teacher-startup-attempts", type=int, default=240)
    parser.add_argument("--teacher-startup-sleep-seconds", type=int, default=2)
    parser.add_argument("--min-teacher-reward", type=float, default=None)
    parser.add_argument("--max-teacher-rows", type=int, default=None)
    parser.add_argument("--filter-collect-wood", type=int, default=None)
    parser.add_argument("--crafter-container-url", default="")
    parser.add_argument("--crafter-container-worker-token", default="")
    parser.add_argument("--teacher-inference-url", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    function = modal.Function.from_name(APP_NAME, FUNCTION_NAME)
    with modal.enable_output():
        result = function.remote(
            config=args.config,
            output_dir=args.output_dir,
            teacher_model=args.teacher_model,
            teacher_api_key=args.teacher_api_key,
            teacher_enforce_eager=args.teacher_enforce_eager,
            teacher_startup_attempts=args.teacher_startup_attempts,
            teacher_startup_sleep_seconds=args.teacher_startup_sleep_seconds,
            min_teacher_reward=args.min_teacher_reward,
            max_teacher_rows=args.max_teacher_rows,
            filter_collect_wood=args.filter_collect_wood,
            crafter_container_url=args.crafter_container_url,
            crafter_container_worker_token=args.crafter_container_worker_token,
            teacher_inference_url=args.teacher_inference_url,
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
