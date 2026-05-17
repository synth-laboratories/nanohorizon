"""Tiny CLI for Modal budget estimation."""

from __future__ import annotations

import argparse
from decimal import Decimal

from .budget import DEFAULT_BUDGET_USD
from .budget import ModalBudget
from .budget import estimate_gpu_upper_bound_usd


def _estimate_command(args: argparse.Namespace) -> int:
    estimate = estimate_gpu_upper_bound_usd(
        gpu=args.gpu,
        gpu_count=args.count,
        timeout_seconds=args.timeout,
    )
    print(f"gpu={estimate.gpu_type}")
    print(f"count={estimate.gpu_count}")
    print(f"timeout_seconds={estimate.timeout_seconds}")
    print(f"hourly_rate_per_gpu_usd={estimate.hourly_rate_per_gpu_usd}")
    print(f"estimated_upper_bound_usd={estimate.estimated_upper_bound_usd}")
    policy = ModalBudget(budget_usd=Decimal(str(args.budget)))
    try:
        policy.assert_allows(estimate)
        print(f"budget_status=ok budget_usd={args.budget}")
        return 0
    except Exception as exc:
        print(f"budget_status=blocked budget_usd={args.budget}")
        print(f"reason={exc}")
        return 2


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m modal_synth",
        description="Budget-aware helper for SMR Modal experiments.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    estimate = subparsers.add_parser(
        "estimate",
        help="Estimate an upper-bound GPU cost and compare it to a budget.",
    )
    estimate.add_argument(
        "--gpu", required=True, help="GPU label, for example A10G or H100."
    )
    estimate.add_argument("--count", type=int, default=1, help="Number of GPUs.")
    estimate.add_argument(
        "--timeout",
        type=int,
        default=3600,
        help="Timeout in seconds used as the upper-bound runtime window.",
    )
    estimate.add_argument(
        "--budget",
        type=str,
        default=str(DEFAULT_BUDGET_USD),
        help="Budget ceiling in USD. Defaults to 10.00.",
    )
    estimate.set_defaults(handler=_estimate_command)
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
