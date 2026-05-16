"""Budget estimation helpers for Modal-backed experiments.

This is deliberately a soft wrapper, not a control-plane failsafe. It gives
agents and developers one obvious way to estimate and cap spend before calling
into the Modal SDK directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from typing import Final

DEFAULT_BUDGET_USD: Final[Decimal] = Decimal("10.00")

# Intentionally conservative defaults. These are upper-bound planning numbers,
# not billing-authoritative reconciliation rates.
DEFAULT_GPU_HOURLY_USD: Final[dict[str, Decimal]] = {
    "CPU": Decimal("0.00"),
    "T4": Decimal("0.45"),
    "L4": Decimal("0.80"),
    "A10G": Decimal("1.20"),
    "A100": Decimal("3.90"),
    "H100": Decimal("10.00"),
}


def _normalize_gpu_label(value: str | None) -> str:
    normalized = str(value or "").strip().upper()
    return normalized or "CPU"


def _round_usd(value: Decimal) -> Decimal:
    return value.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


@dataclass(frozen=True, slots=True)
class BudgetEstimate:
    gpu_type: str
    gpu_count: int
    timeout_seconds: int
    hourly_rate_per_gpu_usd: Decimal
    estimated_upper_bound_usd: Decimal


class BudgetExceededError(ValueError):
    """Raised when a requested Modal launch exceeds the configured budget."""


@dataclass(frozen=True, slots=True)
class ModalBudget:
    """Simple budget policy for one Modal launch request."""

    budget_usd: Decimal = DEFAULT_BUDGET_USD
    strict: bool = True

    def assert_allows(self, estimate: BudgetEstimate) -> None:
        if estimate.estimated_upper_bound_usd <= self.budget_usd:
            return
        raise BudgetExceededError(
            "Modal launch rejected: estimated upper bound "
            f"${estimate.estimated_upper_bound_usd} exceeds budget ${self.budget_usd} "
            f"for {estimate.gpu_count}x {estimate.gpu_type} over {estimate.timeout_seconds}s."
        )


def estimate_gpu_upper_bound_usd(
    *,
    gpu: str | None,
    gpu_count: int = 1,
    timeout_seconds: int = 3600,
    hourly_rate_overrides_usd: dict[str, Decimal] | None = None,
) -> BudgetEstimate:
    """Estimate an upper-bound GPU spend for a single Modal launch.

    The estimate is intentionally simple and conservative:
    - use a static hourly rate table unless overridden
    - multiply by gpu_count
    - assume the full timeout window is consumed
    """

    normalized_gpu = _normalize_gpu_label(gpu)
    normalized_count = max(int(gpu_count or 1), 1)
    normalized_timeout = max(int(timeout_seconds or 0), 1)
    hourly_table = dict(DEFAULT_GPU_HOURLY_USD)
    if hourly_rate_overrides_usd:
        hourly_table.update(
            {
                _normalize_gpu_label(key): Decimal(str(value))
                for key, value in hourly_rate_overrides_usd.items()
            }
        )
    hourly_rate = hourly_table.get(normalized_gpu, Decimal("5.00"))
    estimated = _round_usd(
        hourly_rate
        * Decimal(normalized_count)
        * (Decimal(normalized_timeout) / Decimal(3600))
    )
    return BudgetEstimate(
        gpu_type=normalized_gpu,
        gpu_count=normalized_count,
        timeout_seconds=normalized_timeout,
        hourly_rate_per_gpu_usd=hourly_rate,
        estimated_upper_bound_usd=estimated,
    )
