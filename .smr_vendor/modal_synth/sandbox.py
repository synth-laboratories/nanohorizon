"""Modal sandbox helpers with a default soft budget check."""

from __future__ import annotations

from decimal import Decimal
from typing import Any

from .budget import DEFAULT_BUDGET_USD
from .budget import BudgetEstimate
from .budget import ModalBudget
from .budget import estimate_gpu_upper_bound_usd


def create_sandbox(
    *sandbox_args: Any,
    gpu: str | None = None,
    gpu_count: int = 1,
    timeout: int = 3600,
    budget_usd: Decimal | float | int | str = DEFAULT_BUDGET_USD,
    strict_budget: bool = True,
    hourly_rate_overrides_usd: dict[str, Decimal] | None = None,
    **sandbox_kwargs: Any,
) -> tuple[Any, BudgetEstimate]:
    """Create a Modal sandbox after a default budget check.

    Returns:
        (sandbox, estimate)

    Example:
        sandbox, estimate = create_sandbox(
            "bash", "-lc", "python train.py",
            gpu="A10G",
            timeout=1800,
        )
    """

    try:
        import modal
    except Exception as exc:  # pragma: no cover - runtime dependency
        raise RuntimeError(
            "Modal client is not available in this environment."
        ) from exc

    estimate = estimate_gpu_upper_bound_usd(
        gpu=gpu,
        gpu_count=gpu_count,
        timeout_seconds=timeout,
        hourly_rate_overrides_usd=hourly_rate_overrides_usd,
    )
    policy = ModalBudget(
        budget_usd=Decimal(str(budget_usd)),
        strict=bool(strict_budget),
    )
    if policy.strict:
        policy.assert_allows(estimate)

    launch_kwargs = dict(sandbox_kwargs)
    launch_kwargs["timeout"] = timeout
    if gpu:
        launch_kwargs["gpu"] = gpu

    sandbox = modal.Sandbox.create(
        *sandbox_args,
        **launch_kwargs,
    )
    return sandbox, estimate
