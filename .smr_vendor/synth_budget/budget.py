"""Shared budget helpers for provider wrappers staged into SMR sandboxes."""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal, ROUND_HALF_UP


def _to_decimal(value: Decimal | float | int | str | None) -> Decimal:
    if value is None:
        return Decimal("0")
    return Decimal(str(value))


def _round_usd(value: Decimal) -> Decimal:
    return value.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


class BudgetExceededError(ValueError):
    """Raised when a provider call would exceed the configured session budget."""


@dataclass(slots=True)
class SessionBudget:
    """Simple in-memory budget tracker for one wrapper session."""

    budget_usd: Decimal | float | int | str
    spent_usd: Decimal = field(default_factory=lambda: Decimal("0.00"))

    def remaining_usd(self) -> Decimal:
        return _round_usd(_to_decimal(self.budget_usd) - self.spent_usd)

    def check_increment(
        self,
        increment_usd: Decimal | float | int | str,
        *,
        context: str | None = None,
    ) -> Decimal:
        increment = _round_usd(_to_decimal(increment_usd))
        if increment <= Decimal("0"):
            return increment
        projected_total = _round_usd(self.spent_usd + increment)
        if projected_total <= _round_usd(_to_decimal(self.budget_usd)):
            return increment
        label = str(context or "provider operation").strip() or "provider operation"
        raise BudgetExceededError(
            f"{label} rejected: projected cost ${projected_total} exceeds budget ${_round_usd(_to_decimal(self.budget_usd))}."
        )

    def commit_increment(self, increment_usd: Decimal | float | int | str) -> Decimal:
        increment = _round_usd(_to_decimal(increment_usd))
        if increment <= Decimal("0"):
            return self.spent_usd
        self.spent_usd = _round_usd(self.spent_usd + increment)
        return self.spent_usd
