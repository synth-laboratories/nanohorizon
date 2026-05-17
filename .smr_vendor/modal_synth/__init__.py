"""Small SMR-friendly wrapper for budget-conscious Modal usage.

This package is intentionally lightweight. It is a developer/agent convenience
layer for experiments that want a default upper-bound spend check before they
launch expensive Modal work.
"""

from .budget import DEFAULT_BUDGET_USD
from .budget import BudgetEstimate
from .budget import BudgetExceededError
from .budget import ModalBudget
from .budget import estimate_gpu_upper_bound_usd
from .sandbox import create_sandbox
from .session import ModalBudgetSession

__all__ = [
    "DEFAULT_BUDGET_USD",
    "BudgetEstimate",
    "BudgetExceededError",
    "ModalBudgetSession",
    "ModalBudget",
    "create_sandbox",
    "estimate_gpu_upper_bound_usd",
]
