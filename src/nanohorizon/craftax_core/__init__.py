"""Craftax harness helpers for NanoHorizon."""

from __future__ import annotations

from typing import Any

from .http_shim import CraftaxHTTPShim
from .metadata import CRAFTAX_SURFACES, TODO_TOOL_STRATEGY, build_default_todo_items

__all__ = [
    "CRAFTAX_SURFACES",
    "CraftaxHTTPShim",
    "CraftaxRunner",
    "TODO_TOOL_STRATEGY",
    "build_default_todo_items",
]


def __getattr__(name: str) -> Any:
    if name == "CraftaxRunner":
        from .runner import CraftaxRunner

        return CraftaxRunner
    raise AttributeError(name)
