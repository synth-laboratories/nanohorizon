"""Compatibility package shim that points at the src-layout implementation."""

from __future__ import annotations

from pathlib import Path

_ROOT = Path(__file__).resolve().parent
_SRC_PACKAGE = _ROOT.parent / "src" / "nanohorizon"

__path__ = [str(_SRC_PACKAGE)]
