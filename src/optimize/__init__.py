"""Optimization utilities for automated HPO and NAS-lite workflows."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.optimize.run import run_optimization

__all__ = ["run_optimization"]


def __getattr__(name: str) -> object:
    """Lazily expose package-level helpers without eager submodule imports."""
    if name == "run_optimization":
        from src.optimize.run import run_optimization as _run_optimization

        return _run_optimization
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
