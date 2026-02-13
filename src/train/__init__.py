"""Training package exports."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = ["NoOpStrategy", "StagedUnfreezeStrategy", "Trainer"]


def __getattr__(name: str) -> Any:
    """Lazily expose training exports to avoid import cycles."""
    if name == "Trainer":
        return import_module("src.train.base").Trainer
    if name in {"NoOpStrategy", "StagedUnfreezeStrategy"}:
        strategies_module = import_module("src.train.strategies")
        return getattr(strategies_module, name)
    raise AttributeError(f"module 'src.train' has no attribute {name!r}")
