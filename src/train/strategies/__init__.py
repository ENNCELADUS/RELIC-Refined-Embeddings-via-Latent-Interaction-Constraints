"""Training strategy exports."""

from __future__ import annotations

from importlib import import_module

__all__ = [
    "NoOpStrategy",
    "StagedUnfreezeStrategy",
    "TrainingStrategy",
    "OHEMSampleStrategy",
    "select_ohem_indices",
]


def __getattr__(name: str) -> object:
    """Lazily resolve strategy symbols."""
    if name in {"NoOpStrategy", "StagedUnfreezeStrategy", "TrainingStrategy"}:
        lifecycle_module = import_module("src.train.strategies.lifecycle")
        return getattr(lifecycle_module, name)
    if name in {"OHEMSampleStrategy", "select_ohem_indices"}:
        ohem_module = import_module("src.train.strategies.ohem")
        return getattr(ohem_module, name)
    raise AttributeError(f"module 'src.train.strategies' has no attribute {name!r}")
