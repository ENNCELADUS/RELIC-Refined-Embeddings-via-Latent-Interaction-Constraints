"""Compatibility shim for OHEM strategy utilities."""

from src.train.strategies.ohem import OHEMSampleStrategy, select_ohem_indices

__all__ = ["OHEMSampleStrategy", "select_ohem_indices"]
