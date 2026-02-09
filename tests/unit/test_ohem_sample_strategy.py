"""Unit tests for OHEM selection utility."""

import torch
from src.utils.ohem_sample_strategy import OHEMSampleStrategy, select_ohem_indices


def test_select_ohem_indices_keeps_hardest() -> None:
    losses = torch.tensor([0.2, 1.5, 0.7, 2.2, 0.1], dtype=torch.float32)
    selected = select_ohem_indices(losses=losses, keep_ratio=0.4, min_keep=1)
    assert selected.tolist() == [3, 1]


def test_ohem_strategy_respects_min_keep() -> None:
    losses = torch.tensor([0.3, 0.2, 0.1], dtype=torch.float32)
    strategy = OHEMSampleStrategy(keep_ratio=0.1, min_keep=2)
    selected = strategy.select(losses)
    assert selected.numel() == 2
