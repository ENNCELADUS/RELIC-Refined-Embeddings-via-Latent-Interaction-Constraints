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
    selected = strategy.select(losses, epoch_index=0)
    assert selected.numel() == 2


def test_ohem_strategy_honors_warmup_epochs() -> None:
    losses = torch.tensor([0.3, 0.9, 0.1], dtype=torch.float32)
    strategy = OHEMSampleStrategy(keep_ratio=0.34, min_keep=1, warmup_epochs=2)
    warmup_selected = strategy.select(losses, epoch_index=0)
    post_warmup_selected = strategy.select(losses, epoch_index=2)
    assert warmup_selected.tolist() == [0, 1, 2]
    assert post_warmup_selected.tolist() == [1, 0]
