"""Online hard example mining selection logic."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


def select_ohem_indices(
    losses: torch.Tensor,
    keep_ratio: float = 0.5,
    min_keep: int = 1,
) -> torch.Tensor:
    """Select hardest-sample indices from per-sample losses.

    Args:
        losses: One-dimensional loss tensor with shape ``(batch_size,)``.
        keep_ratio: Ratio of hardest examples to keep in ``(0, 1]``.
        min_keep: Minimum number of selected samples.

    Returns:
        Tensor of indices sorted by descending loss.
    """
    if losses.dim() != 1:
        raise ValueError("losses must be a 1D tensor")
    if losses.numel() == 0:
        raise ValueError("losses must be non-empty")
    if not 0.0 < keep_ratio <= 1.0:
        raise ValueError("keep_ratio must be in (0, 1]")
    if min_keep <= 0:
        raise ValueError("min_keep must be positive")

    keep_count = max(min_keep, int(math.ceil(losses.numel() * keep_ratio)))
    keep_count = min(keep_count, int(losses.numel()))
    detached = losses.detach()
    return torch.topk(detached, k=keep_count, largest=True, sorted=True).indices


@dataclass(frozen=True)
class OHEMSampleStrategy:
    """Configuration for OHEM sample selection.

    Attributes:
        keep_ratio: Ratio of hardest examples to keep.
        min_keep: Minimum selected sample count.
    """

    keep_ratio: float = 0.5
    min_keep: int = 1

    def select(self, losses: torch.Tensor) -> torch.Tensor:
        """Select hard-sample indices.

        Args:
            losses: Per-sample loss tensor.

        Returns:
            Indices of selected hard samples.
        """
        return select_ohem_indices(
            losses=losses,
            keep_ratio=self.keep_ratio,
            min_keep=self.min_keep,
        )
