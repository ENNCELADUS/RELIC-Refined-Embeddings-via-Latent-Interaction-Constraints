"""Shared training configuration dataclasses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class OptimizerConfig:
    """Optimizer parameters.

    Attributes:
        optimizer_type: Optimizer name.
        lr: Base learning rate.
        beta1: Beta1 for Adam-like optimizers.
        beta2: Beta2 for Adam-like optimizers.
        eps: Numerical stability epsilon.
        weight_decay: Weight decay coefficient.
    """

    optimizer_type: str
    lr: float
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    weight_decay: float = 0.0


@dataclass(frozen=True)
class SchedulerConfig:
    """Scheduler parameters.

    Attributes:
        scheduler_type: Scheduler name.
        max_lr: Max learning rate for schedule.
        pct_start: OneCycle warmup fraction.
        div_factor: Initial LR divisor.
        final_div_factor: Final LR divisor.
        anneal_strategy: OneCycle annealing strategy.
    """

    scheduler_type: str
    max_lr: float = 1e-3
    pct_start: float = 0.2
    div_factor: float = 25.0
    final_div_factor: float = 10000.0
    anneal_strategy: Literal["cos", "linear"] = "cos"


@dataclass(frozen=True)
class LossConfig:
    """Configuration for binary classification loss.

    Attributes:
        loss_type: Supported loss name.
        pos_weight: Positive-class weighting factor.
        label_smoothing: Label smoothing ratio in ``[0, 1)``.
    """

    loss_type: str = "bce_with_logits"
    pos_weight: float = 1.0
    label_smoothing: float = 0.0
