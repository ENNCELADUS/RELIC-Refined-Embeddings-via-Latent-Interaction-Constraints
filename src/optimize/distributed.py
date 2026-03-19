"""Distributed coordination helpers for multi-GPU optimization."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, cast

from src.optimize.search_space import SearchParameter
from src.optimize.trial_runner import (
    Direction,
    RunPipelineFn,
    execute_trial,
    run_best_full_pipeline,
)
from src.utils.config import ConfigDict
from src.utils.distributed import DistributedContext, distributed_barrier, dist


@dataclass(frozen=True)
class OptimizationCommand:
    """One distributed optimization command broadcast from rank 0."""

    kind: str
    trial_number: int | None = None
    sampled_values: dict[str, object] = field(default_factory=dict)
    best_values: dict[str, object] = field(default_factory=dict)


class OptimizationChannel(Protocol):
    """Small channel interface for distributed optimization commands."""

    def send(self, command: OptimizationCommand) -> None:
        """Broadcast one command from the main process."""

    def receive(self) -> OptimizationCommand:
        """Receive one command on a worker process."""

    def barrier(self) -> None:
        """Synchronize all ranks after one coordinated action."""


class TorchDistributedOptimizationChannel:
    """Broadcast-based command channel backed by ``torch.distributed``."""

    def __init__(self, distributed_context: DistributedContext) -> None:
        self.distributed_context = distributed_context

    def send(self, command: OptimizationCommand) -> None:
        """Broadcast one command from rank 0."""
        payload: list[object] = [command]
        dist.broadcast_object_list(payload, src=0)

    def receive(self) -> OptimizationCommand:
        """Receive one command on a non-zero rank."""
        payload: list[object] = [None]
        dist.broadcast_object_list(payload, src=0)
        command = payload[0]
        if not isinstance(command, OptimizationCommand):
            raise TypeError("Expected OptimizationCommand payload from distributed broadcast")
        return command

    def barrier(self) -> None:
        """Barrier all ranks."""
        distributed_barrier(self.distributed_context)


def run_distributed_worker_loop(
    *,
    base_config: ConfigDict,
    search_space: list[SearchParameter],
    study_name: str,
    objective_metric: str,
    direction: Direction,
    execution_cfg: ConfigDict,
    run_pipeline_fn: RunPipelineFn,
    channel: OptimizationChannel,
    execute_trial_fn=execute_trial,
    run_best_full_pipeline_fn=run_best_full_pipeline,
) -> None:
    """Run worker-rank loop for distributed HPO coordination."""
    while True:
        command = channel.receive()
        if command.kind == "stop":
            return
        if command.kind == "run_trial":
            if command.trial_number is None:
                raise ValueError("run_trial command requires trial_number")
            execute_trial_fn(
                base_config=base_config,
                search_space=search_space,
                sampled_values=command.sampled_values,
                study_name=study_name,
                trial_number=command.trial_number,
                objective_metric=objective_metric,
                direction=direction,
                execution_cfg=execution_cfg,
                run_pipeline_fn=run_pipeline_fn,
            )
            channel.barrier()
            continue
        if command.kind == "run_best_pipeline":
            run_best_full_pipeline_fn(
                base_config=base_config,
                search_space=search_space,
                best_values=command.best_values,
                study_name=study_name,
                run_pipeline_fn=run_pipeline_fn,
                ddp_per_trial=True,
            )
            channel.barrier()
            continue
        raise ValueError(f"Unsupported optimization command: {command.kind}")


def build_optimization_channel(
    distributed_context: DistributedContext,
) -> OptimizationChannel | None:
    """Return default distributed channel when running multi-process optimization."""
    if not distributed_context.is_distributed:
        return None
    return TorchDistributedOptimizationChannel(distributed_context)


__all__ = [
    "OptimizationChannel",
    "OptimizationCommand",
    "TorchDistributedOptimizationChannel",
    "build_optimization_channel",
    "run_distributed_worker_loop",
]
