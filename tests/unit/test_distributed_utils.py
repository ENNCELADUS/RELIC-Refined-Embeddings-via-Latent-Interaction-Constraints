"""Unit tests for distributed utility helpers."""

from __future__ import annotations

import pytest
import src.utils.distributed as distributed_module


def test_distributed_barrier_nccl_uses_local_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = distributed_module.DistributedContext(
        ddp_enabled=True,
        is_distributed=True,
        rank=1,
        local_rank=2,
        world_size=4,
    )
    observed: dict[str, object] = {}

    monkeypatch.setattr(distributed_module.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(distributed_module.dist, "get_backend", lambda: "nccl")
    monkeypatch.setattr(distributed_module.torch.cuda, "is_available", lambda: True)

    def fake_barrier(*, device_ids: list[int]) -> None:
        observed["device_ids"] = device_ids

    monkeypatch.setattr(distributed_module.dist, "barrier", fake_barrier)

    distributed_module.distributed_barrier(context)

    assert observed["device_ids"] == [2]


def test_distributed_barrier_non_nccl_uses_plain_barrier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = distributed_module.DistributedContext(
        ddp_enabled=True,
        is_distributed=True,
        rank=0,
        local_rank=0,
        world_size=2,
    )
    calls: list[str] = []

    monkeypatch.setattr(distributed_module.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(distributed_module.dist, "get_backend", lambda: "gloo")
    monkeypatch.setattr(distributed_module.torch.cuda, "is_available", lambda: False)

    def fake_barrier() -> None:
        calls.append("barrier")

    monkeypatch.setattr(distributed_module.dist, "barrier", fake_barrier)

    distributed_module.distributed_barrier(context)

    assert calls == ["barrier"]


def test_initialize_distributed_reuses_existing_process_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LOCAL_RANK", "3")
    monkeypatch.setattr(distributed_module.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(distributed_module.dist, "get_rank", lambda: 2)
    monkeypatch.setattr(distributed_module.dist, "get_world_size", lambda: 4)

    context = distributed_module.initialize_distributed(ddp_enabled=True)

    assert context.ddp_enabled is True
    assert context.is_distributed is True
    assert context.rank == 2
    assert context.local_rank == 3
    assert context.world_size == 4
    assert context.owns_process_group is False


def test_cleanup_distributed_skips_destroy_for_reused_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = distributed_module.DistributedContext(
        ddp_enabled=True,
        is_distributed=True,
        rank=0,
        local_rank=0,
        world_size=4,
        owns_process_group=False,
    )
    calls: list[str] = []

    monkeypatch.setattr(distributed_module.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(
        distributed_module.dist,
        "destroy_process_group",
        lambda: calls.append("destroy"),
    )

    distributed_module.cleanup_distributed(context)

    assert calls == []


def test_cleanup_distributed_destroys_owned_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = distributed_module.DistributedContext(
        ddp_enabled=True,
        is_distributed=True,
        rank=0,
        local_rank=0,
        world_size=4,
        owns_process_group=True,
    )
    calls: list[str] = []

    monkeypatch.setattr(distributed_module.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(
        distributed_module.dist,
        "destroy_process_group",
        lambda: calls.append("destroy"),
    )

    distributed_module.cleanup_distributed(context)

    assert calls == ["destroy"]
