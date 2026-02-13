"""Unit tests for V6 sequence-native dataloader wiring."""

from __future__ import annotations

from pathlib import Path

import pytest
import src.utils.data_io as data_io
import src.utils.data_io_v6 as data_io_v6
import torch


def _write_split(path: Path) -> None:
    path.write_text("P1\tP2\t1\nP2\tP3\t0\n", encoding="utf-8")


def _write_sequence_csv(path: Path) -> None:
    path.write_text(
        "protein_id,sequence\nP1,AAAA\nP2,BBBB\nP3,CCCC\n",
        encoding="utf-8",
    )


def _v6_data_config(tmp_path: Path, sequence_csv: Path) -> dict[str, object]:
    benchmark_root = tmp_path / "benchmark"
    benchmark_root.mkdir(parents=True, exist_ok=True)
    train_split = tmp_path / "train.tsv"
    valid_split = tmp_path / "valid.tsv"
    test_split = tmp_path / "test.tsv"
    _write_split(train_split)
    _write_split(valid_split)
    _write_split(test_split)
    return {
        "run_config": {"seed": 7},
        "data_config": {
            "benchmark": {"root_dir": str(benchmark_root)},
            "embeddings": {
                "source": "esm3",
                "cache_dir": str(tmp_path / "unused_cache"),
                "sequence_file": str(sequence_csv),
                "id_column": "protein_id",
                "sequence_column": "sequence",
            },
            "dataloader": {
                "train_dataset": str(train_split),
                "valid_dataset": str(valid_split),
                "test_dataset": str(test_split),
                "num_workers": 0,
                "pin_memory": False,
                "drop_last": False,
                "sampling": {"strategy": "none"},
            },
        },
        "training_config": {"batch_size": 2},
        "model_config": {"model": "v6"},
    }


def test_build_dataloaders_routes_to_v6(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    sequence_csv = tmp_path / "sequences.csv"
    _write_sequence_csv(sequence_csv)
    config = _v6_data_config(tmp_path=tmp_path, sequence_csv=sequence_csv)
    calls: dict[str, object] = {}

    def _fake_build_dataloaders_v6(
        config: dict[str, object],
        distributed: bool = False,
        rank: int = 0,
        world_size: int = 1,
    ) -> dict[str, object]:
        calls["config"] = config
        calls["distributed"] = distributed
        calls["rank"] = rank
        calls["world_size"] = world_size
        return {"train": "train", "valid": "valid", "test": "test"}

    monkeypatch.setattr(data_io_v6, "build_dataloaders_v6", _fake_build_dataloaders_v6)

    dataloaders = data_io.build_dataloaders(
        config=config,
        distributed=True,
        rank=1,
        world_size=4,
    )

    assert dataloaders["train"] == "train"
    assert calls["config"] is config
    assert calls["distributed"] is True
    assert calls["rank"] == 1
    assert calls["world_size"] == 4


def test_build_dataloaders_v6_batch_contract(tmp_path: Path) -> None:
    sequence_csv = tmp_path / "sequences.csv"
    _write_sequence_csv(sequence_csv)
    config = _v6_data_config(tmp_path=tmp_path, sequence_csv=sequence_csv)

    dataloaders = data_io_v6.build_dataloaders_v6(config=config)
    train_batch = next(iter(dataloaders["train"]))

    assert isinstance(train_batch["seq_a"], list)
    assert isinstance(train_batch["seq_b"], list)
    assert all(isinstance(item, str) for item in train_batch["seq_a"])
    assert all(isinstance(item, str) for item in train_batch["seq_b"])
    assert isinstance(train_batch["label"], torch.Tensor)
    assert isinstance(train_batch["protein_a_id"], torch.Tensor)
    assert isinstance(train_batch["protein_b_id"], torch.Tensor)
