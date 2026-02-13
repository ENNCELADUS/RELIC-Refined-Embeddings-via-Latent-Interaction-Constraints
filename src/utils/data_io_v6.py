"""Sequence-native dataloaders for V6 (ESM3+LoRA) model runs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from src.embed.io import (
    _collect_required_protein_ids,
    _discover_sequences,
    _resolve_sequence_search_roots,
)
from src.utils.config import ConfigDict, as_bool, as_float, as_int, as_str, get_section
from src.utils.data_samplers import StagedOHEMBatchSampler


def _optional_string(value: object) -> str | None:
    """Return stripped string value when present."""
    if isinstance(value, str):
        stripped = value.strip()
        return stripped if stripped else None
    return None


@dataclass(frozen=True)
class PairRecord:
    """Single protein pair interaction record."""

    protein_a: str
    protein_b: str
    label: int


def _read_pair_records(file_path: Path) -> list[PairRecord]:
    """Read pair records from tab-separated dataset files."""
    records: list[PairRecord] = []
    with file_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = [part.strip() for part in line.strip().split("\t")]
            if len(parts) < 2:
                continue
            if not parts[0] or not parts[1]:
                continue
            if len(parts) >= 3 and parts[2]:
                try:
                    label = int(float(parts[2]))
                except ValueError:
                    continue
            else:
                label = 1
            records.append(PairRecord(parts[0], parts[1], label))
    if not records:
        raise ValueError(f"No valid PPI records found in {file_path}")
    return records


def _resolve_split_paths(config: ConfigDict) -> dict[str, Path]:
    """Resolve and validate train/valid/test split file paths."""
    data_cfg = get_section(config, "data_config")
    benchmark_cfg = get_section(data_cfg, "benchmark")
    dataloader_cfg = get_section(data_cfg, "dataloader")

    benchmark_root = Path(str(benchmark_cfg.get("root_dir", "")))
    if not benchmark_root.exists():
        raise FileNotFoundError(f"Benchmark root does not exist: {benchmark_root}")

    split_paths = {
        "train": Path(str(dataloader_cfg.get("train_dataset", ""))),
        "valid": Path(str(dataloader_cfg.get("valid_dataset", ""))),
        "test": Path(str(dataloader_cfg.get("test_dataset", ""))),
    }
    for split_name, split_path in split_paths.items():
        if not split_path.exists():
            raise FileNotFoundError(f"{split_name} dataset path not found: {split_path}")
    return split_paths


def _sequence_lookup_overrides(config: ConfigDict) -> tuple[Path | None, str | None, str | None]:
    """Resolve optional sequence lookup overrides from config."""
    data_cfg = get_section(config, "data_config")
    embeddings_raw = data_cfg.get("embeddings", {})
    if not isinstance(embeddings_raw, dict):
        raise ValueError("data_config.embeddings must be a mapping")
    sequences_raw = data_cfg.get("sequences", {})
    if not isinstance(sequences_raw, dict):
        raise ValueError("data_config.sequences must be a mapping")

    sequence_file_raw = _optional_string(sequences_raw.get("file_path")) or _optional_string(
        embeddings_raw.get("sequence_file")
    )
    id_column = _optional_string(sequences_raw.get("id_column")) or _optional_string(
        embeddings_raw.get("id_column")
    )
    sequence_column = _optional_string(sequences_raw.get("sequence_column")) or _optional_string(
        embeddings_raw.get("sequence_column")
    )

    sequence_file = Path(sequence_file_raw) if sequence_file_raw is not None else None
    return sequence_file, id_column, sequence_column


class SequencePairDataset(Dataset[dict[str, object]]):
    """PPI dataset that returns raw sequence pairs for sequence-native models."""

    def __init__(self, file_path: Path, sequences: dict[str, str]) -> None:
        self._records = _read_pair_records(file_path=file_path)
        self._sequences = dict(sequences)
        proteins = sorted(
            {record.protein_a for record in self._records}
            | {record.protein_b for record in self._records}
        )
        self._protein_to_id = {protein: index for index, protein in enumerate(proteins)}

        required_ids = {record.protein_a for record in self._records}
        required_ids.update(record.protein_b for record in self._records)
        missing_ids = sorted(
            protein_id for protein_id in required_ids if protein_id not in self._sequences
        )
        if missing_ids:
            preview = ", ".join(missing_ids[:10])
            raise FileNotFoundError(
                f"Sequence lookup is missing proteins required by {file_path}: "
                f"{preview} (missing={len(missing_ids)})"
            )

    def __len__(self) -> int:
        """Return dataset size."""
        return len(self._records)

    def labels(self) -> list[int]:
        """Return binary labels for all records."""
        return [int(record.label) for record in self._records]

    def __getitem__(self, index: int) -> dict[str, object]:
        """Return one dataset example."""
        record = self._records[index]
        return {
            "seq_a": self._sequences[record.protein_a],
            "seq_b": self._sequences[record.protein_b],
            "label": torch.tensor(float(record.label), dtype=torch.float32),
            "protein_a_id": torch.tensor(self._protein_to_id[record.protein_a], dtype=torch.long),
            "protein_b_id": torch.tensor(self._protein_to_id[record.protein_b], dtype=torch.long),
        }


def _collate_sequence_batch(batch: list[dict[str, object]]) -> dict[str, object]:
    """Collate sequence-pair examples into a batched dictionary."""
    seq_a = [str(sample["seq_a"]) for sample in batch]
    seq_b = [str(sample["seq_b"]) for sample in batch]

    labels: list[torch.Tensor] = []
    protein_a_ids: list[torch.Tensor] = []
    protein_b_ids: list[torch.Tensor] = []
    for sample in batch:
        label_value = sample.get("label")
        protein_a_id_value = sample.get("protein_a_id")
        protein_b_id_value = sample.get("protein_b_id")
        if not isinstance(label_value, torch.Tensor):
            raise TypeError("label must be a torch.Tensor")
        if not isinstance(protein_a_id_value, torch.Tensor):
            raise TypeError("protein_a_id must be a torch.Tensor")
        if not isinstance(protein_b_id_value, torch.Tensor):
            raise TypeError("protein_b_id must be a torch.Tensor")
        labels.append(label_value)
        protein_a_ids.append(protein_a_id_value)
        protein_b_ids.append(protein_b_id_value)

    return {
        "seq_a": seq_a,
        "seq_b": seq_b,
        "label": torch.stack(labels, dim=0),
        "protein_a_id": torch.stack(protein_a_ids, dim=0),
        "protein_b_id": torch.stack(protein_b_ids, dim=0),
    }


def _build_split_loader_v6(
    split_path: Path,
    config: ConfigDict,
    sequences: dict[str, str],
    seed: int,
    shuffle: bool,
    distributed: bool,
    rank: int,
    world_size: int,
) -> DataLoader[dict[str, object]]:
    """Build one split dataloader for sequence-native training/evaluation."""
    training_cfg = get_section(config, "training_config")
    data_cfg = get_section(config, "data_config")
    dataloader_cfg = get_section(data_cfg, "dataloader")

    batch_size = as_int(training_cfg.get("batch_size", 8), "training_config.batch_size")
    num_workers = as_int(
        dataloader_cfg.get("num_workers", 0),
        "data_config.dataloader.num_workers",
    )
    pin_memory = as_bool(
        dataloader_cfg.get("pin_memory", False),
        "data_config.dataloader.pin_memory",
    )
    drop_last = as_bool(dataloader_cfg.get("drop_last", False), "data_config.dataloader.drop_last")

    dataset = SequencePairDataset(file_path=split_path, sequences=sequences)
    sampler: DistributedSampler[dict[str, object]] | None = None
    batch_sampler: StagedOHEMBatchSampler | None = None
    should_shuffle = shuffle

    sampling_raw = dataloader_cfg.get("sampling", {})
    if not isinstance(sampling_raw, dict):
        raise ValueError("data_config.dataloader.sampling must be a mapping")
    sampling_cfg = sampling_raw
    sampling_strategy = as_str(
        sampling_cfg.get("strategy", "none"),
        "data_config.dataloader.sampling.strategy",
    ).lower()

    if shuffle and sampling_strategy == "ohem":
        labels = dataset.labels()
        pos_count = sum(labels)
        neg_count = len(labels) - pos_count
        natural_ratio = float(neg_count) / float(max(1, pos_count))
        batch_sampler = StagedOHEMBatchSampler(
            labels=labels,
            batch_size=batch_size,
            warmup_pos_neg_ratio=as_float(
                sampling_cfg.get("warmup_pos_neg_ratio", natural_ratio),
                "data_config.dataloader.sampling.warmup_pos_neg_ratio",
            ),
            warmup_epochs=as_int(
                sampling_cfg.get("warmup_epochs", 0),
                "data_config.dataloader.sampling.warmup_epochs",
            ),
            pool_multiplier=as_int(
                sampling_cfg.get("pool_multiplier", 32),
                "data_config.dataloader.sampling.pool_multiplier",
            ),
            cap_protein=as_int(
                sampling_cfg.get("cap_protein", 4),
                "data_config.dataloader.sampling.cap_protein",
            ),
            rank=rank if distributed else 0,
            world_size=world_size if distributed else 1,
            shuffle=shuffle,
            drop_last=drop_last,
            seed=seed,
        )
        should_shuffle = False

    if distributed and batch_sampler is None:
        sampler = DistributedSampler(
            dataset=dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=shuffle,
            seed=seed,
            drop_last=drop_last,
        )
        should_shuffle = False

    if batch_sampler is not None:
        return DataLoader(
            dataset=dataset,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=_collate_sequence_batch,
        )

    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=should_shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        sampler=sampler,
        collate_fn=_collate_sequence_batch,
    )


def build_dataloaders_v6(
    config: ConfigDict,
    distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
) -> dict[str, DataLoader[dict[str, object]]]:
    """Build V6 sequence-native train/valid/test dataloaders."""
    run_cfg = get_section(config, "run_config")
    split_path_map = _resolve_split_paths(config=config)
    split_paths = list(split_path_map.values())
    required_ids = _collect_required_protein_ids(split_paths=split_paths)
    search_roots = _resolve_sequence_search_roots(config=config, split_paths=split_paths)
    sequence_file, id_column, sequence_column = _sequence_lookup_overrides(config=config)
    sequences = _discover_sequences(
        required_ids=required_ids,
        search_roots=search_roots,
        explicit_sequence_file=sequence_file,
        id_column_override=id_column,
        sequence_column_override=sequence_column,
    )
    seed = as_int(run_cfg.get("seed", 0), "run_config.seed")

    return {
        "train": _build_split_loader_v6(
            split_path=split_path_map["train"],
            config=config,
            sequences=sequences,
            seed=seed,
            shuffle=True,
            distributed=distributed,
            rank=rank,
            world_size=world_size,
        ),
        "valid": _build_split_loader_v6(
            split_path=split_path_map["valid"],
            config=config,
            sequences=sequences,
            seed=seed + 1,
            shuffle=False,
            distributed=False,
            rank=rank,
            world_size=world_size,
        ),
        "test": _build_split_loader_v6(
            split_path=split_path_map["test"],
            config=config,
            sequences=sequences,
            seed=seed + 2,
            shuffle=False,
            distributed=False,
            rank=rank,
            world_size=world_size,
        ),
    }
