"""Data input utilities for PRING-backed protein pair training."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from src.utils.config import ConfigDict, as_bool, as_int, get_section


@dataclass(frozen=True)
class PPIPairRecord:
    """Single protein-protein interaction record.

    Attributes:
        protein_a: First protein identifier.
        protein_b: Second protein identifier.
        label: Binary interaction label.
    """

    protein_a: str
    protein_b: str
    label: int


def _stable_int_from_text(text: str, seed: int) -> int:
    """Map a text value and seed to a deterministic integer.

    Args:
        text: Input text token.
        seed: Integer seed namespace.

    Returns:
        Deterministic integer hash.
    """
    digest = hashlib.sha256(f"{seed}:{text}".encode()).hexdigest()
    return int(digest[:8], 16)


def _read_ppi_records(file_path: Path, max_samples: int | None) -> list[PPIPairRecord]:
    """Read tab-separated PRING interaction records.

    Args:
        file_path: Input interaction file path.
        max_samples: Optional max number of records to read.

    Returns:
        Parsed interaction records.

    Raises:
        ValueError: If no valid records are parsed.
    """
    records: list[PPIPairRecord] = []
    with file_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split("\t")
            if len(parts) != 3:
                continue
            label = int(parts[2])
            records.append(PPIPairRecord(parts[0], parts[1], label))
            if max_samples is not None and len(records) >= max_samples:
                break
    if not records:
        raise ValueError(f"No valid PPI records found in {file_path}")
    return records


class PRINGPairDataset(Dataset[dict[str, torch.Tensor]]):
    """Synthetic embedding dataset generated from PRING pair identities.

    TODO: Replace synthetic embedding generation with on-disk embedding
    lookup from ``data_config.embeddings.cache_dir``.

    Args:
        file_path: PPI split file path.
        input_dim: Embedding feature dimension.
        max_sequence_length: Sequence length used for synthetic tensors.
        seed: Base seed for deterministic synthetic embeddings.
        max_samples: Optional maximum samples to load.
    """

    def __init__(
        self,
        file_path: Path,
        input_dim: int,
        max_sequence_length: int,
        seed: int,
        max_samples: int | None = None,
    ) -> None:
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if max_sequence_length <= 0:
            raise ValueError("max_sequence_length must be positive")
        self._records = _read_ppi_records(file_path=file_path, max_samples=max_samples)
        self._input_dim = int(input_dim)
        self._seq_len = int(max_sequence_length)
        self._seed = int(seed)

    def __len__(self) -> int:
        """Return dataset length."""
        return len(self._records)

    def _build_embedding(self, protein_id: str) -> torch.Tensor:
        """Create deterministic synthetic embedding for one protein ID.

        TODO: Load precomputed embeddings from file cache instead of
        generating synthetic random tensors.

        Args:
            protein_id: Protein identifier.

        Returns:
            Embedding tensor of shape ``(seq_len, input_dim)``.
        """
        local_seed = _stable_int_from_text(protein_id, self._seed)
        generator = torch.Generator().manual_seed(local_seed)
        return torch.randn(self._seq_len, self._input_dim, generator=generator)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """Return one dataset example.

        Args:
            index: Sample index.

        Returns:
            Dictionary containing paired embeddings, lengths, and label.
        """
        record = self._records[index]
        emb_a = self._build_embedding(record.protein_a)
        emb_b = self._build_embedding(record.protein_b)
        seq_len = torch.tensor(self._seq_len, dtype=torch.long)
        label = torch.tensor(float(record.label), dtype=torch.float32)
        return {
            "emb_a": emb_a,
            "emb_b": emb_b,
            "len_a": seq_len,
            "len_b": seq_len,
            "label": label,
        }


def _collate_batch(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Collate dataset samples into a dense batch.

    Args:
        batch: List of sample dictionaries.

    Returns:
        Batched tensor dictionary.
    """
    return {
        "emb_a": torch.stack([sample["emb_a"] for sample in batch], dim=0),
        "emb_b": torch.stack([sample["emb_b"] for sample in batch], dim=0),
        "len_a": torch.stack([sample["len_a"] for sample in batch], dim=0),
        "len_b": torch.stack([sample["len_b"] for sample in batch], dim=0),
        "label": torch.stack([sample["label"] for sample in batch], dim=0),
    }


def _build_split_loader(
    split_path: Path,
    config: ConfigDict,
    input_dim: int,
    max_sequence_length: int,
    seed: int,
    shuffle: bool,
    distributed: bool,
    rank: int,
    world_size: int,
) -> DataLoader[dict[str, torch.Tensor]]:
    """Build a split-specific data loader.

    Args:
        split_path: Dataset file path.
        config: Root configuration mapping.
        input_dim: Embedding feature dimension.
        max_sequence_length: Sequence length used for synthetic tensors.
        seed: Dataset seed.
        shuffle: Whether to shuffle samples.
        distributed: Whether DDP sampling is enabled.
        rank: Global rank for distributed sampler.
        world_size: World size for distributed sampler.

    Returns:
        Configured dataloader for the requested split.
    """
    training_cfg = get_section(config, "training_config")
    data_cfg = get_section(config, "data_config")
    dataloader_cfg = get_section(data_cfg, "dataloader")
    batch_size = as_int(training_cfg.get("batch_size", 8), "training_config.batch_size")
    num_workers = as_int(dataloader_cfg.get("num_workers", 0), "data_config.dataloader.num_workers")
    pin_memory = as_bool(
        dataloader_cfg.get("pin_memory", False), "data_config.dataloader.pin_memory"
    )
    drop_last = as_bool(dataloader_cfg.get("drop_last", False), "data_config.dataloader.drop_last")
    max_samples_value = dataloader_cfg.get("max_samples_per_split")
    max_samples = (
        as_int(max_samples_value, "data_config.dataloader.max_samples_per_split")
        if max_samples_value is not None
        else None
    )
    dataset = PRINGPairDataset(
        file_path=split_path,
        input_dim=input_dim,
        max_sequence_length=max_sequence_length,
        seed=seed,
        max_samples=max_samples,
    )
    sampler: DistributedSampler[dict[str, torch.Tensor]] | None = None
    should_shuffle = shuffle
    if distributed:
        sampler = DistributedSampler(
            dataset=dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=shuffle,
            seed=seed,
            drop_last=drop_last,
        )
        should_shuffle = False

    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=should_shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        sampler=sampler,
        collate_fn=_collate_batch,
    )


def build_dataloaders(
    config: ConfigDict,
    distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
) -> dict[str, DataLoader[dict[str, torch.Tensor]]]:
    """Build train/valid/test loaders from the global config.

    Args:
        config: Root configuration mapping.
        distributed: Whether distributed data loading is enabled.
        rank: Global rank for distributed loading.
        world_size: Number of distributed processes.

    Returns:
        Split dataloader mapping with keys ``train``, ``valid``, and ``test``.

    Raises:
        FileNotFoundError: If benchmark root or split files are missing.
    """
    run_cfg = get_section(config, "run_config")
    data_cfg = get_section(config, "data_config")
    model_cfg = get_section(config, "model_config")
    benchmark_cfg = get_section(data_cfg, "benchmark")
    dataloader_cfg = get_section(data_cfg, "dataloader")
    benchmark_root = Path(str(benchmark_cfg.get("root_dir", "")))
    if not benchmark_root.exists():
        raise FileNotFoundError(f"Benchmark root does not exist: {benchmark_root}")

    input_dim = as_int(model_cfg.get("input_dim", 0), "model_config.input_dim")
    max_sequence_length = as_int(
        data_cfg.get("max_sequence_length", 64), "data_config.max_sequence_length"
    )
    seed = as_int(run_cfg.get("seed", 0), "run_config.seed")

    train_path = Path(str(dataloader_cfg.get("train_dataset", "")))
    valid_path = Path(str(dataloader_cfg.get("valid_dataset", "")))
    test_path = Path(str(dataloader_cfg.get("test_dataset", "")))
    for split_path in (train_path, valid_path, test_path):
        if not split_path.exists():
            raise FileNotFoundError(f"Dataset path not found: {split_path}")

    return {
        "train": _build_split_loader(
            split_path=train_path,
            config=config,
            input_dim=input_dim,
            max_sequence_length=max_sequence_length,
            seed=seed,
            shuffle=True,
            distributed=distributed,
            rank=rank,
            world_size=world_size,
        ),
        "valid": _build_split_loader(
            split_path=valid_path,
            config=config,
            input_dim=input_dim,
            max_sequence_length=max_sequence_length,
            seed=seed + 1,
            shuffle=False,
            distributed=False,
            rank=rank,
            world_size=world_size,
        ),
        "test": _build_split_loader(
            split_path=test_path,
            config=config,
            input_dim=input_dim,
            max_sequence_length=max_sequence_length,
            seed=seed + 2,
            shuffle=False,
            distributed=False,
            rank=rank,
            world_size=world_size,
        ),
    }
