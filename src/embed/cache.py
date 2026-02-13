"""Embedding cache read/write and validation helpers."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import cast

import torch

from src.embed.config import CACHE_SCHEMA_VERSION, EMBEDDINGS_SUBDIR


def _load_json_mapping(path: Path, description: str) -> dict[str, object]:
    """Load a JSON object mapping from disk."""
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{description} at {path} must be a JSON object")
    return cast(dict[str, object], payload)


def _load_index(index_path: Path) -> dict[str, str]:
    """Load and validate embedding index file."""
    if not index_path.exists():
        return {}
    raw_index = _load_json_mapping(index_path, "Embedding index")
    index: dict[str, str] = {}
    for protein_id, rel_path in raw_index.items():
        if not isinstance(protein_id, str) or not protein_id:
            raise ValueError(f"Invalid protein ID key in index: {protein_id!r}")
        if not isinstance(rel_path, str) or not rel_path:
            raise ValueError(f"Invalid embedding path for protein '{protein_id}' in index")
        index[protein_id] = rel_path
    return index


def _load_metadata(metadata_path: Path) -> dict[str, object]:
    """Load metadata mapping when present."""
    if not metadata_path.exists():
        return {}
    return _load_json_mapping(metadata_path, "Embedding metadata")


def _write_json_atomic(path: Path, payload: Mapping[str, object]) -> None:
    """Atomically write JSON payload to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f"{path.name}.tmp")
    with temp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    temp_path.replace(path)


def _embedding_relative_path(protein_id: str) -> str:
    """Return deterministic relative cache path for one protein ID."""
    digest = hashlib.sha256(protein_id.encode("utf-8")).hexdigest()
    return f"{EMBEDDINGS_SUBDIR}/{digest}.pt"


def _resolve_embedding_path(cache_dir: Path, relative_path: str) -> Path:
    """Resolve and validate embedding path under cache root."""
    rel_path = Path(relative_path)
    if rel_path.is_absolute():
        raise ValueError(
            f"Embedding index path must be relative, got absolute path: {relative_path}"
        )

    cache_root = cache_dir.resolve()
    absolute_path = (cache_dir / rel_path).resolve()
    if absolute_path != cache_root and cache_root not in absolute_path.parents:
        raise ValueError(f"Embedding index path escapes cache directory: {relative_path}")
    return absolute_path


def _save_tensor_atomic(path: Path, tensor: torch.Tensor) -> None:
    """Atomically save an embedding tensor to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f"{path.name}.tmp")
    torch.save(tensor, temp_path)
    temp_path.replace(path)


def load_cached_embedding(
    cache_dir: Path,
    index: Mapping[str, str],
    protein_id: str,
    expected_input_dim: int | None = None,
    max_sequence_length: int | None = None,
) -> torch.Tensor:
    """Load one embedding tensor from cache with schema validation."""
    relative_path = index.get(protein_id)
    if relative_path is None:
        raise FileNotFoundError(f"Protein '{protein_id}' is missing from embedding index")

    embedding_path = _resolve_embedding_path(cache_dir=cache_dir, relative_path=relative_path)
    if not embedding_path.exists():
        raise FileNotFoundError(
            f"Embedding file missing for protein '{protein_id}': {embedding_path}"
        )

    tensor_object: object = torch.load(embedding_path, map_location="cpu")
    if not isinstance(tensor_object, torch.Tensor):
        raise ValueError(f"Embedding file for protein '{protein_id}' does not contain a tensor")

    tensor = tensor_object.detach().to(dtype=torch.float32)
    if tensor.dim() == 3 and tensor.size(0) == 1:
        tensor = tensor.squeeze(0)
    if tensor.dim() != 2:
        raise ValueError(
            f"Embedding for protein '{protein_id}' must be 2D, got shape {tuple(tensor.shape)}"
        )
    if tensor.size(0) <= 0:
        raise ValueError(f"Embedding for protein '{protein_id}' has empty sequence length")
    if expected_input_dim is not None and tensor.size(1) != expected_input_dim:
        raise ValueError(
            f"Embedding dim mismatch for protein '{protein_id}': "
            f"expected {expected_input_dim}, got {tensor.size(1)}"
        )
    if max_sequence_length is not None and tensor.size(0) > max_sequence_length:
        raise ValueError(
            f"Embedding length exceeds max_sequence_length for protein '{protein_id}': "
            f"{tensor.size(0)} > {max_sequence_length}"
        )

    return tensor.contiguous()


def _find_missing_or_invalid_ids(
    required_ids: set[str],
    cache_dir: Path,
    index: Mapping[str, str],
    input_dim: int,
    max_sequence_length: int,
) -> tuple[set[str], dict[str, str]]:
    """Return missing and invalid IDs after cache validation."""
    missing_ids: set[str] = set()
    invalid_ids: dict[str, str] = {}

    for protein_id in sorted(required_ids):
        if protein_id not in index:
            missing_ids.add(protein_id)
            continue
        try:
            load_cached_embedding(
                cache_dir=cache_dir,
                index=index,
                protein_id=protein_id,
                expected_input_dim=input_dim,
                max_sequence_length=max_sequence_length,
            )
        except FileNotFoundError:
            missing_ids.add(protein_id)
        except (ValueError, RuntimeError, TypeError) as error:
            invalid_ids[protein_id] = str(error)

    return missing_ids, invalid_ids


def _parse_str_dict(payload: object, name: str) -> dict[str, str]:
    """Parse an object payload as a string-to-string mapping."""
    if not isinstance(payload, dict):
        raise ValueError(f"{name} must be a dictionary")
    parsed: dict[str, str] = {}
    for key, value in payload.items():
        if not isinstance(key, str) or not isinstance(value, str):
            raise ValueError(f"{name} must contain string keys and values")
        parsed[key] = value
    return parsed


def _parse_str_list(payload: object, name: str) -> list[str]:
    """Parse an object payload as a list of strings."""
    if not isinstance(payload, list):
        raise ValueError(f"{name} must be a list")
    parsed: list[str] = []
    for item in payload:
        if not isinstance(item, str):
            raise ValueError(f"{name} must contain strings")
        parsed.append(item)
    return parsed


def _shard_ids_for_rank(
    sorted_ids: Sequence[str],
    rank: int,
    world_size: int,
) -> set[str]:
    """Return rank-local shard of sorted protein IDs."""
    shard: set[str] = set()
    for idx, protein_id in enumerate(sorted_ids):
        if idx % world_size == rank:
            shard.add(protein_id)
    return shard


def _expected_metadata(
    source: str,
    model_name: str,
    input_dim: int,
    max_sequence_length: int,
) -> dict[str, object]:
    """Build expected metadata payload for this run configuration."""
    return {
        "schema_version": CACHE_SCHEMA_VERSION,
        "source": source,
        "model_name": model_name,
        "input_dim": input_dim,
        "max_sequence_length": max_sequence_length,
        "format": "torch_pt_per_protein",
    }


def _metadata_matches(current: Mapping[str, object], expected: Mapping[str, object]) -> bool:
    """Return whether current metadata matches expected runtime settings."""
    if not current:
        return False
    return all(current.get(key) == value for key, value in expected.items())


def _build_missing_ids_error_message(missing_ids: set[str]) -> str:
    """Build concise missing-ID message."""
    preview = ", ".join(sorted(missing_ids)[:10])
    return f"Missing embeddings for {len(missing_ids)} proteins: {preview}"


def _build_invalid_ids_error_message(invalid_ids: Mapping[str, str]) -> str:
    """Build concise invalid-ID message."""
    sampled_items = sorted(invalid_ids.items())[:5]
    details = "; ".join(f"{protein_id}: {reason}" for protein_id, reason in sampled_items)
    return f"Invalid embeddings detected for {len(invalid_ids)} proteins ({details})"
