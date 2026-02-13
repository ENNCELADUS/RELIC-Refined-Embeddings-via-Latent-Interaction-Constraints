"""Embedding cache management and ESM3 embedding generation orchestrator."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

import torch
import torch.distributed as dist

from src.embed.cache import (
    _build_invalid_ids_error_message,
    _build_missing_ids_error_message,
    _embedding_relative_path,
    _expected_metadata,
    _find_missing_or_invalid_ids,
    _load_index,
    _load_metadata,
    _metadata_matches,
    _parse_str_dict,
    _parse_str_list,
    _shard_ids_for_rank,
    _write_json_atomic,
    load_cached_embedding,
)
from src.embed.config import (
    EMBEDDINGS_SUBDIR,
    INDEX_FILENAME,
    METADATA_FILENAME,
    EmbeddingCacheManifest,
    _parse_embedding_settings,
)
from src.embed.esm_client import _generate_missing_embeddings, _log_embedding_progress
from src.embed.io import (
    _collect_required_protein_ids,
    _discover_sequences,
    _resolve_sequence_search_roots,
)
from src.utils.config import ConfigDict

LOGGER = logging.getLogger(__name__)


def _distributed_generation_context(allow_generation: bool) -> tuple[int, int] | None:
    """Return ``(rank, world_size)`` when distributed generation should be used."""
    if not allow_generation:
        return None
    if not dist.is_available() or not dist.is_initialized():
        return None
    world_size = dist.get_world_size()
    if world_size <= 1:
        return None
    return dist.get_rank(), world_size


def ensure_embeddings_ready(
    config: ConfigDict,
    split_paths: Sequence[Path],
    input_dim: int,
    max_sequence_length: int,
    allow_generation: bool = True,
) -> EmbeddingCacheManifest:
    """Ensure all required embeddings exist and are valid before training/eval.

    Args:
        config: Root runtime configuration dictionary.
        split_paths: Train/valid/test split file paths.
        input_dim: Expected embedding dimension.
        max_sequence_length: Maximum sequence length.
        allow_generation: Whether missing/invalid embeddings may be regenerated.

    Returns:
        Embedding cache manifest containing index and required IDs.

    Raises:
        FileNotFoundError: If required embeddings are unavailable.
        ValueError: If source/config or embedding tensors are invalid.
    """
    if input_dim <= 0:
        raise ValueError("input_dim must be positive")
    if max_sequence_length <= 0:
        raise ValueError("max_sequence_length must be positive")

    settings = _parse_embedding_settings(config)
    required_ids = _collect_required_protein_ids(
        split_paths=split_paths,
    )
    distributed_context = _distributed_generation_context(allow_generation=allow_generation)
    distributed_rank: int | None = None
    distributed_world_size: int | None = None
    if distributed_context is not None:
        distributed_rank, distributed_world_size = distributed_context

    cache_dir = settings.cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / EMBEDDINGS_SUBDIR).mkdir(parents=True, exist_ok=True)
    index_path = cache_dir / INDEX_FILENAME
    metadata_path = cache_dir / METADATA_FILENAME

    index = _load_index(index_path=index_path)
    metadata = _load_metadata(metadata_path=metadata_path)
    expected_metadata = _expected_metadata(
        source=settings.source,
        model_name=settings.model_name,
        input_dim=input_dim,
        max_sequence_length=max_sequence_length,
    )
    metadata_matches = _metadata_matches(current=metadata, expected=expected_metadata)

    if metadata and not metadata_matches:
        LOGGER.warning(
            "Embedding cache metadata mismatch at %s; required IDs will be re-embedded",
            metadata_path,
        )

    if metadata_matches:
        missing_ids, invalid_ids = _find_missing_or_invalid_ids(
            required_ids=required_ids,
            cache_dir=cache_dir,
            index=index,
            input_dim=input_dim,
            max_sequence_length=max_sequence_length,
        )
        ids_to_generate = set(missing_ids) | set(invalid_ids)
    else:
        missing_ids = set(required_ids)
        invalid_ids = {}
        ids_to_generate = set(required_ids)

    if distributed_context is not None:
        if distributed_rank is None or distributed_world_size is None:
            raise RuntimeError("Distributed generation context must include rank and world size")

        index_payload_list: list[object] = [dict(index) if distributed_rank == 0 else {}]
        ids_payload_list: list[object] = [sorted(ids_to_generate) if distributed_rank == 0 else []]
        dist.broadcast_object_list(index_payload_list, src=0)
        dist.broadcast_object_list(ids_payload_list, src=0)

        index = _parse_str_dict(index_payload_list[0], "broadcast embedding index")
        ids_to_generate = set(_parse_str_list(ids_payload_list[0], "broadcast ids_to_generate"))

    if ids_to_generate:
        if not allow_generation:
            if invalid_ids:
                raise ValueError(_build_invalid_ids_error_message(invalid_ids))
            raise FileNotFoundError(_build_missing_ids_error_message(ids_to_generate))

        search_roots = _resolve_sequence_search_roots(config=config, split_paths=split_paths)
        if distributed_context is None:
            discovered_sequences = _discover_sequences(
                required_ids=ids_to_generate,
                search_roots=search_roots,
                explicit_sequence_file=settings.sequence_file,
                id_column_override=settings.id_column,
                sequence_column_override=settings.sequence_column,
            )
            generated_updates = _generate_missing_embeddings(
                missing_ids=ids_to_generate,
                sequences=discovered_sequences,
                settings=settings,
                cache_dir=cache_dir,
                input_dim=input_dim,
                max_sequence_length=max_sequence_length,
            )
            index.update(generated_updates)
            _write_json_atomic(path=index_path, payload=index)
            _write_json_atomic(path=metadata_path, payload=expected_metadata)
        else:
            if distributed_rank is None or distributed_world_size is None:
                raise RuntimeError(
                    "Distributed generation context must include rank and world size"
                )
            sorted_ids_to_generate = sorted(ids_to_generate)
            local_ids = _shard_ids_for_rank(
                sorted_ids=sorted_ids_to_generate,
                rank=distributed_rank,
                world_size=distributed_world_size,
            )
            _log_embedding_progress("Assigned %d proteins to this rank", len(local_ids))
            local_updates: dict[str, str] = {}
            local_error: str | None = None
            try:
                if local_ids:
                    discovered_sequences = _discover_sequences(
                        required_ids=local_ids,
                        search_roots=search_roots,
                        explicit_sequence_file=settings.sequence_file,
                        id_column_override=settings.id_column,
                        sequence_column_override=settings.sequence_column,
                    )
                    local_updates = _generate_missing_embeddings(
                        missing_ids=local_ids,
                        sequences=discovered_sequences,
                        settings=settings,
                        cache_dir=cache_dir,
                        input_dim=input_dim,
                        max_sequence_length=max_sequence_length,
                    )
            except (FileNotFoundError, RuntimeError, TypeError, ValueError) as error:
                local_error = f"rank {distributed_rank}: {error}"

            error_reports: list[object] = [None for _ in range(distributed_world_size)]
            dist.all_gather_object(error_reports, local_error)
            errors = [report for report in error_reports if isinstance(report, str) and report]
            if errors:
                raise RuntimeError("Distributed embedding generation failed: " + " | ".join(errors))

            if distributed_rank == 0:
                gathered_updates: list[object] = [None for _ in range(distributed_world_size)]
                dist.gather_object(local_updates, gathered_updates, dst=0)
                for rank_updates_payload in gathered_updates:
                    index.update(_parse_str_dict(rank_updates_payload, "gathered index updates"))
                _write_json_atomic(path=index_path, payload=index)
                _write_json_atomic(path=metadata_path, payload=expected_metadata)
            else:
                dist.gather_object(local_updates, None, dst=0)

            dist.barrier()
            index = _load_index(index_path=index_path)

    final_missing_ids, final_invalid_ids = _find_missing_or_invalid_ids(
        required_ids=required_ids,
        cache_dir=cache_dir,
        index=index,
        input_dim=input_dim,
        max_sequence_length=max_sequence_length,
    )
    if final_missing_ids:
        raise FileNotFoundError(_build_missing_ids_error_message(final_missing_ids))
    if final_invalid_ids:
        raise ValueError(_build_invalid_ids_error_message(final_invalid_ids))

    return EmbeddingCacheManifest(
        cache_dir=cache_dir,
        index=index,
        required_ids=frozenset(required_ids),
    )
