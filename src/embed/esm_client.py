"""ESM runtime loading and embedding generation helpers."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path
from typing import cast

import torch
import torch.distributed as dist

from src.embed.cache import _embedding_relative_path, _save_tensor_atomic
from src.embed.config import (
    EMBED_PROGRESS_LOGGER_NAME,
    _EmbeddingSettings,
    _Esm3ClassProtocol,
    _Esm3ModelProtocol,
    _LogitsConfigCtorProtocol,
    _LogitsOutputProtocol,
    _ProteinCtorProtocol,
)

LOGGER = logging.getLogger(__name__)


def _get_embed_progress_logger() -> logging.Logger:
    """Return dedicated logger for distributed embedding progress."""
    logger = logging.getLogger(EMBED_PROGRESS_LOGGER_NAME)
    if logger.handlers:
        return logger
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger


def _log_embedding_progress(message: str, *args: object) -> None:
    """Log embedding progress from all ranks."""
    if dist.is_available() and dist.is_initialized():
        progress_logger = _get_embed_progress_logger()
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        progress_logger.info("[embed][rank=%d/%d] " + message, rank, world_size, *args)
        return
    LOGGER.info(message, *args)


def _resolve_embedding_device(requested_device: str) -> str:
    """Resolve embedding runtime device from config setting."""
    if requested_device != "auto":
        return requested_device
    if torch.cuda.is_available():
        return "cuda"
    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        return "mps"
    return "cpu"


def _load_esm3_runtime(
    model_name: str,
    requested_device: str,
) -> tuple[_Esm3ModelProtocol, _ProteinCtorProtocol, _LogitsConfigCtorProtocol, str]:
    """Load ESM3 runtime objects lazily."""
    try:
        from esm.models.esm3 import ESM3
        from esm.sdk.api import ESMProtein, LogitsConfig
    except ImportError as error:
        raise RuntimeError("ESM3 runtime is unavailable. Install the 'esm' dependency.") from error

    device = _resolve_embedding_device(requested_device)
    esm3_class = cast(_Esm3ClassProtocol, ESM3)
    protein_ctor = cast(_ProteinCtorProtocol, ESMProtein)
    logits_ctor = cast(_LogitsConfigCtorProtocol, LogitsConfig)
    model = esm3_class.from_pretrained(model_name).to(device)
    model.eval()
    return model, protein_ctor, logits_ctor, device


def _embed_sequence_with_esm3(
    model: _Esm3ModelProtocol,
    protein_ctor: _ProteinCtorProtocol,
    logits_ctor: _LogitsConfigCtorProtocol,
    sequence: str,
) -> torch.Tensor:
    """Embed one protein sequence using ESM3."""
    protein = protein_ctor(sequence=sequence)
    logits_config = logits_ctor(sequence=True, return_embeddings=True)
    with torch.no_grad():
        protein_tensor = model.encode(protein)
        sequence_output = model.logits(protein_tensor, logits_config)
    sequence_output_typed = cast(_LogitsOutputProtocol, sequence_output)
    embeddings = sequence_output_typed.embeddings
    if embeddings is None:
        raise ValueError("ESM3 did not return embeddings")

    embedding_tensor = embeddings.detach().to(dtype=torch.float32).cpu()
    if embedding_tensor.dim() == 3 and embedding_tensor.size(0) == 1:
        embedding_tensor = embedding_tensor.squeeze(0)
    if embedding_tensor.dim() != 2:
        raise ValueError(
            f"ESM3 returned unexpected embedding shape: {tuple(embedding_tensor.shape)}"
        )
    if embedding_tensor.size(0) <= 0:
        raise ValueError("ESM3 returned an empty embedding sequence")
    return embedding_tensor.contiguous()


def _generate_missing_embeddings(
    missing_ids: set[str],
    sequences: Mapping[str, str],
    settings: _EmbeddingSettings,
    cache_dir: Path,
    input_dim: int,
    max_sequence_length: int,
) -> dict[str, str]:
    """Generate and persist missing embeddings for required proteins."""
    if settings.source != "esm3":
        raise ValueError(f"Unsupported embedding source: {settings.source}")

    model, protein_ctor, logits_ctor, resolved_device = _load_esm3_runtime(
        model_name=settings.model_name,
        requested_device=settings.device,
    )
    _log_embedding_progress(
        "Generating embeddings for %d proteins using model=%s device=%s",
        len(missing_ids),
        settings.model_name,
        resolved_device,
    )

    generated_index_updates: dict[str, str] = {}
    total = len(missing_ids)
    for idx, protein_id in enumerate(sorted(missing_ids), start=1):
        sequence = sequences.get(protein_id)
        if sequence is None:
            raise FileNotFoundError(f"Missing sequence for protein '{protein_id}'")

        truncated_sequence = sequence[:max_sequence_length]
        if not truncated_sequence:
            raise ValueError(f"Protein '{protein_id}' has an empty sequence after cleaning")

        embedding_tensor = _embed_sequence_with_esm3(
            model=model,
            protein_ctor=protein_ctor,
            logits_ctor=logits_ctor,
            sequence=truncated_sequence,
        )
        if embedding_tensor.size(0) > max_sequence_length:
            embedding_tensor = embedding_tensor[:max_sequence_length]
        if embedding_tensor.size(1) != input_dim:
            raise ValueError(
                f"Embedding dim mismatch for protein '{protein_id}': "
                f"expected {input_dim}, got {embedding_tensor.size(1)}"
            )

        relative_path = _embedding_relative_path(protein_id)
        output_path = cache_dir / relative_path
        _save_tensor_atomic(path=output_path, tensor=embedding_tensor)
        generated_index_updates[protein_id] = relative_path

        if idx == total or idx % 100 == 0:
            _log_embedding_progress("Embedded %d/%d proteins", idx, total)

    return generated_index_updates
