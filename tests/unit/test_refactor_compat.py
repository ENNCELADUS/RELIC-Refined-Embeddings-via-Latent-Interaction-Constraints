"""Characterization tests for public import compatibility during refactors."""

from __future__ import annotations

import src.run as run_module
from src.embed import EmbeddingCacheManifest, ensure_embeddings_ready, load_cached_embedding
from src.train.config import LossConfig as SharedLossConfig
from src.train.strategies.ohem import OHEMSampleStrategy as SharedOHEMSampleStrategy
from src.utils.losses import LossConfig, binary_classification_loss
from src.utils.ohem_sample_strategy import OHEMSampleStrategy, select_ohem_indices


def test_run_module_entrypoint_contract() -> None:
    """Lock top-level run-module callable exports used by tests and scripts."""
    assert callable(run_module.main)
    assert callable(run_module.execute_pipeline)
    assert callable(run_module.run_training_stage)
    assert callable(run_module.run_evaluation_stage)
    assert isinstance(run_module.EVAL_CSV_COLUMNS, list)


def test_embed_public_exports_contract() -> None:
    """Lock embed module exports expected by data loading and tests."""
    assert EmbeddingCacheManifest.__name__ == "EmbeddingCacheManifest"
    assert callable(ensure_embeddings_ready)
    assert callable(load_cached_embedding)


def test_legacy_losses_and_ohem_imports_contract() -> None:
    """Lock legacy utility import paths expected across code and tests."""
    assert LossConfig.__name__ == "LossConfig"
    assert callable(binary_classification_loss)
    assert OHEMSampleStrategy.__name__ == "OHEMSampleStrategy"
    assert callable(select_ohem_indices)


def test_losses_legacy_import_points_to_shared_train_config() -> None:
    """Lock compatibility alias between legacy losses path and shared config."""
    assert LossConfig is SharedLossConfig


def test_ohem_legacy_import_points_to_shared_train_strategy() -> None:
    """Lock compatibility alias between legacy utils path and train strategy module."""
    assert OHEMSampleStrategy is SharedOHEMSampleStrategy
