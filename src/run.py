"""Centralized, config-driven pipeline runner script entrypoint."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

from torch.nn.parallel import DistributedDataParallel

# Ensure absolute imports resolve when invoked as `python src/run.py`.
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.run.bootstrap import (
    _rank_from_env,
    configure_root_logging,
    parse_args,
    set_global_seed,
)
from src.run.pipeline_orchestrator import (
    _ddp_find_unused_parameters,
)
from src.run.pipeline_orchestrator import (
    execute_pipeline as _execute_pipeline_impl,
)
from src.run.stage_evaluate import EVAL_CSV_COLUMNS, _metrics_from_config, run_evaluation_stage
from src.run.stage_train import (
    _training_validation_metrics,
    build_model,
    build_strategy,
    build_trainer,
    run_training_stage,
)
from src.utils.config import ConfigDict, load_config
from src.utils.data_io import build_dataloaders
from src.utils.device import resolve_device
from src.utils.distributed import cleanup_distributed, initialize_distributed

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

ROOT_LOGGER = logging.getLogger(__name__)


def _configure_root_logging() -> None:
    """Configure process-level logging; suppress non-main rank noise."""
    configure_root_logging(logging_module=logging, rank=_rank_from_env())


def execute_pipeline(config: ConfigDict) -> None:
    """Execute pipeline according to configured run mode."""
    _execute_pipeline_impl(
        config=config,
        build_dataloaders_fn=build_dataloaders,
        build_model_fn=build_model,
        run_training_stage_fn=run_training_stage,
        run_evaluation_stage_fn=run_evaluation_stage,
        initialize_distributed_fn=initialize_distributed,
        cleanup_distributed_fn=cleanup_distributed,
        resolve_device_fn=resolve_device,
        distributed_data_parallel_cls=DistributedDataParallel,
    )


def main() -> None:
    """Run CLI entrypoint."""
    _configure_root_logging()
    args = parse_args()
    config = load_config(args.config)
    ROOT_LOGGER.info("Loaded config: %s", args.config)
    execute_pipeline(config=config)


__all__ = [
    "EVAL_CSV_COLUMNS",
    "_configure_root_logging",
    "_ddp_find_unused_parameters",
    "_metrics_from_config",
    "_training_validation_metrics",
    "build_dataloaders",
    "build_model",
    "build_strategy",
    "build_trainer",
    "cleanup_distributed",
    "execute_pipeline",
    "initialize_distributed",
    "main",
    "parse_args",
    "resolve_device",
    "run_evaluation_stage",
    "run_training_stage",
    "set_global_seed",
]


if __name__ == "__main__":
    main()
