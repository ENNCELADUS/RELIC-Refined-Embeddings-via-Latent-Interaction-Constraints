"""Evaluation-stage execution helpers."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import torch

from src.evaluate import Evaluator
from src.run.stage_train import _build_loss_config, _build_stage_runtime, _load_checkpoint
from src.utils.config import ConfigDict, as_str_list, extract_model_kwargs, get_section
from src.utils.distributed import DistributedContext, distributed_barrier
from src.utils.logging import append_csv_row, log_stage_event

EVAL_CSV_COLUMNS = [
    "split",
    "auroc",
    "auprc",
    "accuracy",
    "sensitivity",
    "specificity",
    "precision",
    "recall",
    "f1",
    "mcc",
]


def _metrics_from_config(eval_cfg: ConfigDict) -> list[str]:
    """Extract configured metric names."""
    metrics = eval_cfg.get("metrics", [])
    if not isinstance(metrics, Sequence) or isinstance(metrics, (str, bytes)):
        raise ValueError("evaluate.metrics must be a sequence")
    return as_str_list(metrics, "evaluate.metrics")


def run_evaluation_stage(
    config: ConfigDict,
    model: torch.nn.Module,
    device: torch.device,
    dataloaders: dict[str, torch.utils.data.DataLoader[dict[str, object]]],
    run_id: str,
    checkpoint_path: Path,
    distributed_context: DistributedContext,
) -> dict[str, float]:
    """Run test evaluation and persist ``evaluate.csv``."""
    checkpoint_path_resolved = Path(checkpoint_path)
    model_name, _ = extract_model_kwargs(config)
    log_dir, _, logger = _build_stage_runtime(
        model_name=model_name,
        stage="evaluate",
        run_id=run_id,
        distributed_context=distributed_context,
    )
    if distributed_context.is_main_process:
        log_stage_event(
            logger,
            "stage_start",
            run_id=run_id,
            checkpoint=checkpoint_path_resolved,
        )
    _load_checkpoint(model=model, checkpoint_path=checkpoint_path_resolved, device=device)
    if distributed_context.is_main_process:
        log_stage_event(logger, "checkpoint_loaded", path=checkpoint_path_resolved)
    model.eval()
    eval_cfg = get_section(config, "evaluate")
    training_cfg = get_section(config, "training_config")
    configured_metrics = _metrics_from_config(eval_cfg)
    metrics_to_compute = sorted(set(configured_metrics + EVAL_CSV_COLUMNS[1:]))
    evaluator = Evaluator(metrics=metrics_to_compute, loss_config=_build_loss_config(training_cfg))
    with torch.no_grad():
        metrics = evaluator.evaluate(
            model=model,
            data_loader=dataloaders["test"],
            device=device,
            prefix=None,
        )
    if distributed_context.is_main_process:
        csv_row: dict[str, float | int | str] = {"split": "test"}
        for metric_name in EVAL_CSV_COLUMNS[1:]:
            csv_row[metric_name] = float(metrics.get(metric_name, 0.0))
        append_csv_row(
            csv_path=log_dir / "evaluate.csv",
            row=csv_row,
            fieldnames=EVAL_CSV_COLUMNS,
        )
        log_stage_event(logger, "evaluation_metrics", **csv_row)
        log_stage_event(logger, "csv_written", path=log_dir / "evaluate.csv")
        log_stage_event(
            logger,
            "stage_done",
            run_id=run_id,
        )
    distributed_barrier(distributed_context)
    return metrics
