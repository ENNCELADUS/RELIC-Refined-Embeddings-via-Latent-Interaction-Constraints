"""Centralized, config-driven pipeline runner."""

from __future__ import annotations

import argparse
import logging
import random
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Literal, cast

import numpy as np
import torch
from torch import nn

from src.evaluate import Evaluator
from src.model import V3
from src.train import NoOpStrategy, StagedUnfreezeStrategy, Trainer
from src.train.base import OptimizerConfig, SchedulerConfig
from src.utils.config import (
    ConfigDict,
    as_bool,
    as_float,
    as_int,
    as_str,
    as_str_list,
    extract_model_kwargs,
    get_section,
    load_config,
)
from src.utils.data_io import build_dataloaders
from src.utils.device import resolve_device
from src.utils.distributed import initialize_distributed
from src.utils.early_stop import EarlyStopping
from src.utils.logging import (
    append_csv_row,
    generate_run_id,
    prepare_stage_directories,
    setup_stage_logger,
)
from src.utils.ohem_sample_strategy import OHEMSampleStrategy

ROOT_LOGGER = logging.getLogger(__name__)
AnnealStrategy = Literal["cos", "linear"]
ModelFactory = Callable[[ConfigDict], nn.Module]


def _build_v3_model(model_kwargs: ConfigDict) -> nn.Module:
    """Build V3 model instance."""
    return V3(**model_kwargs)


MODEL_FACTORIES: dict[str, ModelFactory] = {
    "v3": _build_v3_model,
}


def _parse_anneal_strategy(value: object) -> AnnealStrategy:
    """Parse OneCycle anneal strategy."""
    anneal_strategy = as_str(value, "training_config.scheduler.anneal_strategy").lower()
    if anneal_strategy not in {"cos", "linear"}:
        raise ValueError("training_config.scheduler.anneal_strategy must be 'cos' or 'linear'")
    return cast(AnnealStrategy, anneal_strategy)


def _metrics_from_config(eval_cfg: ConfigDict) -> list[str]:
    """Extract configured metric names."""
    metrics = eval_cfg.get("metrics", [])
    if not isinstance(metrics, Sequence) or isinstance(metrics, (str, bytes)):
        raise ValueError("evaluate.metrics must be a sequence")
    return as_str_list(metrics, "evaluate.metrics")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed CLI namespace with the config path.
    """
    parser = argparse.ArgumentParser(description="Run RELIC training/evaluation pipeline.")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML.")
    return parser.parse_args()


def set_global_seed(seed: int) -> None:
    """Set random seeds for reproducibility.

    Args:
        seed: Global random seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(config: ConfigDict) -> nn.Module:
    """Build model from ``model_config``.

    Args:
        config: Global run configuration dictionary.

    Returns:
        Instantiated PyTorch model.

    Raises:
        ValueError: If the model name is not supported.
    """
    model_name, model_kwargs = extract_model_kwargs(config)
    factory = MODEL_FACTORIES.get(model_name)
    if factory is not None:
        return factory(model_kwargs)
    raise ValueError(f"Unknown model: {model_name}")


def build_trainer(
    config: ConfigDict,
    model: nn.Module,
    device: torch.device,
    steps_per_epoch: int,
) -> Trainer:
    """Instantiate trainer with optimizer/scheduler configs.

    Args:
        config: Global run configuration dictionary.
        model: Instantiated model.
        device: Target torch device.
        steps_per_epoch: Number of training steps per epoch.

    Returns:
        Configured trainer instance.
    """
    training_cfg = get_section(config, "training_config")
    data_cfg = get_section(config, "data_config")
    dataloader_cfg = get_section(data_cfg, "dataloader")
    optimizer_cfg = get_section(training_cfg, "optimizer")
    scheduler_cfg = get_section(training_cfg, "scheduler")
    sampling_cfg = get_section(dataloader_cfg, "sampling")

    optimizer_config = OptimizerConfig(
        optimizer_type=as_str(optimizer_cfg.get("type", "adamw"), "training_config.optimizer.type"),
        lr=as_float(optimizer_cfg.get("lr", 1e-4), "training_config.optimizer.lr"),
        beta1=as_float(optimizer_cfg.get("beta1", 0.9), "training_config.optimizer.beta1"),
        beta2=as_float(optimizer_cfg.get("beta2", 0.999), "training_config.optimizer.beta2"),
        eps=as_float(optimizer_cfg.get("eps", 1e-8), "training_config.optimizer.eps"),
        weight_decay=as_float(
            optimizer_cfg.get("weight_decay", 0.0),
            "training_config.optimizer.weight_decay",
        ),
    )
    scheduler_config = SchedulerConfig(
        scheduler_type=as_str(scheduler_cfg.get("type", "none"), "training_config.scheduler.type"),
        max_lr=as_float(
            scheduler_cfg.get("max_lr", optimizer_config.lr), "training_config.scheduler.max_lr"
        ),
        pct_start=as_float(
            scheduler_cfg.get("pct_start", 0.2),
            "training_config.scheduler.pct_start",
        ),
        div_factor=as_float(
            scheduler_cfg.get("div_factor", 25.0),
            "training_config.scheduler.div_factor",
        ),
        final_div_factor=as_float(
            scheduler_cfg.get("final_div_factor", 10000.0),
            "training_config.scheduler.final_div_factor",
        ),
        anneal_strategy=_parse_anneal_strategy(scheduler_cfg.get("anneal_strategy", "cos")),
    )

    sampling_strategy = as_str(
        sampling_cfg.get("strategy", "none"), "data_config.dataloader.sampling.strategy"
    ).lower()
    ohem_strategy = None
    if sampling_strategy == "ohem":
        ohem_strategy = OHEMSampleStrategy(
            keep_ratio=as_float(
                sampling_cfg.get("keep_ratio", 0.5),
                "data_config.dataloader.sampling.keep_ratio",
            ),
            min_keep=as_int(
                sampling_cfg.get("min_keep", 1), "data_config.dataloader.sampling.min_keep"
            ),
        )

    device_cfg = get_section(config, "device_config")
    total_epochs = as_int(training_cfg.get("epochs", 1), "training_config.epochs")
    return Trainer(
        model=model,
        device=device,
        optimizer_config=optimizer_config,
        scheduler_config=scheduler_config,
        use_amp=as_bool(
            device_cfg.get("use_mixed_precision", False), "device_config.use_mixed_precision"
        ),
        total_epochs=total_epochs,
        steps_per_epoch=steps_per_epoch,
        ohem_strategy=ohem_strategy,
    )


def build_strategy(config: ConfigDict) -> NoOpStrategy | StagedUnfreezeStrategy:
    """Build optional training strategy from config.

    Args:
        config: Global run configuration dictionary.

    Returns:
        Strategy implementation for training lifecycle hooks.
    """
    training_cfg = get_section(config, "training_config")
    strategy_cfg = training_cfg.get("strategy")
    if not isinstance(strategy_cfg, dict):
        return NoOpStrategy()
    strategy_type = str(strategy_cfg.get("type", "none")).lower()
    if strategy_type == "staged_unfreeze":
        prefixes_value = strategy_cfg.get("initial_trainable_prefixes", ["output_head"])
        if not isinstance(prefixes_value, list):
            raise ValueError("strategy.initial_trainable_prefixes must be a list")
        prefixes = tuple(str(prefix) for prefix in prefixes_value)
        return StagedUnfreezeStrategy(
            unfreeze_epoch=as_int(
                strategy_cfg.get("unfreeze_epoch", 1), "training_config.strategy.unfreeze_epoch"
            ),
            initial_trainable_prefixes=prefixes,
        )
    return NoOpStrategy()


def _save_checkpoint(model: nn.Module, checkpoint_path: Path) -> None:
    """Persist model weights to disk.

    Args:
        model: Model to serialize.
        checkpoint_path: Destination checkpoint path.
    """
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), checkpoint_path)


def _load_checkpoint(model: nn.Module, checkpoint_path: Path, device: torch.device) -> None:
    """Load model weights from disk.

    Args:
        model: Model receiving loaded weights.
        checkpoint_path: Source checkpoint path.
        device: Device map target.
    """
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)


def run_training_stage(
    stage: str,
    config: ConfigDict,
    model: nn.Module,
    device: torch.device,
    dataloaders: dict[str, torch.utils.data.DataLoader[dict[str, torch.Tensor]]],
    run_id: str,
    checkpoint_to_load: Path | None = None,
) -> Path:
    """Run stage training loop.

    Args:
        stage: Training stage name (`pretrain` or `finetune`).
        config: Global run configuration dictionary.
        model: Model to train.
        device: Target torch device.
        dataloaders: Split dataloaders keyed by `train`, `valid`, and `test`.
        run_id: Stage run identifier.
        checkpoint_to_load: Optional checkpoint to load before training.

    Returns:
        Path to the best checkpoint produced during the stage.
    """
    model_name, _ = extract_model_kwargs(config)
    log_dir, model_dir = prepare_stage_directories(
        model_name=model_name,
        stage=stage,
        run_id=run_id,
    )
    stage_logger = setup_stage_logger(
        name=f"relic.{model_name}.{stage}.{run_id}",
        log_file=log_dir / "log.log",
    )
    stage_logger.info("Starting %s stage.", stage)
    training_cfg = get_section(config, "training_config")
    eval_cfg = get_section(config, "evaluate")

    if checkpoint_to_load is not None:
        stage_logger.info("Loading checkpoint: %s", checkpoint_to_load)
        _load_checkpoint(model=model, checkpoint_path=checkpoint_to_load, device=device)

    trainer = build_trainer(
        config=config,
        model=model,
        device=device,
        steps_per_epoch=len(dataloaders["train"]),
    )
    strategy = build_strategy(config)
    evaluator = Evaluator(metrics=_metrics_from_config(eval_cfg))

    monitor_metric = as_str(
        training_cfg.get("monitor_metric", "auprc"), "training_config.monitor_metric"
    ).lower()
    monitor_key = f"val_{monitor_metric}"
    patience = as_int(
        training_cfg.get("early_stopping_patience", 5),
        "training_config.early_stopping_patience",
    )
    early_stopping = EarlyStopping(patience=patience, mode="max")
    epochs = as_int(training_cfg.get("epochs", 1), "training_config.epochs")
    save_best_only = as_bool(
        get_section(config, "run_config").get("save_best_only", True),
        "run_config.save_best_only",
    )

    best_checkpoint_path = model_dir / "best_model.pth"
    csv_path = log_dir / "training_step.csv"
    strategy.on_train_begin(trainer)

    for epoch in range(epochs):
        strategy.on_epoch_begin(trainer, epoch)
        train_stats = trainer.train_one_epoch(dataloaders["train"])
        model.eval()
        with torch.no_grad():
            val_stats = evaluator.evaluate(
                model=model,
                data_loader=dataloaders["valid"],
                device=device,
                prefix="val",
            )
        model.train()
        strategy.on_epoch_end(trainer, epoch)

        row: dict[str, float | int] = {
            "epoch": epoch + 1,
            "train_loss": train_stats["loss"],
            "learning_rate": train_stats["lr"],
        }
        for key, value in val_stats.items():
            row[key] = value
        append_csv_row(csv_path=csv_path, row=row)

        monitor_value = float(val_stats.get(monitor_key, 0.0))
        improved, should_stop = early_stopping.update(monitor_value)
        if improved:
            _save_checkpoint(model=model, checkpoint_path=best_checkpoint_path)
        if not save_best_only:
            _save_checkpoint(
                model=model,
                checkpoint_path=model_dir / f"checkpoint_epoch_{epoch + 1:03d}.pth",
            )

        stage_logger.info(
            "Epoch %d | train_loss=%.4f | %s=%.4f",
            epoch + 1,
            train_stats["loss"],
            monitor_key,
            monitor_value,
        )
        if should_stop:
            stage_logger.info("Early stopping triggered at epoch %d.", epoch + 1)
            break

    if not best_checkpoint_path.exists():
        _save_checkpoint(model=model, checkpoint_path=best_checkpoint_path)
    return best_checkpoint_path


def run_evaluation_stage(
    config: ConfigDict,
    model: nn.Module,
    device: torch.device,
    dataloaders: dict[str, torch.utils.data.DataLoader[dict[str, torch.Tensor]]],
    run_id: str,
    checkpoint_path: Path,
) -> dict[str, float]:
    """Run test evaluation and persist ``evaluate.csv``.

    Args:
        config: Global run configuration dictionary.
        model: Model to evaluate.
        device: Target torch device.
        dataloaders: Split dataloaders keyed by `train`, `valid`, and `test`.
        run_id: Evaluation run identifier.
        checkpoint_path: Checkpoint to evaluate.

    Returns:
        Dictionary of computed test metrics.
    """
    model_name, _ = extract_model_kwargs(config)
    log_dir, _ = prepare_stage_directories(model_name=model_name, stage="evaluate", run_id=run_id)
    logger = setup_stage_logger(
        name=f"relic.{model_name}.evaluate.{run_id}",
        log_file=log_dir / "log.log",
    )
    _load_checkpoint(model=model, checkpoint_path=checkpoint_path, device=device)
    model.eval()
    eval_cfg = get_section(config, "evaluate")
    evaluator = Evaluator(metrics=_metrics_from_config(eval_cfg))
    with torch.no_grad():
        metrics = evaluator.evaluate(
            model=model,
            data_loader=dataloaders["test"],
            device=device,
            prefix="test",
        )
    append_csv_row(csv_path=log_dir / "evaluate.csv", row=metrics)
    logger.info("Evaluation metrics: %s", metrics)
    return metrics


def execute_pipeline(config: ConfigDict) -> None:
    """Execute pipeline according to configured run mode.

    Args:
        config: Global run configuration dictionary.

    Raises:
        ValueError: If a required checkpoint path or run mode is invalid.
    """
    run_cfg = get_section(config, "run_config")
    device_cfg = get_section(config, "device_config")
    seed = as_int(run_cfg.get("seed", 0), "run_config.seed")
    set_global_seed(seed=seed)
    initialize_distributed(
        ddp_enabled=as_bool(device_cfg.get("ddp_enabled", False), "device_config.ddp_enabled")
    )
    device = resolve_device(as_str(device_cfg.get("device", "cpu"), "device_config.device"))

    dataloaders = build_dataloaders(config=config)
    model = build_model(config=config).to(device)
    mode = as_str(run_cfg.get("mode", "full_pipeline"), "run_config.mode").lower()

    pretrain_run_id = generate_run_id(run_cfg.get("pretrain_run_id"))
    finetune_run_id = generate_run_id(run_cfg.get("finetune_run_id"))
    eval_run_id = generate_run_id(run_cfg.get("eval_run_id"))
    load_checkpoint_value = run_cfg.get("load_checkpoint_path")
    load_checkpoint_path = (
        Path(str(load_checkpoint_value))
        if isinstance(load_checkpoint_value, str) and load_checkpoint_value
        else None
    )

    if mode in {"pretrain_only", "train_only"}:
        run_training_stage(
            stage="pretrain",
            config=config,
            model=model,
            device=device,
            dataloaders=dataloaders,
            run_id=pretrain_run_id,
        )
        return

    if mode == "finetune_from_pretrain":
        if load_checkpoint_path is None:
            raise ValueError("load_checkpoint_path is required for finetune_from_pretrain")
        run_training_stage(
            stage="finetune",
            config=config,
            model=model,
            device=device,
            dataloaders=dataloaders,
            run_id=finetune_run_id,
            checkpoint_to_load=load_checkpoint_path,
        )
        return

    if mode == "full_pipeline":
        pretrain_checkpoint = run_training_stage(
            stage="pretrain",
            config=config,
            model=model,
            device=device,
            dataloaders=dataloaders,
            run_id=pretrain_run_id,
        )
        finetune_checkpoint = run_training_stage(
            stage="finetune",
            config=config,
            model=model,
            device=device,
            dataloaders=dataloaders,
            run_id=finetune_run_id,
            checkpoint_to_load=pretrain_checkpoint,
        )
        run_evaluation_stage(
            config=config,
            model=model,
            device=device,
            dataloaders=dataloaders,
            run_id=eval_run_id,
            checkpoint_path=finetune_checkpoint,
        )
        return

    if mode == "eval_only":
        if load_checkpoint_path is None:
            raise ValueError("load_checkpoint_path is required for eval_only")
        run_evaluation_stage(
            config=config,
            model=model,
            device=device,
            dataloaders=dataloaders,
            run_id=eval_run_id,
            checkpoint_path=load_checkpoint_path,
        )
        return

    raise ValueError(f"Unsupported run mode: {mode}")


def main() -> None:
    """Run CLI entrypoint."""
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    config = load_config(args.config)
    ROOT_LOGGER.info("Loaded config: %s", args.config)
    execute_pipeline(config=config)


if __name__ == "__main__":
    main()
