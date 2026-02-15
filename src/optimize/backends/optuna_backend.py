"""Optuna backend adapter for RELIC optimization workflow."""

from __future__ import annotations

import importlib
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Protocol, cast

from src.optimize.search_space import SearchParameter, sample_parameter
from src.optimize.trial_runner import Direction, RunPipelineFn, TrialExecutionResult, execute_trial
from src.utils.config import ConfigDict, as_bool, as_float, as_int, as_str

LOGGER = logging.getLogger(__name__)


class _TrialProtocol(Protocol):
    """Protocol for Optuna-like trial object."""

    number: int

    def set_user_attr(self, key: str, value: object) -> None:
        """Set user attr."""

    def report(self, value: float, step: int) -> None:
        """Report intermediate value."""

    def should_prune(self) -> bool:
        """Return whether the trial should be pruned."""


class _FrozenTrialProtocol(Protocol):
    """Protocol for frozen trials in study output."""

    number: int
    state: object
    value: float | None
    params: Mapping[str, object]
    user_attrs: Mapping[str, object]


@dataclass(frozen=True)
class TrialRecord:
    """Serializable trial summary."""

    number: int
    state: str
    value: float | None
    run_id: str | None
    params: dict[str, object]


@dataclass(frozen=True)
class OptunaResult:
    """Final optimization result from Optuna backend."""

    study_name: str
    direction: Direction
    objective_metric: str
    best_value: float
    best_params: dict[str, object]
    trial_records: tuple[TrialRecord, ...]


def run_optuna_optimization(
    *,
    base_config: ConfigDict,
    optimization_cfg: ConfigDict,
    search_space: list[SearchParameter],
    run_pipeline_fn: RunPipelineFn,
    optuna_module: ModuleType | None = None,
) -> OptunaResult:
    """Run an Optuna study against the current pipeline.

    Args:
        base_config: Root config template.
        optimization_cfg: ``optimization`` section mapping.
        search_space: Parsed search parameters.
        run_pipeline_fn: Pipeline executor callable.
        optuna_module: Optional injected Optuna module for tests.

    Returns:
        Study result summary.
    """
    optuna = optuna_module if optuna_module is not None else _import_optuna()
    study_name = as_str(optimization_cfg.get("study_name", "relic_hpo"), "optimization.study_name")
    objective_metric = as_str(
        optimization_cfg.get("objective_metric", "val_auprc"),
        "optimization.objective_metric",
    )
    direction = as_str(optimization_cfg.get("direction", "maximize"), "optimization.direction")
    _validate_direction(direction)

    execution_cfg = _as_config_dict(optimization_cfg.get("execution", {}), "optimization.execution")
    budget_cfg = _as_config_dict(optimization_cfg.get("budget", {}), "optimization.budget")
    n_trials = as_int(budget_cfg.get("n_trials", 20), "optimization.budget.n_trials")
    timeout_minutes = as_float(
        budget_cfg.get("timeout_minutes", 120),
        "optimization.budget.timeout_minutes",
    )
    timeout_seconds = max(int(timeout_minutes * 60), 1)

    sampler_cfg = _as_config_dict(optimization_cfg.get("sampler", {}), "optimization.sampler")
    pruner_cfg = _as_config_dict(optimization_cfg.get("pruner", {}), "optimization.pruner")
    storage_cfg = _as_config_dict(optimization_cfg.get("storage", {}), "optimization.storage")

    sampler = _build_sampler(optuna, sampler_cfg)
    pruner = _build_pruner(optuna, pruner_cfg)
    storage_url = _resolve_storage_url(storage_cfg=storage_cfg)

    study = optuna.create_study(
        study_name=study_name,
        direction=direction,
        sampler=sampler,
        pruner=pruner,
        storage=storage_url,
        load_if_exists=True,
    )

    catch_oom_as_pruned = as_bool(
        execution_cfg.get("catch_oom_as_pruned", True),
        "optimization.execution.catch_oom_as_pruned",
    )

    def objective(trial: object) -> float:
        sampled_values = _sample_trial_values(trial=trial, search_space=search_space)
        typed_trial = cast(_TrialProtocol, trial)
        trial_number = int(typed_trial.number)
        try:
            trial_result = execute_trial(
                base_config=base_config,
                search_space=search_space,
                sampled_values=sampled_values,
                study_name=study_name,
                trial_number=trial_number,
                objective_metric=objective_metric,
                direction=direction,
                execution_cfg=execution_cfg,
                run_pipeline_fn=run_pipeline_fn,
            )
        except RuntimeError as exc:
            if catch_oom_as_pruned and _is_cuda_oom(exc):
                raise optuna.TrialPruned(f"OOM pruned: {exc}") from exc
            raise
        _report_objective_history(optuna=optuna, trial=typed_trial, trial_result=trial_result)
        return trial_result.objective_value

    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout_seconds,
    )

    trial_records = tuple(_collect_trial_records(study_trials=study.trials))
    return OptunaResult(
        study_name=study_name,
        direction=direction,
        objective_metric=objective_metric,
        best_value=float(study.best_value),
        best_params=dict(study.best_params),
        trial_records=trial_records,
    )


def _import_optuna() -> ModuleType:
    """Import Optuna lazily with actionable error."""
    try:
        module = importlib.import_module("optuna")
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Optuna is required for optimization backend 'optuna'. "
            "Install it with: pip install optuna"
        ) from exc
    return module


def _build_sampler(optuna: ModuleType, sampler_cfg: ConfigDict) -> object:
    """Build sampler from config."""
    samplers = optuna.samplers
    sampler_name = as_str(sampler_cfg.get("name", "TPESampler"), "optimization.sampler.name")
    seed = as_int(sampler_cfg.get("seed", 0), "optimization.sampler.seed")
    if sampler_name == "TPESampler":
        return samplers.TPESampler(seed=seed)
    if sampler_name == "RandomSampler":
        return samplers.RandomSampler(seed=seed)
    raise ValueError("optimization.sampler.name must be TPESampler or RandomSampler")


def _build_pruner(optuna: ModuleType, pruner_cfg: ConfigDict) -> object:
    """Build pruner from config."""
    pruners = optuna.pruners
    pruner_name = as_str(pruner_cfg.get("name", "MedianPruner"), "optimization.pruner.name")
    if pruner_name == "MedianPruner":
        return pruners.MedianPruner(
            n_startup_trials=as_int(
                pruner_cfg.get("n_startup_trials", 5),
                "optimization.pruner.n_startup_trials",
            ),
            n_warmup_steps=as_int(
                pruner_cfg.get("n_warmup_steps", 3),
                "optimization.pruner.n_warmup_steps",
            ),
        )
    if pruner_name == "NopPruner":
        return pruners.NopPruner()
    raise ValueError("optimization.pruner.name must be MedianPruner or NopPruner")


def _resolve_storage_url(*, storage_cfg: ConfigDict) -> str | None:
    """Resolve optional Optuna storage URL and ensure local dirs exist."""
    storage_type = as_str(storage_cfg.get("type", "sqlite"), "optimization.storage.type").lower()
    if storage_type == "none":
        return None
    if storage_type != "sqlite":
        raise ValueError("optimization.storage.type must be sqlite or none")

    storage_url = as_str(
        storage_cfg.get("url", "sqlite:///artifacts/hpo/relic_hpo.db"),
        "optimization.storage.url",
    )
    if storage_url.startswith("sqlite:///"):
        sqlite_path = Path(storage_url[len("sqlite:///") :])
        sqlite_path.parent.mkdir(parents=True, exist_ok=True)
    return storage_url


def _sample_trial_values(
    *,
    trial: object,
    search_space: Sequence[SearchParameter],
) -> dict[str, object]:
    """Sample all configured parameters for one trial."""
    sampled: dict[str, object] = {}
    for parameter in search_space:
        sampled[parameter.name] = sample_parameter(trial=trial, parameter=parameter)
    return sampled


def _report_objective_history(
    *,
    optuna: ModuleType,
    trial: _TrialProtocol,
    trial_result: TrialExecutionResult,
) -> None:
    """Report trial intermediate values for pruner compatibility."""
    trial.set_user_attr("run_id", trial_result.run_id)
    trial.set_user_attr("train_csv_path", str(trial_result.train_csv_path))
    trial.set_user_attr("checkpoint_path", str(trial_result.checkpoint_path))
    for step, value in enumerate(trial_result.objective_history, start=1):
        trial.report(value, step)
        if trial.should_prune():
            raise optuna.TrialPruned(f"Pruned at step {step}")


def _collect_trial_records(*, study_trials: Sequence[object]) -> list[TrialRecord]:
    """Collect serializable trial records from study trials."""
    records: list[TrialRecord] = []
    for raw_trial in study_trials:
        trial = cast(_FrozenTrialProtocol, raw_trial)
        user_attrs_raw = trial.user_attrs
        run_id_raw = user_attrs_raw.get("run_id")
        run_id = run_id_raw if isinstance(run_id_raw, str) else None
        state = _trial_state_name(trial.state)
        value_raw = trial.value
        value = float(value_raw) if isinstance(value_raw, (int, float)) else None
        params_raw = trial.params
        records.append(
            TrialRecord(
                number=int(trial.number),
                state=state,
                value=value,
                run_id=run_id,
                params=dict(params_raw),
            )
        )
    return records


def _trial_state_name(state: object) -> str:
    """Normalize Optuna trial state value to string."""
    state_name = getattr(state, "name", None)
    if isinstance(state_name, str):
        return state_name
    return str(state)


def _is_cuda_oom(exc: RuntimeError) -> bool:
    """Return whether runtime error message indicates CUDA OOM."""
    return "out of memory" in str(exc).lower()


def _validate_direction(direction: str) -> None:
    """Validate direction string."""
    if direction.lower() not in {"maximize", "minimize"}:
        raise ValueError("optimization.direction must be 'maximize' or 'minimize'")


def _as_config_dict(value: object, field_name: str) -> ConfigDict:
    """Cast optional section to config mapping."""
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be a mapping")
    return dict(value)


__all__ = ["OptunaResult", "TrialRecord", "run_optuna_optimization"]
