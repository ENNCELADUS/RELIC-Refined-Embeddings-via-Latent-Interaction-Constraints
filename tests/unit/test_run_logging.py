"""Unit tests for process-level logging and DDP config flags."""

from __future__ import annotations

import logging

import pytest
import src.run as run_module
from src.utils.config import ConfigDict


def _base_config() -> ConfigDict:
    return {
        "device_config": {"device": "cpu", "ddp_enabled": False},
        "training_config": {},
    }


def test_configure_root_logging_main_rank(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RANK", "0")
    observed: dict[str, object] = {}
    captured_warning_flags: list[bool] = []

    def fake_capture_warnings(enabled: bool) -> None:
        captured_warning_flags.append(enabled)

    def fake_basic_config(**kwargs: object) -> None:
        observed.update(kwargs)

    monkeypatch.setattr(run_module.logging, "captureWarnings", fake_capture_warnings)
    monkeypatch.setattr(run_module.logging, "basicConfig", fake_basic_config)

    run_module._configure_root_logging()

    assert captured_warning_flags == [True]
    assert observed["level"] == logging.INFO
    assert observed["force"] is True


def test_configure_root_logging_non_main_rank(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RANK", "3")
    observed: dict[str, object] = {}

    def fake_basic_config(**kwargs: object) -> None:
        observed.update(kwargs)

    monkeypatch.setattr(run_module.logging, "basicConfig", fake_basic_config)
    monkeypatch.setattr(run_module.logging, "captureWarnings", lambda _: None)

    run_module._configure_root_logging()

    assert observed["level"] == logging.CRITICAL
    assert observed["force"] is True


def test_ddp_find_unused_parameters_uses_strategy_default() -> None:
    config = _base_config()
    assert run_module._ddp_find_unused_parameters(config) is False

    training_cfg = config["training_config"]
    assert isinstance(training_cfg, dict)
    training_cfg["strategy"] = {"type": "staged_unfreeze"}

    assert run_module._ddp_find_unused_parameters(config) is True


def test_ddp_find_unused_parameters_honors_device_override() -> None:
    config = _base_config()
    training_cfg = config["training_config"]
    assert isinstance(training_cfg, dict)
    training_cfg["strategy"] = {"type": "staged_unfreeze"}

    device_cfg = config["device_config"]
    assert isinstance(device_cfg, dict)
    device_cfg["find_unused_parameters"] = False

    assert run_module._ddp_find_unused_parameters(config) is False


def test_metrics_from_config_preserves_case_and_order() -> None:
    eval_cfg: ConfigDict = {"metrics": ["AUROC", "AUPRC", "accuracy"]}

    metrics = run_module._metrics_from_config(eval_cfg)

    assert metrics == ["AUROC", "AUPRC", "accuracy"]


def test_training_validation_metrics_rejects_empty_list() -> None:
    with pytest.raises(
        ValueError,
        match="training_config.logging.validation_metrics must not be empty",
    ):
        run_module._training_validation_metrics({"logging": {"validation_metrics": []}})


def test_metrics_from_config_rejects_non_sequence() -> None:
    with pytest.raises(ValueError, match="evaluate.metrics must be a sequence"):
        run_module._metrics_from_config({"metrics": 123})


def test_training_validation_metrics_lowercases_entries() -> None:
    training_cfg: ConfigDict = {"logging": {"validation_metrics": ["AUPRC", "AuRoC", "accuracy"]}}

    metrics = run_module._training_validation_metrics(training_cfg)

    assert metrics == ["auprc", "auroc", "accuracy"]
