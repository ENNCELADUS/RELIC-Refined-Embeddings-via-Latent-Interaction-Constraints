"""Unit tests for optimization trial runner helpers."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest
from src.optimize.trial_runner import (
    objective_metric_to_csv_header,
    pick_objective_value,
    read_objective_history,
)


def _write_training_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    fieldnames = ["Epoch", "Epoch Time", "Train Loss", "Val Loss", "Val auprc", "Learning Rate"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def test_objective_metric_to_csv_header_accepts_val_prefix() -> None:
    assert objective_metric_to_csv_header("val_auprc") == "Val auprc"
    assert objective_metric_to_csv_header("auprc") == "Val auprc"


def test_read_objective_history_reads_column(tmp_path: Path) -> None:
    csv_path = tmp_path / "training_step.csv"
    _write_training_csv(
        csv_path,
        [
            {
                "Epoch": 1,
                "Epoch Time": 1.1,
                "Train Loss": 0.8,
                "Val Loss": 0.7,
                "Val auprc": 0.45,
                "Learning Rate": 1.0e-4,
            },
            {
                "Epoch": 2,
                "Epoch Time": 1.2,
                "Train Loss": 0.6,
                "Val Loss": 0.5,
                "Val auprc": 0.61,
                "Learning Rate": 8.0e-5,
            },
        ],
    )

    history, column = read_objective_history(csv_path=csv_path, objective_metric="val_auprc")

    assert column == "Val auprc"
    assert history == pytest.approx([0.45, 0.61])


def test_pick_objective_value_handles_direction() -> None:
    history = [0.3, 0.5, 0.4]
    assert pick_objective_value(history=history, direction="maximize") == pytest.approx(0.5)
    assert pick_objective_value(history=history, direction="minimize") == pytest.approx(0.3)


@pytest.mark.parametrize("direction", ["up", "", "MAX"])
def test_pick_objective_value_rejects_invalid_direction(direction: str) -> None:
    with pytest.raises(ValueError, match="optimization.direction"):
        pick_objective_value(history=[0.1], direction=direction)
