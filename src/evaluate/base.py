"""Generic evaluator for validation and test stages."""

from __future__ import annotations

import math
from collections.abc import Callable

import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch import nn
from torch.utils.data import DataLoader

from src.utils.losses import LossConfig, binary_classification_loss


def _safe_float(value: float) -> float:
    """Convert metric value to a finite float.

    Args:
        value: Candidate numeric metric value.

    Returns:
        Original value if finite, otherwise ``0.0``.
    """
    if math.isnan(value) or math.isinf(value):
        return 0.0
    return float(value)


class Evaluator:
    """Metric computation for a single data loader pass.

    Args:
        metrics: Metric names to compute.
        loss_config: Loss hyperparameters for consistent loss reporting.
    """

    def __init__(self, metrics: list[str], loss_config: LossConfig) -> None:
        self.metrics = [metric.lower() for metric in metrics]
        self.loss_config = loss_config

    @staticmethod
    def _forward_model(model: nn.Module, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Execute model forward and validate output contract.

        Args:
            model: Model to evaluate.
            batch: Model input batch on target device.

        Returns:
            Model output dictionary.

        Raises:
            ValueError: If model output is not a dictionary.
        """
        try:
            output = model(**batch)
        except TypeError:
            output = model(batch=batch)
        if not isinstance(output, dict):
            raise ValueError("Model forward output must be a dictionary")
        return output

    @staticmethod
    def _binary_stats(labels: torch.Tensor, predictions: torch.Tensor) -> tuple[float, float]:
        """Compute sensitivity and specificity.

        Args:
            labels: Ground-truth binary labels.
            predictions: Predicted binary labels.

        Returns:
            Tuple of ``(sensitivity, specificity)``.
        """
        matrix = confusion_matrix(
            labels.cpu().numpy(),
            predictions.cpu().numpy(),
            labels=[0, 1],
        )
        tn, fp, fn, tp = matrix.ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        return sensitivity, specificity

    @staticmethod
    def _score_auc_metric(
        has_both_classes: bool,
        score_fn: Callable[[], float],
    ) -> float:
        """Return AUC-like score when both classes are present, else ``0.0``."""
        if not has_both_classes:
            return 0.0
        return _safe_float(score_fn())

    def _metric_scorers(
        self,
        labels: torch.Tensor,
        probabilities: torch.Tensor,
        predictions: torch.Tensor,
        has_both_classes: bool,
    ) -> dict[str, Callable[[], float]]:
        """Build score functions keyed by metric name."""
        label_array = labels.cpu().numpy()
        prob_array = probabilities.cpu().numpy()
        pred_array = predictions.cpu().numpy()
        needs_binary_stats = "sensitivity" in self.metrics or "specificity" in self.metrics
        sensitivity = 0.0
        specificity = 0.0
        if needs_binary_stats:
            sensitivity, specificity = self._binary_stats(labels=labels, predictions=predictions)

        def score_auroc() -> float:
            return self._score_auc_metric(
                has_both_classes=has_both_classes,
                score_fn=lambda: roc_auc_score(label_array, prob_array),
            )

        def score_auprc() -> float:
            return self._score_auc_metric(
                has_both_classes=has_both_classes,
                score_fn=lambda: average_precision_score(label_array, prob_array),
            )

        return {
            "auroc": score_auroc,
            "auprc": score_auprc,
            "accuracy": lambda: _safe_float(accuracy_score(label_array, pred_array)),
            "sensitivity": lambda: _safe_float(sensitivity),
            "specificity": lambda: _safe_float(specificity),
            "precision": lambda: _safe_float(
                precision_score(label_array, pred_array, zero_division=0)
            ),
            "recall": lambda: _safe_float(recall_score(label_array, pred_array, zero_division=0)),
            "f1": lambda: _safe_float(f1_score(label_array, pred_array, zero_division=0)),
            "mcc": lambda: _safe_float(matthews_corrcoef(label_array, pred_array)),
        }

    def _compute_metrics(
        self,
        labels: torch.Tensor,
        probabilities: torch.Tensor,
    ) -> dict[str, float]:
        """Compute configured metrics for binary classification.

        Args:
            labels: Ground-truth binary labels.
            probabilities: Predicted probabilities in ``[0, 1]``.

        Returns:
            Metric dictionary without split prefix.
        """
        results: dict[str, float] = {}
        has_both_classes = torch.unique(labels).numel() > 1
        predictions = (probabilities >= 0.5).long()
        scorers = self._metric_scorers(
            labels=labels,
            probabilities=probabilities,
            predictions=predictions,
            has_both_classes=has_both_classes,
        )
        for metric in self.metrics:
            score_fn = scorers.get(metric)
            if score_fn is None:
                continue
            results[metric] = score_fn()
        return results

    def evaluate(
        self,
        model: nn.Module,
        data_loader: DataLoader[dict[str, torch.Tensor]],
        device: torch.device,
        prefix: str | None = "val",
    ) -> dict[str, float]:
        """Evaluate metrics on a loader.

        The caller controls ``model.eval()`` and ``torch.no_grad()`` context.

        Args:
            model: Model to evaluate.
            data_loader: Data loader for the split.
            device: Device where evaluation runs.
            prefix: Optional metric name prefix for output keys.

        Returns:
            Metric dictionary with prefixed names.
        """
        all_probs: list[torch.Tensor] = []
        all_labels: list[torch.Tensor] = []
        total_loss = 0.0
        batch_count = 0

        for batch in data_loader:
            batch_count += 1
            prepared_batch = {key: value.to(device) for key, value in batch.items()}
            output = self._forward_model(model=model, batch=prepared_batch)
            logits = output["logits"]
            labels = prepared_batch["label"].float()
            loss = binary_classification_loss(
                logits=logits,
                labels=labels,
                loss_config=self.loss_config,
                reduction="mean",
            )
            total_loss += float(loss.detach().item())
            reduced_logits = (
                logits.squeeze(-1) if logits.dim() > 1 and logits.size(-1) == 1 else logits
            )
            all_probs.append(torch.sigmoid(reduced_logits).detach().cpu())
            all_labels.append(labels.detach().cpu())

        probs_tensor = torch.cat(all_probs, dim=0)
        labels_tensor = torch.cat(all_labels, dim=0).long()
        metric_values = self._compute_metrics(labels=labels_tensor, probabilities=probs_tensor)
        metric_values["loss"] = total_loss / max(1, batch_count)
        if prefix is None:
            return metric_values
        return {f"{prefix}_{key}": value for key, value in metric_values.items()}
