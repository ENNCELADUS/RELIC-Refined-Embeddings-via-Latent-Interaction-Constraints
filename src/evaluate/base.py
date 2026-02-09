"""Generic evaluator for validation and test stages."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as functional
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
    """

    def __init__(self, metrics: list[str]) -> None:
        self.metrics = [metric.lower() for metric in metrics]

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
        label_array = labels.cpu().numpy()
        prob_array = probabilities.cpu().numpy()
        pred_array = predictions.cpu().numpy()

        for metric in self.metrics:
            if metric == "auroc":
                if not has_both_classes:
                    results[metric] = 0.0
                else:
                    results[metric] = _safe_float(roc_auc_score(label_array, prob_array))
            elif metric == "auprc":
                if not has_both_classes:
                    results[metric] = 0.0
                else:
                    results[metric] = _safe_float(average_precision_score(label_array, prob_array))
            elif metric == "accuracy":
                results[metric] = _safe_float(accuracy_score(label_array, pred_array))
            elif metric == "sensitivity":
                sensitivity, _ = self._binary_stats(labels=labels, predictions=predictions)
                results[metric] = _safe_float(sensitivity)
            elif metric == "specificity":
                _, specificity = self._binary_stats(labels=labels, predictions=predictions)
                results[metric] = _safe_float(specificity)
            elif metric == "precision":
                results[metric] = _safe_float(
                    precision_score(label_array, pred_array, zero_division=0)
                )
            elif metric == "recall":
                results[metric] = _safe_float(
                    recall_score(label_array, pred_array, zero_division=0)
                )
            elif metric == "f1":
                results[metric] = _safe_float(f1_score(label_array, pred_array, zero_division=0))
            elif metric == "mcc":
                results[metric] = _safe_float(matthews_corrcoef(label_array, pred_array))
        return results

    def evaluate(
        self,
        model: nn.Module,
        data_loader: DataLoader[dict[str, torch.Tensor]],
        device: torch.device,
        prefix: str = "val",
    ) -> dict[str, float]:
        """Evaluate metrics on a loader.

        The caller controls ``model.eval()`` and ``torch.no_grad()`` context.

        Args:
            model: Model to evaluate.
            data_loader: Data loader for the split.
            device: Device where evaluation runs.
            prefix: Metric name prefix for output keys.

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
            logits = output["logits"].squeeze(-1)
            labels = prepared_batch["label"].float()
            loss = output.get("loss")
            if loss is None:
                loss = functional.binary_cross_entropy_with_logits(logits, labels)
            total_loss += float(loss.detach().item())
            all_probs.append(torch.sigmoid(logits).detach().cpu())
            all_labels.append(labels.detach().cpu())

        probs_tensor = torch.cat(all_probs, dim=0)
        labels_tensor = torch.cat(all_labels, dim=0).long()
        metric_values = self._compute_metrics(labels=labels_tensor, probabilities=probs_tensor)
        metric_values["loss"] = total_loss / max(1, batch_count)
        return {f"{prefix}_{key}": value for key, value in metric_values.items()}
