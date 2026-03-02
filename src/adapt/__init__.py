"""Domain adaptation utilities."""

from src.adapt.shot import (
    DomainAdaptationConfig,
    OutputHeadFeatureHook,
    ShotOptimizerConfig,
    ShotSchedulerConfig,
    assign_pseudo_labels,
    compute_centroids,
    diversity_loss,
    entropy_loss,
    freeze_parameters_by_prefix,
    logits_to_probabilities,
    parse_domain_adaptation_config,
    pseudo_label_loss,
    should_run_shot_adaptation,
)

__all__ = [
    "DomainAdaptationConfig",
    "OutputHeadFeatureHook",
    "ShotOptimizerConfig",
    "ShotSchedulerConfig",
    "assign_pseudo_labels",
    "compute_centroids",
    "diversity_loss",
    "entropy_loss",
    "freeze_parameters_by_prefix",
    "logits_to_probabilities",
    "parse_domain_adaptation_config",
    "pseudo_label_loss",
    "should_run_shot_adaptation",
]
