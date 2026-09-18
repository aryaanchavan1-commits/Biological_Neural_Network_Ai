"""Evaluation module for BIO-NN."""

from .metrics import (
    accuracy,
    balanced_accuracy,
    precision,
    recall,
    f1_score,
    confusion_matrix,
    average_accuracy,
    forgetting,
    backward_transfer,
    forward_transfer,
)
from .efficiency import (
    parameter_count,
    model_size_mb,
    total_spikes,
    average_firing_rate,
    sparsity,
    active_synapses,
    connectivity_sparsity,
)
from .robustness import (
    add_gaussian_noise,
    add_salt_pepper_noise,
    corrupt_features,
    evaluate_robustness,
)

__all__ = [
    "accuracy",
    "balanced_accuracy",
    "precision",
    "recall",
    "f1_score",
    "confusion_matrix",
    "average_accuracy",
    "forgetting",
    "backward_transfer",
    "forward_transfer",
    "parameter_count",
    "model_size_mb",
    "total_spikes",
    "average_firing_rate",
    "sparsity",
    "active_synapses",
    "connectivity_sparsity",
    "add_gaussian_noise",
    "add_salt_pepper_noise",
    "corrupt_features",
    "evaluate_robustness",
]
