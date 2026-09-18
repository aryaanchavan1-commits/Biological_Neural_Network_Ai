"""Efficiency metrics for SNN models.

Parameter count, model size, spike counts, firing rates, sparsity,
and connectivity analysis.
"""

from __future__ import annotations

from typing import List, Sequence, Union

import numpy as np
import torch
import torch.nn as nn


def parameter_count(model: nn.Module) -> int:
    """Total number of learnable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_size_mb(model: nn.Module) -> float:
    """Model size in megabytes (float32)."""
    num_params = parameter_count(model)
    return (num_params * 4) / (1024 * 1024)


def total_spikes(spike_history: Union[List[torch.Tensor], torch.Tensor]) -> int:
    """Total number of spikes across all recorded time steps.

    Args:
        spike_history: Either a list of tensors (one per time step) or
                       a single stacked tensor of shape (T, ...).
    """
    if isinstance(spike_history, list):
        return int(sum(s.float().sum().item() for s in spike_history))
    return int(spike_history.float().sum().item())


def average_firing_rate(spike_history: Union[List[torch.Tensor], torch.Tensor]) -> float:
    """Mean firing rate: total_spikes / total_neurons_across_time."""
    if isinstance(spike_history, list):
        if not spike_history:
            return 0.0
        total = sum(s.float().sum().item() for s in spike_history)
        # T × batch × neurons
        first = spike_history[0]
        T = len(spike_history)
        n_elements = T * first.numel()
    else:
        total = spike_history.float().sum().item()
        n_elements = spike_history.numel()

    return float(total / max(n_elements, 1))


def sparsity(spike_history: Union[List[torch.Tensor], torch.Tensor]) -> float:
    """Fraction of zeros (non-spikes) in the spike train.

    1.0 = completely silent, 0.0 = every neuron fires every step.
    """
    return 1.0 - average_firing_rate(spike_history)


def active_synapses(weights: torch.Tensor, threshold: float = 1e-3) -> int:
    """Number of synapses with absolute weight above ``threshold``."""
    w = weights.detach().cpu()
    return int((w.abs() > threshold).sum().item())


def connectivity_sparsity(weights: torch.Tensor, threshold: float = 1e-3) -> float:
    """Fraction of synapses below ``threshold`` (i.e. effectively disconnected)."""
    w = weights.detach().cpu()
    total = w.numel()
    if total == 0:
        return 1.0
    inactive = (w.abs() <= threshold).sum().item()
    return float(inactive / total)
