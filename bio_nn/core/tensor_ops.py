"""Shared tensor utilities used across the BIO-NN framework."""

from __future__ import annotations

from typing import Optional, Union

import torch


def move_to_device(
    tensor: torch.Tensor, device: Union[str, torch.device]
) -> torch.Tensor:
    """Move a tensor to *device*, no-op if already there.

    Args:
        tensor: The tensor to move.
        device: Target device string or ``torch.device``.

    Returns:
        Tensor on the target device.
    """
    device = torch.device(device) if isinstance(device, str) else device
    if tensor.device == device:
        return tensor
    return tensor.to(device)


def get_device(config: Optional[dict] = None) -> torch.device:
    """Resolve the ``torch.device`` from an optional config dict.

    Resolution order:
    1. ``config["device"]`` if present.
    2. CUDA if available, else CPU.

    Args:
        config: Optional configuration dictionary.

    Returns:
        A ``torch.device``.
    """
    if config and "device" in config:
        return torch.device(config["device"])
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def spike_counts(spikes: torch.Tensor) -> torch.Tensor:
    """Count the number of spikes per neuron across the time axis.

    Assumes *spikes* has shape ``(time_steps, n_neurons, ...)`` or
    ``(batch, time_steps, n_neurons, ...)``.  The time axis is ``-3`` for
    4-D tensors and ``0`` for lower-rank tensors.

    Args:
        spikes: Binary spike tensor.

    Returns:
        Per-neuron spike counts (same shape minus the time axis).
    """
    dim = -3 if spikes.ndim >= 3 else 0
    return spikes.sum(dim=dim)


def firing_rates(spikes: torch.Tensor, time_steps: int) -> torch.Tensor:
    """Compute per-neuron firing rate (spikes / time_steps).

    Args:
        spikes:     Binary spike tensor.
        time_steps: Number of simulation time-steps.

    Returns:
        Firing-rate tensor (same shape minus the time axis).
    """
    return spike_counts(spikes).float() / max(time_steps, 1)


def sparsity(spikes: torch.Tensor) -> float:
    """Fraction of zero entries in *spikes*.

    Args:
        spikes: Binary spike tensor.

    Returns:
        Sparsity value in ``[0, 1]`` (1.0 = fully sparse / all zeros).
    """
    total = spikes.numel()
    if total == 0:
        return 1.0
    return 1.0 - (spikes.count_nonzero().item() / total)


def total_spikes(spikes: torch.Tensor) -> int:
    """Total number of spikes (non-zero entries) across the tensor.

    Args:
        spikes: Binary spike tensor.

    Returns:
        Scalar count of spikes.
    """
    return spikes.count_nonzero().item()
