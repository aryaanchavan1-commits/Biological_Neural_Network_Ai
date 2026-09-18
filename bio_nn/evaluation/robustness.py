"""Robustness evaluation utilities.

Implements noise injection functions and a convenience evaluator that
measures model accuracy under varying corruption levels.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


# ------------------------------------------------------------------
# Noise functions
# ------------------------------------------------------------------


def add_gaussian_noise(data: torch.Tensor, std: float = 0.1) -> torch.Tensor:
    """Additive Gaussian noise with standard deviation ``std``."""
    return data + torch.randn_like(data) * std


def add_salt_pepper_noise(data: torch.Tensor, amount: float = 0.05) -> torch.Tensor:
    """Salt-and-pepper noise: randomly set ``amount`` fraction of elements
    to 0 or 1 (assuming data in [0, 1]).
    """
    noisy = data.clone()
    # Salt (1)
    mask_salt = torch.rand_like(data) < (amount / 2)
    noisy[mask_salt] = 1.0
    # Pepper (0)
    mask_pepper = torch.rand_like(data) < (amount / 2)
    noisy[mask_pepper] = 0.0
    return noisy


def corrupt_features(data: torch.Tensor, fraction: float = 0.1) -> torch.Tensor:
    """Randomly zero out a ``fraction`` of input features."""
    mask = torch.rand_like(data) > fraction
    return data * mask.float()


# ------------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------------


def evaluate_robustness(
    model: nn.Module,
    dataloader: DataLoader,
    noise_fn: Callable[[torch.Tensor, float], torch.Tensor],
    noise_levels: List[float],
    device: torch.device,
    num_steps: int = 15,
) -> Dict[str, Any]:
    """Evaluate model accuracy under different noise intensities.

    Args:
        model:        Trained SNN model.
        dataloader:   Test data loader.
        noise_fn:     Noise injection function(data, level) → corrupted data.
        noise_levels: List of noise magnitudes to test.
        device:       Torch device.
        num_steps:    Simulation time steps.

    Returns:
        dict mapping ``noise_level`` → ``accuracy`` (float).
    """
    model.eval()
    results: Dict[str, float] = {}

    for level in noise_levels:
        all_preds: List[torch.Tensor] = []
        all_targets: List[torch.Tensor] = []

        with torch.no_grad():
            for data, targets in dataloader:
                data, targets = data.to(device), targets.to(device)
                noisy_data = noise_fn(data, level)

                out = model(noisy_data)
                if isinstance(out, dict):
                    output = out.get("output", out.get("spikes"))
                else:
                    output = out

                if output is None:
                    continue
                if output.dim() == 3:
                    output = output[-1]

                preds = output.argmax(dim=-1)
                all_preds.append(preds.cpu())
                all_targets.append(targets.cpu())

        if all_preds:
            from .metrics import accuracy as acc_fn
            results[level] = acc_fn(torch.cat(all_preds), torch.cat(all_targets))
        else:
            results[level] = 0.0

    return results
