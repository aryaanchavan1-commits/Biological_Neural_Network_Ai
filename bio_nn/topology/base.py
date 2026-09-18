from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class BaseTopology(nn.Module, ABC):
    """Abstract base class for network topology / connectivity masks.

    Subclasses generate a binary mask of shape ``(n_in, n_out)`` that
    controls which synaptic connections exist.  The mask is multiplied
    element-wise with the weight matrix to enforce sparse connectivity.

    Args:
        config: Dictionary of topology-specific parameters.
    """

    def __init__(self, config: dict) -> None:
        super().__init__()
        self.config = config

    @abstractmethod
    def generate(self, n_neurons_in: int, n_neurons_out: int) -> torch.Tensor:
        """Generate an adjacency / weight mask.

        Args:
            n_neurons_in:  Size of the pre-synaptic layer.
            n_neurons_out: Size of the post-synaptic layer.

        Returns:
            Tensor of shape ``(n_in, n_out)`` with binary 0/1 values.
        """
        ...

    def apply_mask(self, weights: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Apply the topology mask to a weight matrix.

        Args:
            weights: Weight tensor of shape ``(n_out, n_in)``.
            mask:    Binary mask of shape ``(n_in, n_out)``.

        Returns:
            Weighted tensor with masked connections zeroed out.
        """
        return weights * mask.t()

    def forward(self, n_neurons_in: int, n_neurons_out: int) -> torch.Tensor:
        return self.generate(n_neurons_in, n_neurons_out)
