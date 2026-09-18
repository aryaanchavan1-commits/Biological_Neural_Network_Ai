from __future__ import annotations

import torch

from .base import BaseTopology


class RandomTopology(BaseTopology):
    """Random sparse connectivity.

    Each potential connection exists independently with probability
    ``1 - sparsity``.  A sparsity of 0.0 means fully connected; 1.0
    means no connections at all.

    Config keys:
        sparsity (float): Fraction of connections to remove.  Default: 0.5.
    """

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self.sparsity: float = config.get("sparsity", 0.5)

    def generate(self, n_neurons_in: int, n_neurons_out: int) -> torch.Tensor:
        """Generate a random binary mask.

        Args:
            n_neurons_in:  Number of pre-synaptic neurons.
            n_neurons_out: Number of post-synaptic neurons.

        Returns:
            Mask tensor ``(n_in, n_out)`` with 1s where connections exist.
        """
        mask = torch.bernoulli(
            torch.full((n_neurons_in, n_neurons_out), 1.0 - self.sparsity)
        )
        return mask
