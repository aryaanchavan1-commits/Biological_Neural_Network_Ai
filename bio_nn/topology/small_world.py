from __future__ import annotations

import torch

from .base import BaseTopology


class SmallWorldTopology(BaseTopology):
    """Small-world topology inspired by the Watts-Strogatz model.

    Construction:
        1. Start with a ring lattice where each neuron connects to its
           ``k`` nearest neighbours on each side.
        2. For each connection, with probability ``beta`` rewire the
           target to a randomly chosen neuron (avoiding self-loops and
           duplicates).

    This yields a network with high clustering and short average path
    length — a hallmark of biological neural circuits.

    Config keys:
        k (int):   Number of nearest neighbours on each side.  Default: 4.
        beta (float): Rewiring probability per edge.  Default: 0.1.
    """

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self.k: int = config.get("k", 4)
        self.beta: float = config.get("beta", 0.1)

    def generate(self, n_neurons_in: int, n_neurons_out: int) -> torch.Tensor:
        """Generate a small-world binary mask.

        When ``n_in != n_out`` the ring lattice is built on the
        *pre-synaptic* size and each pre-neuron connects to ``k``
        nearest pre-synaptic neighbours, then a random subset of
        post-synaptic neurons is selected to match the requested
        ``n_out`` dimension.

        For the common case ``n_in == n_out`` the classic Watts-Strogatz
        construction is used directly.

        Args:
            n_neurons_in:  Number of pre-synaptic neurons.
            n_neurons_out: Number of post-synaptic neurons.

        Returns:
            Mask tensor ``(n_in, n_out)`` with binary 0/1 values.
        """
        if n_neurons_in == n_neurons_out:
            return self._ws_ring(n_neurons_in)
        return self._ws_bipartite(n_neurons_in, n_neurons_out)

    # ------------------------------------------------------------------
    # Classic WS ring (square case)
    # ------------------------------------------------------------------

    def _ws_ring(self, n: int) -> torch.Tensor:
        mask = torch.zeros(n, n, dtype=torch.float32)
        half_k = self.k // 2

        for i in range(n):
            targets = []
            for j in range(1, half_k + 1):
                targets.append((i + j) % n)
                targets.append((i - j) % n)

            for t in targets:
                if torch.rand(1).item() < self.beta:
                    new_t = torch.randint(0, n, (1,)).item()
                    while new_t == i:
                        new_t = torch.randint(0, n, (1,)).item()
                    mask[i, new_t] = 1.0
                else:
                    mask[i, t] = 1.0

        return mask

    # ------------------------------------------------------------------
    # Bipartite variant (rectangular case)
    # ------------------------------------------------------------------

    def _ws_bipartite(self, n_in: int, n_out: int) -> torch.Tensor:
        half_k = min(self.k, n_out)
        mask = torch.zeros(n_in, n_out, dtype=torch.float32)

        for i in range(n_in):
            base_targets = set()
            for j in range(1, half_k + 1):
                base_targets.add((i + j) % n_out)
                base_targets.add((i - j) % n_out)

            for t in base_targets:
                if torch.rand(1).item() < self.beta:
                    new_t = torch.randint(0, n_out, (1,)).item()
                    mask[i, new_t] = 1.0
                else:
                    mask[i, t] = 1.0

        return mask
