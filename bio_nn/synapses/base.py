from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class BaseSynapse(nn.Module, ABC):
    """Abstract base class for all synapse models.

    A synapse maps pre-synaptic spikes to post-synaptic input currents
    through a learned or fixed weight matrix.
    """

    def __init__(self, in_features: int, out_features: int, config: dict) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.config = config

        self.weights = nn.Parameter(
            torch.empty(out_features, in_features)
        )

    # ------------------------------------------------------------------
    # Interface
    # ------------------------------------------------------------------

    @abstractmethod
    def forward(self, pre_spikes: torch.Tensor) -> torch.Tensor:
        """Compute post-synaptic input from pre-synaptic spikes.

        Args:
            pre_spikes: (batch, in_features) binary spike tensor.

        Returns:
            post_input: (batch, out_features) summed weighted input.
        """
        ...

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def get_weights(self) -> torch.Tensor:
        """Return a detached copy of the weight matrix."""
        return self.weights.detach().clone()

    def set_weights(self, w: torch.Tensor) -> None:
        """Replace the weight matrix with the provided tensor."""
        if w.shape != (self.out_features, self.in_features):
            raise ValueError(
                f"Shape mismatch: expected ({self.out_features}, {self.in_features}), "
                f"got {tuple(w.shape)}"
            )
        with torch.no_grad():
            self.weights.copy_(w)

    def get_statistics(self) -> dict:
        """Return basic weight statistics."""
        w = self.weights.detach()
        return {
            "mean": w.mean().item(),
            "std": w.std().item(),
            "min": w.min().item(),
            "max": w.max().item(),
            "num_weights": w.numel(),
        }

    def extra_repr(self) -> str:
        return f"in={self.in_features}, out={self.out_features}"
