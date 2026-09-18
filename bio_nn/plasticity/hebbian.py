"""Basic Hebbian plasticity rule.

    Δw_ij = η * pre_i * post_j

Supports optional weight normalisation, hard clipping, and an anti-Hebbian
component that penalises co-active pre/post pairs.
"""

from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn as nn

from bio_nn.plasticity.base import BasePlasticity


class HebbianPlasticity(BasePlasticity):
    """Hebbian / anti-Hebbian learning with weight normalisation."""

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)
        self.lr: float = config.get("learning_rate", 0.01)
        self.normalize: bool = config.get("normalize", False)
        self.w_min: float = config.get("w_min", 0.0)
        self.w_max: float = config.get("w_max", 1.0)
        self.anti_hebbian: bool = config.get("anti_hebbian", False)
        self.anti_lr: float = config.get("anti_learning_rate", self.lr)

    # ------------------------------------------------------------------

    def forward(
        self,
        weights: torch.Tensor,
        pre_spikes: torch.Tensor,
        post_spikes: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        # pre_spikes: (batch, n_pre), post_spikes: (batch, n_post)
        # Outer product averaged over batch → (n_pre, n_post)
        hebbian = torch.einsum("bi,bj->ij", pre_spikes, post_spikes) / pre_spikes.shape[0]

        delta = self.lr * hebbian

        if self.anti_hebbian:
            # Anti-Hebbian: penalise when pre active but post silent
            # term = pre_i * (1 - post_j)  averaged over batch
            anti = torch.einsum(
                "bi,bj->ij", pre_spikes, 1.0 - post_spikes
            ) / pre_spikes.shape[0]
            delta = delta - self.anti_lr * anti

        weights = weights + delta

        if self.normalize:
            norms = weights.norm(dim=1, keepdim=True).clamp(min=1e-8)
            weights = weights / norms

        weights = weights.clamp(self.w_min, self.w_max)
        self._step_count += 1
        return weights

    # ------------------------------------------------------------------

    def get_statistics(self) -> Dict[str, Any]:
        base = super().get_statistics()
        base.update({
            "learning_rate": self.lr,
            "normalize": self.normalize,
            "anti_hebbian": self.anti_hebbian,
        })
        return base
