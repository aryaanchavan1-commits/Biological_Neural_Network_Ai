"""Connection and neuron growth implementations.

Provides three growth strategies:
* **activity-based** – grow connections between co-active neurons.
* **random** – add a fixed fraction of random connections.
* **weight-based** – add connections near existing strong ones (nearest
  neighbour in weight space).

Neuron growth tracks dormant units and recruits them when average
activity crosses a threshold.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from .base import BaseStructuralPlasticity


def _default_config() -> Dict[str, Any]:
    return {
        "growth_rate": 0.01,
        "activity_threshold": 0.1,
        "max_connections": 0.5,
        "weight_init_mean": 0.0,
        "weight_init_std": 0.01,
        "strategy": "activity",
        "neuron_growth_enabled": False,
        "neuron_activity_threshold": 0.3,
    }


class ConnectionGrowth(BaseStructuralPlasticity):
    """Add new synaptic connections during training.

    Supports three connection-growth strategies and optional neuron
    recruitment.

    Parameters
    ----------
    config : dict, optional
        Keys mirror the module-level ``_default_config``.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        cfg = {**_default_config(), **(config or {})}
        self.growth_rate: float = cfg["growth_rate"]
        self.activity_threshold: float = cfg["activity_threshold"]
        self.max_connections: float = cfg["max_connections"]
        self.weight_init_mean: float = cfg["weight_init_mean"]
        self.weight_init_std: float = cfg["weight_init_std"]
        self.strategy: str = cfg["strategy"]
        self.neuron_growth_enabled: bool = cfg["neuron_growth_enabled"]
        self.neuron_activity_threshold: float = cfg["neuron_activity_threshold"]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def modify(
        self,
        weights: torch.Tensor,
        activity: Optional[Dict[str, torch.Tensor]] = None,
        epoch: Optional[int] = None,
        step: Optional[int] = None,
    ) -> torch.Tensor:
        """Apply growth to *weights* and return the modified tensor."""
        if activity is None:
            return weights

        fired = weights  # will be overwritten per strategy
        n_post, n_pre = weights.shape

        max_allowed = int(self.max_connections * n_pre)
        current_per_row = (weights != 0).sum(dim=1)
        headroom = (max_allowed - current_per_row).clamp(min=0).float()
        total_budget = int(headroom.sum().item())

        if total_budget <= 0:
            return weights

        n_new = max(1, int(self.growth_rate * total_budget))

        mask = (weights == 0).float()
        if self.strategy == "activity":
            grown = self._activity_growth(weights, activity, mask, n_new)
        elif self.strategy == "random":
            grown = self._random_growth(weights, mask, n_new)
        elif self.strategy == "weight":
            grown = self._weight_growth(weights, mask, n_new)
        else:
            grown = self._random_growth(weights, mask, n_new)

        result = weights + grown

        if self.neuron_growth_enabled and activity is not None:
            result = self._neuron_growth(result, activity)

        actual = (grown != 0).sum().item()
        self._record_growth(
            int(actual),
            strategy=self.strategy,
            epoch=epoch,
            step=step,
        )
        return result

    # ------------------------------------------------------------------
    # Connection-growth strategies
    # ------------------------------------------------------------------

    def _activity_growth(
        self,
        weights: torch.Tensor,
        activity: Dict[str, torch.Tensor],
        mask: torch.Tensor,
        n_new: int,
    ) -> torch.Tensor:
        """Grow connections where both pre- and post-neurons are active."""
        firing_rate = activity.get("firing_rate")
        if firing_rate is None:
            return torch.zeros_like(weights)

        # Ensure firing_rate is 1-D matching weight dimensions
        if firing_rate.dim() > 1:
            firing_rate = firing_rate.mean(dim=tuple(range(firing_rate.dim() - 1)))

        pre_active = (firing_rate > self.activity_threshold).float()
        post_active = pre_active  # symmetric assumption; caller can override

        # Outer product gives candidate pairs
        candidate = torch.outer(post_active, pre_active) * mask
        n_available = int(candidate.sum().item())
        if n_available == 0:
            return torch.zeros_like(weights)

        n_grow = min(n_new, n_available)
        flat = candidate.view(-1)
        idx = torch.multinomial(flat, n_grow, replacement=False)
        grown = torch.zeros_like(weights)
        grown.view(-1)[idx] = torch.normal(
            self.weight_init_mean, self.weight_init_std, size=(n_grow,),
            device=weights.device, dtype=weights.dtype,
        )
        return grown

    def _random_growth(
        self,
        weights: torch.Tensor,
        mask: torch.Tensor,
        n_new: int,
    ) -> torch.Tensor:
        """Add random connections among currently silent synapses."""
        flat_mask = mask.view(-1)
        n_available = int(flat_mask.sum().item())
        if n_available == 0:
            return torch.zeros_like(weights)

        n_grow = min(n_new, n_available)
        idx = torch.multinomial(flat_mask, n_grow, replacement=False)
        grown = torch.zeros_like(weights)
        grown.view(-1)[idx] = torch.normal(
            self.weight_init_mean, self.weight_init_std, size=(n_grow,),
            device=weights.device, dtype=weights.dtype,
        )
        return grown

    def _weight_growth(
        self,
        weights: torch.Tensor,
        mask: torch.Tensor,
        n_new: int,
    ) -> torch.Tensor:
        """Add connections near existing strong ones (nearest-neighbour heuristic).

        For every candidate zero-weight synapse, compute the average
        absolute weight of its row + column as a proxy for proximity
        in weight space.  Grow the highest-scoring candidates.
        """
        abs_w = weights.abs()
        row_sum = abs_w.sum(dim=1, keepdim=True)  # (n_post, 1)
        col_sum = abs_w.sum(dim=0, keepdim=True)  # (1, n_pre)
        # Proximity score: average of row and column sums at each position
        proximity = (row_sum + col_sum) * 0.5
        candidate = proximity * mask

        flat = candidate.view(-1)
        n_available = int((flat > 0).sum().item())
        if n_available == 0:
            return torch.zeros_like(weights)

        n_grow = min(n_new, n_available)
        idx = torch.multinomial(flat, n_grow, replacement=False)
        grown = torch.zeros_like(weights)
        grown.view(-1)[idx] = torch.normal(
            self.weight_init_mean, self.weight_init_std, size=(n_grow,),
            device=weights.device, dtype=weights.dtype,
        )
        return grown

    # ------------------------------------------------------------------
    # Neuron growth
    # ------------------------------------------------------------------

    def _neuron_growth(
        self,
        weights: torch.Tensor,
        activity: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Recruit a dormant neuron when average activity is too high.

        A dormant neuron is one whose entire row/column in the weight
        matrix is zero.  When the *mean* firing rate exceeds the
        threshold the last dormant neuron is un-silenced by initialising
        a small random row of outgoing weights.
        """
        firing_rate = activity.get("firing_rate")
        if firing_rate is None:
            return weights

        n_post, n_pre = weights.shape
        mean_rate = firing_rate.mean().item()
        if mean_rate < self.neuron_activity_threshold:
            return weights

        # Find a dormant post-neuron (zero outgoing weights)
        row_norms = weights.abs().sum(dim=1)
        dormant = (row_norms == 0).nonzero(as_tuple=False).squeeze(1)
        if dormant.numel() == 0:
            return weights

        recruit = dormant[0].item()
        new_weights = torch.normal(
            self.weight_init_mean, self.weight_init_std,
            size=(1, n_pre), device=weights.device, dtype=weights.dtype,
        )
        result = weights.clone()
        result[recruit] = new_weights.squeeze(0)
        return result
