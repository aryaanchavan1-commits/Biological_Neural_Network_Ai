"""Connection and neuron pruning implementations.

Provides four pruning strategies:
* **weight** – remove connections whose absolute weight is below a threshold.
* **activity** – remove connections where pre- or post-neuron activity is low.
* **magnitude** – remove the smallest-weight connections up to a target rate.
* **structured** – remove entire neurons (rows/columns) with low importance.

Neuron pruning removes units whose total synaptic weight or firing rate
drops below a threshold.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from .base import BaseStructuralPlasticity


def _default_config() -> Dict[str, Any]:
    return {
        "pruning_rate": 0.01,
        "weight_threshold": 0.01,
        "activity_threshold": 0.05,
        "strategy": "magnitude",
        "structured": False,
        "neuron_pruning_enabled": False,
        "neuron_firing_threshold": 0.02,
        "neuron_weight_threshold": 0.01,
    }


class ConnectionPruning(BaseStructuralPlasticity):
    """Remove synaptic connections during training.

    Parameters
    ----------
    config : dict, optional
        Keys mirror the module-level ``_default_config``.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        cfg = {**_default_config(), **(config or {})}
        self.pruning_rate: float = cfg["pruning_rate"]
        self.weight_threshold: float = cfg["weight_threshold"]
        self.activity_threshold: float = cfg["activity_threshold"]
        self.strategy: str = cfg["strategy"]
        self.structured: bool = cfg["structured"]
        self.neuron_pruning_enabled: bool = cfg["neuron_pruning_enabled"]
        self.neuron_firing_threshold: float = cfg["neuron_firing_threshold"]
        self.neuron_weight_threshold: float = cfg["neuron_weight_threshold"]

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
        """Apply pruning to *weights* and return the modified tensor."""
        result = weights.clone()

        if self.strategy == "weight":
            mask = self._weight_mask(result)
        elif self.strategy == "activity":
            mask = self._activity_mask(result, activity)
        elif self.strategy == "magnitude":
            mask = self._magnitude_mask(result)
        elif self.strategy == "structured":
            return self._structured_prune(result, activity, epoch, step)
        else:
            mask = self._magnitude_mask(result)

        pruned_count = int((result != 0).sum().item() - (result * mask != 0).sum().item())
        result = result * mask

        if self.neuron_pruning_enabled and activity is not None:
            result = self._neuron_prune(result, activity)

        self._record_pruning(
            max(pruned_count, 0),
            strategy=self.strategy,
            epoch=epoch,
            step=step,
        )
        return result

    # ------------------------------------------------------------------
    # Connection-pruning strategies
    # ------------------------------------------------------------------

    def _weight_mask(self, weights: torch.Tensor) -> torch.Tensor:
        """Zero out connections whose absolute weight is below the threshold."""
        keep = (weights.abs() >= self.weight_threshold).float()
        return keep

    def _activity_mask(
        self,
        weights: torch.Tensor,
        activity: Optional[Dict[str, torch.Tensor]],
    ) -> torch.Tensor:
        """Zero out connections where pre- or post-neuron is inactive."""
        if activity is None:
            return torch.ones_like(weights)

        firing_rate = activity.get("firing_rate")
        if firing_rate is None:
            return torch.ones_like(weights)

        if firing_rate.dim() > 1:
            firing_rate = firing_rate.mean(dim=tuple(range(firing_rate.dim() - 1)))

        active = (firing_rate > self.activity_threshold).float()
        # Keep only connections where both ends are active
        mask = torch.outer(active, active)
        return mask

    def _magnitude_mask(self, weights: torch.Tensor) -> torch.Tensor:
        """Keep only the top fraction of connections by absolute weight.

        This implements a global magnitude pruning mask that prunes
        ``self.pruning_rate`` of the non-zero connections.
        """
        abs_w = weights.abs()
        n_total = (weights != 0).sum().item()
        if n_total == 0:
            return torch.ones_like(weights)

        n_prune = max(0, int(self.pruning_rate * n_total))
        if n_prune == 0:
            return torch.ones_like(weights)

        flat = abs_w.view(-1)
        # Threshold = n_prune-th smallest non-zero value
        nonzero_vals = flat[flat > 0]
        if nonzero_vals.numel() <= n_prune:
            return torch.zeros_like(weights)

        threshold = nonzero_vals.sort().values[n_prune - 1]
        keep = (abs_w >= threshold).float()
        return keep

    def _structured_prune(
        self,
        weights: torch.Tensor,
        activity: Optional[Dict[str, torch.Tensor]],
        epoch: Optional[int],
        step: Optional[int],
    ) -> torch.Tensor:
        """Remove entire neurons (rows) whose total weight is smallest."""
        n_post, n_pre = weights.shape
        row_importance = weights.abs().sum(dim=1)
        n_prune = max(1, int(self.pruning_rate * n_post))
        if n_prune >= n_post:
            return weights

        _, least_important = row_importance.topk(n_prune, largest=False)
        result = weights.clone()
        result[least_important] = 0.0
        pruned = n_prune * n_pre
        self._record_pruning(
            int(pruned),
            strategy="structured",
            neurons_pruned=n_prune,
            epoch=epoch,
            step=step,
        )
        return result

    # ------------------------------------------------------------------
    # Neuron pruning
    # ------------------------------------------------------------------

    def _neuron_prune(
        self,
        weights: torch.Tensor,
        activity: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Remove neurons with low firing rate or low total synaptic weight."""
        firing_rate = activity.get("firing_rate")
        if firing_rate is None:
            return weights

        if firing_rate.dim() > 1:
            firing_rate = firing_rate.mean(dim=tuple(range(firing_rate.dim() - 1)))

        # Neurons that fire too rarely
        inactive = (firing_rate < self.neuron_firing_threshold)
        # Neurons with tiny total weight
        row_totals = weights.abs().sum(dim=1)
        weak = (row_totals < self.neuron_weight_threshold)

        remove = inactive | weak
        if not remove.any():
            return weights

        result = weights.clone()
        result[remove] = 0.0
        self._record_pruning(
            int(remove.sum().item()),
            strategy="neuron_pruning",
        )
        return result
