"""Adaptive topology: combined growth + pruning with configurable rules.

Applies pruning first (remove weak connections), then growth (add new
connections), mirroring the biological cycle of synapse elimination
preceding synaptogenesis.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch

from .base import BaseStructuralPlasticity
from .growth import ConnectionGrowth
from .pruning import ConnectionPruning


class AdaptiveTopology(BaseStructuralPlasticity):
    """Combined growth + pruning with configurable rules.

    Parameters
    ----------
    config : dict, optional
        Must contain ``"growth"`` and ``"pruning"`` sub-dicts that are
        forwarded to :class:`ConnectionGrowth` and
        :class:`ConnectionPruning` respectively.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        cfg = config or {}
        self.growth = ConnectionGrowth(cfg.get("growth", {}))
        self.pruning = ConnectionPruning(cfg.get("pruning", {}))

    def modify(
        self,
        weights: torch.Tensor,
        activity: Optional[Dict[str, torch.Tensor]] = None,
        epoch: Optional[int] = None,
        step: Optional[int] = None,
    ) -> torch.Tensor:
        """Prune first, then grow.  Returns the modified weight tensor."""
        weights = self.pruning.modify(weights, activity, epoch, step)
        weights = self.growth.modify(weights, activity, epoch, step)
        return weights

    # ------------------------------------------------------------------
    # Delegated history helpers
    # ------------------------------------------------------------------

    def get_growth_history(self) -> List[Dict[str, Any]]:
        return self.growth.get_growth_history()

    def get_pruning_history(self) -> List[Dict[str, Any]]:
        return self.pruning.get_pruning_history()

    def get_statistics(self) -> Dict[str, Any]:
        """Merge stats from both sub-modules."""
        g = self.growth.get_statistics()
        p = self.pruning.get_statistics()
        return {
            "growth": g,
            "pruning": p,
            "total_growth_events": g["total_growth_events"],
            "total_pruning_events": p["total_pruning_events"],
            "total_connections_grown": g["total_connections_grown"],
            "total_connections_pruned": p["total_connections_pruned"],
        }
