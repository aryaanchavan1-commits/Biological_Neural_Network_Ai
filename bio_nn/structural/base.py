"""Base class for all structural plasticity modules.

Provides the interface for growth and pruning operations that modify
network topology during training.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn


class BaseStructuralPlasticity(nn.Module, ABC):
    """Abstract base for structural plasticity rules.

    Subclasses must implement ``modify`` which mutates (or returns new)
    weight tensors to reflect connection/neuron growth and pruning.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.config = config or {}
        self._growth_history: List[Dict[str, Any]] = []
        self._pruning_history: List[Dict[str, Any]] = []

    @abstractmethod
    def modify(
        self,
        weights: torch.Tensor,
        activity: Optional[Dict[str, torch.Tensor]] = None,
        epoch: Optional[int] = None,
        step: Optional[int] = None,
    ) -> torch.Tensor:
        """Modify network structure (weights / mask).

        Args:
            weights:  Current weight tensor.
            activity: Dict of neuron activity statistics.  Expected keys
                      depend on the concrete subclass (e.g. ``"firing_rate"``).
            epoch:    Current epoch number.
            step:     Current training step.

        Returns:
            Modified weight tensor (pruned connections zeroed, grown
            connections initialised).
        """
        ...

    # ------------------------------------------------------------------
    # History helpers
    # ------------------------------------------------------------------

    def get_growth_history(self) -> List[Dict[str, Any]]:
        """Return list of growth-event dicts recorded so far."""
        return list(self._growth_history)

    def get_pruning_history(self) -> List[Dict[str, Any]]:
        """Return list of pruning-event dicts recorded so far."""
        return list(self._pruning_history)

    def get_statistics(self) -> Dict[str, Any]:
        """Return aggregate statistics over the full history."""
        total_growth = sum(e.get("count", 0) for e in self._growth_history)
        total_pruned = sum(e.get("count", 0) for e in self._pruning_history)
        return {
            "total_growth_events": len(self._growth_history),
            "total_pruning_events": len(self._pruning_history),
            "total_connections_grown": total_growth,
            "total_connections_pruned": total_pruned,
        }

    def _record_growth(self, count: int, **extra: Any) -> None:
        self._growth_history.append({"count": count, **extra})

    def _record_pruning(self, count: int, **extra: Any) -> None:
        self._pruning_history.append({"count": count, **extra})
