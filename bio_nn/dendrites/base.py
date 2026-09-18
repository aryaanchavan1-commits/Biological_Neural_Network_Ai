"""Abstract base class for all dendritic computation modules."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import torch
import torch.nn as nn


class BaseDendrite(nn.Module, ABC):
    """Base class for dendritic processing blocks.

    Subclasses implement ``forward`` which splits input across dendritic
    branches and recombines the results.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__()
        self._config = config or {}
        self._branch_activity: Dict[str, float] = {}

    @abstractmethod
    def forward(
        self,
        input_spikes: torch.Tensor,
        branch_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Process input through dendritic branches.

        Args:
            input_spikes:  Input tensor ``(batch, in_features)``.
            branch_weights: Optional per-branch gating weights.

        Returns:
            Combined dendritic output tensor.
        """
        ...

    def get_branch_activity(self) -> Dict[str, float]:
        """Return last-recorded per-branch activity statistics."""
        return dict(self._branch_activity)
