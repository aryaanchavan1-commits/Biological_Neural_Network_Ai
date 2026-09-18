"""Abstract base class for all plasticity rules.

Every plasticity rule inherits from PlasticityRule (in core/base.py) and
adds rule-specific state tracking and a ``get_statistics`` interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict

import torch
import torch.nn as nn

from bio_nn.core.base import PlasticityRule


class BasePlasticity(PlasticityRule, ABC):
    """Extended base for plasticity rules with state and statistics."""

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__()
        self.config = config
        self._step_count: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @abstractmethod
    def forward(
        self,
        weights: torch.Tensor,
        pre_spikes: torch.Tensor,
        post_spikes: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        ...

    def get_state(self) -> Dict[str, Any]:
        """Return serialisable internal state."""
        return {"step_count": self._step_count}

    def get_statistics(self) -> Dict[str, Any]:
        """Return diagnostic statistics for logging."""
        return {"step_count": self._step_count}
