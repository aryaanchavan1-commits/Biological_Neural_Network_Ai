"""Abstract base class for prediction modules."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import torch
import torch.nn as nn


class BasePrediction(nn.Module, ABC):
    """Base class for generative / predictive coding modules."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__()
        self._config = config or {}
        self._prediction_count: int = 0
        self._error_history: list[float] = []

    @abstractmethod
    def predict(self, context: torch.Tensor) -> torch.Tensor:
        """Generate a prediction from *context*.

        Args:
            context: Input context tensor ``(batch, ...)``.

        Returns:
            Predicted tensor.
        """
        ...

    @abstractmethod
    def compute_error(
        self, prediction: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """Compute prediction error.

        Args:
            prediction: Model's prediction.
            target:     Ground-truth target.

        Returns:
            Error tensor.
        """
        ...

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        """Convenience alias for ``predict``."""
        return self.predict(context)

    def get_state(self) -> Dict[str, Any]:
        return {
            "config": self._config,
            "prediction_count": self._prediction_count,
            "error_history": list(self._error_history[-100:]),
        }
