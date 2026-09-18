"""Prediction error computation — multiple error metrics."""

from __future__ import annotations

from typing import Any, Dict, Literal, Optional

import torch
import torch.nn as nn

from .base import BasePrediction


class PredictionError(BasePrediction):
    """Computes prediction error in one of several modes.

    Modes:
        * ``"signed"``   — raw difference ``prediction - target``.
        * ``"absolute"`` — ``|prediction - target|``.
        * ``"squared"``  — ``(prediction - target)^2``.
        * ``"surprise"`` — negative log-likelihood assuming the prediction
          defines a Gaussian mean with unit variance.

    Parameters:
        error_type:   One of the modes listed above.
        clamp_range:  Optional ``(min, max)`` to clamp signed errors.
    """

    def __init__(
        self,
        error_type: Literal["signed", "absolute", "squared", "surprise"] = "squared",
        clamp_range: Optional[tuple[float, float]] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(config)
        self.error_type = error_type
        self.clamp_range = clamp_range
        self._running_error: float = 0.0

    # ------------------------------------------------------------------

    def predict(self, context: torch.Tensor) -> torch.Tensor:
        """Identity pass-through (error module does not generate predictions)."""
        return context

    def compute_error(
        self, prediction: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """Compute the configured error metric."""
        if self.error_type == "signed":
            error = prediction - target
            if self.clamp_range is not None:
                error = error.clamp(self.clamp_range[0], self.clamp_range[1])
        elif self.error_type == "absolute":
            error = torch.abs(prediction - target)
        elif self.error_type == "squared":
            error = (prediction - target) ** 2
        elif self.error_type == "surprise":
            neg_log_prob = 0.5 * ((prediction - target) ** 2)
            error = neg_log_prob
        else:
            raise ValueError(f"Unknown error_type '{self.error_type}'.")

        scalar = error.mean().item()
        self._running_error = 0.99 * self._running_error + 0.01 * scalar
        self._error_history.append(scalar)
        self._prediction_count += 1
        return error

    def reset(self) -> None:
        self._running_error = 0.0
        self._error_history.clear()
        self._prediction_count = 0

    def get_state(self) -> Dict[str, Any]:
        base = super().get_state()
        base["running_error"] = self._running_error
        base["error_type"] = self.error_type
        return base

    def extra_repr(self) -> str:
        return f"error_type={self.error_type}, clamp_range={self.clamp_range}"
