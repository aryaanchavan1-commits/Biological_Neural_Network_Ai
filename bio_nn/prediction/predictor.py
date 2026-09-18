"""Prediction module — linear, MLP, and temporal predictors."""

from __future__ import annotations

from typing import Any, Dict, Literal, Optional

import torch
import torch.nn as nn

from .base import BasePrediction


class Predictor(BasePrediction):
    """Configurable prediction head.

    Predictor types:
        * ``"linear"``  — single affine transform.
        * ``"mlp"``     — 1-2 hidden layers with ReLU.
        * ``"temporal"`` — uses the previous time-step's prediction as
          additional context.

    Parameters:
        context_size:  Dimensionality of the context vector.
        output_size:   Dimensionality of the prediction.
        predictor_type: One of ``"linear"``, ``"mlp"``, ``"temporal"``.
        hidden_size:   Width of hidden layers (MLP / temporal modes).
        n_layers:      Number of hidden layers in MLP mode (1 or 2).
    """

    def __init__(
        self,
        context_size: int,
        output_size: int,
        predictor_type: Literal["linear", "mlp", "temporal"] = "mlp",
        hidden_size: int = 128,
        n_layers: int = 1,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(config)
        self.context_size = context_size
        self.output_size = output_size
        self.predictor_type = predictor_type

        if predictor_type == "linear":
            self._net = nn.Linear(context_size, output_size)
        elif predictor_type == "mlp":
            layers: list[nn.Module] = []
            in_dim = context_size
            for _ in range(n_layers):
                layers += [nn.Linear(in_dim, hidden_size), nn.ReLU(inplace=True)]
                in_dim = hidden_size
            layers.append(nn.Linear(in_dim, output_size))
            self._net = nn.Sequential(*layers)
        elif predictor_type == "temporal":
            self._net = nn.Linear(context_size + output_size, output_size)
        else:
            raise ValueError(f"Unknown predictor_type '{predictor_type}'.")

        self._prev_prediction: Optional[torch.Tensor] = None
        self._history: list[float] = []

    # ------------------------------------------------------------------

    def predict(self, context: torch.Tensor) -> torch.Tensor:
        """Generate a prediction from *context*.

        For the temporal predictor, the previous prediction is concatenated.
        """
        if self.predictor_type == "temporal":
            if self._prev_prediction is None:
                batch = context.shape[0]
                prev = torch.zeros(batch, self.output_size, device=context.device, dtype=context.dtype)
            else:
                prev = self._prev_prediction
            inp = torch.cat([context, prev], dim=-1)
        else:
            inp = context

        prediction = self._net(inp)
        self._prev_prediction = prediction.detach()
        self._prediction_count += 1
        self._history.append(prediction.mean().item())
        return prediction

    def compute_error(
        self, prediction: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """Compute MSE error between *prediction* and *target*."""
        err = ((prediction - target) ** 2).mean()
        self._error_history.append(err.item())
        return err

    def reset(self) -> None:
        self._prev_prediction = None
        self._error_history.clear()
        self._prediction_count = 0
        self._history.clear()

    def get_state(self) -> Dict[str, Any]:
        base = super().get_state()
        base["prev_prediction"] = (
            self._prev_prediction.clone() if self._prev_prediction is not None else None
        )
        base["history"] = list(self._history[-100:])
        return base

    def extra_repr(self) -> str:
        return (
            f"context_size={self.context_size}, output_size={self.output_size}, "
            f"predictor_type={self.predictor_type}"
        )
