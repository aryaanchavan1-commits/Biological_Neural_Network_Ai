"""Compartment-style dendritic computation — parallel branches with recombination."""

from __future__ import annotations

from typing import Any, Dict, Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseDendrite


class CompartmentDendrite(BaseDendrite):
    """Splits input across *n_branches* compartments, each with its own
    linear transform and nonlinearity, then recombines.

    Combination modes:
        * ``"multiply"``  — element-wise product of branch outputs.
        * ``"add"``       — element-wise sum.
        * ``"max"``       — element-wise maximum.
        * ``"attention"`` — learned attention over branches.

    Parameters:
        in_features:    Dimensionality of the input.
        n_branches:     Number of dendritic compartments.
        combine:        Recombination strategy.
        activation:     Nonlinearity applied inside each compartment.
        out_features:   Output dimensionality (defaults to *in_features*).
    """

    def __init__(
        self,
        in_features: int,
        n_branches: int = 4,
        combine: Literal["multiply", "add", "max", "attention"] = "add",
        activation: str = "relu",
        out_features: Optional[int] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(config)
        self.in_features = in_features
        self.n_branches = n_branches
        self.combine = combine
        out_feat = out_features or in_features

        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_features, out_feat),
                self._make_activation(activation),
            )
            for _ in range(n_branches)
        ])

        if combine == "attention":
            self._gate = nn.Sequential(
                nn.Linear(in_features, n_branches),
                nn.Softmax(dim=-1),
            )

        self._branch_outputs: list[torch.Tensor] = []

    # ------------------------------------------------------------------

    def forward(
        self,
        input_spikes: torch.Tensor,
        branch_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        self._branch_outputs = [branch(input_spikes) for branch in self.branches]

        if self.combine == "multiply":
            out = self._branch_outputs[0]
            for b in self._branch_outputs[1:]:
                out = out * b
        elif self.combine == "max":
            stacked = torch.stack(self._branch_outputs, dim=0)
            out, _ = stacked.max(dim=0)
        elif self.combine == "attention":
            gates = self._gate(input_spikes)
            stacked = torch.stack(self._branch_outputs, dim=1)
            out = (gates.unsqueeze(-1) * stacked).sum(dim=1)
        else:
            out = torch.stack(self._branch_outputs, dim=0).sum(dim=0)

        if branch_weights is not None:
            out = out * branch_weights

        for i, b_out in enumerate(self._branch_outputs):
            self._branch_activity[f"branch_{i}_mean"] = b_out.mean().item()

        return out

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, n_branches={self.n_branches}, "
            f"combine={self.combine}"
        )

    @staticmethod
    def _make_activation(name: str) -> nn.Module:
        activations = {
            "relu": nn.ReLU(inplace=True),
            "sigmoid": nn.Sigmoid(),
            "tanh": nn.Tanh(),
            "silu": nn.SiLU(inplace=True),
        }
        if name not in activations:
            raise ValueError(f"Unsupported activation '{name}'. Choose from {list(activations)}")
        return activations[name]
