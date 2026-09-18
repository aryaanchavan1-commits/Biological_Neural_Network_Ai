"""Dendritic nonlinear gating — subsets of inputs gated by separate branches."""

from __future__ import annotations

from typing import Any, Dict, Literal, Optional

import torch
import torch.nn as nn

from .base import BaseDendrite


class NonlinearDendrite(BaseDendrite):
    """Each branch gates a different subset of the input features.

    The input is partitioned (with overlap) across *n_branches* groups.
    Each branch applies its own learned gate nonlinearity to its slice.

    Parameters:
        in_features: Dimensionality of the input.
        n_branches:  Number of gating branches.
        gate_type:   Nonlinearity for the gate (``"relu"``, ``"sigmoid"``,
                     ``"tanh"``).
        overlap:     Number of shared features between adjacent branches.
    """

    def __init__(
        self,
        in_features: int,
        n_branches: int = 4,
        gate_type: Literal["relu", "sigmoid", "tanh"] = "sigmoid",
        overlap: int = 0,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(config)
        self.in_features = in_features
        self.n_branches = n_branches
        self.gate_type = gate_type
        self.overlap = overlap

        slice_size = max(1, (in_features + overlap * (n_branches - 1)) // n_branches)
        self._slices: list[tuple[int, int]] = []
        start = 0
        for i in range(n_branches):
            end = min(start + slice_size, in_features)
            self._slices.append((start, end))
            start = end - overlap
            if start >= in_features:
                break
        self.n_branches = len(self._slices)

        self.branch_gates = nn.ModuleList([
            nn.Linear(s[1] - s[0], s[1] - s[0])
            for s in self._slices
        ])

        self._gate_fn = self._make_gate(gate_type)

    # ------------------------------------------------------------------

    def forward(
        self,
        input_spikes: torch.Tensor,
        branch_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        output = torch.zeros_like(input_spikes)

        for i, (s, e) in enumerate(self._slices):
            chunk = input_spikes[:, s:e]
            gate = self._gate_fn(self.branch_gates[i](chunk))
            gated = chunk * gate
            output[:, s:e] = output[:, s:e] + gated
            self._branch_activity[f"branch_{i}_mean"] = gated.mean().item()

        if branch_weights is not None:
            output = output * branch_weights

        return output

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, n_branches={self.n_branches}, "
            f"gate_type={self.gate_type}, overlap={self.overlap}"
        )

    @staticmethod
    def _make_gate(name: str) -> nn.Module:
        if name == "relu":
            return nn.ReLU()
        if name == "sigmoid":
            return nn.Sigmoid()
        if name == "tanh":
            return nn.Tanh()
        raise ValueError(f"Unsupported gate_type '{name}'.")
