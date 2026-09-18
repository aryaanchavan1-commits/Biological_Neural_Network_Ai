"""Synaptic memory — information stored as weight patterns with decay."""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from .base import BaseMemory


class SynapticMemory(BaseMemory):
    """Stores information directly in a weight matrix.

    Strong weights persist through consolidation; weak weights decay
    toward zero over time — mimicking long-term potentiation / depression.

    Parameters:
        input_size:             Dimensionality of input vectors.
        memory_size:            Number of memory slots (rows of the matrix).
        decay_rate:             Per-step multiplicative decay (0 = no decay).
        consolidation_threshold: Weights above this magnitude are consolidated
                                 and exempt from decay.
    """

    def __init__(
        self,
        input_size: int,
        memory_size: int,
        decay_rate: float = 0.01,
        consolidation_threshold: float = 0.5,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(config)
        self.input_size = input_size
        self.memory_size = memory_size
        self.decay_rate = decay_rate
        self.consolidation_threshold = consolidation_threshold

        self._weights = nn.Parameter(torch.randn(memory_size, input_size) * 0.01)
        self._read_weights = nn.Linear(input_size, memory_size, bias=False)

        self._consolidation_mask = torch.ones(memory_size, input_size)
        self._access_counts = torch.zeros(memory_size)
        self._history: list[dict] = []

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def read(self, query: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Read from synaptic memory.

        If *query* is provided, performs content-based weighted read;
        otherwise returns the full weight matrix.
        """
        if query is not None:
            attn = torch.softmax(self._read_weights(query), dim=-1)
            return torch.matmul(attn, self._weights)

        self._read_count += 1
        return self._weights

    def write(self, input_data: torch.Tensor) -> None:
        """Update memory weights via Hebbian-like rule.

        Args:
            input_data: ``(batch, input_size)`` vectors to incorporate.
        """
        flat = input_data.reshape(-1, self.input_size)
        with torch.no_grad():
            for vec in flat:
                contribution = torch.outer(vec, vec)
                if contribution.shape != self._weights.shape:
                    contribution = contribution[:self.memory_size, :self.input_size]
                self._weights.data += contribution * 0.01

        self._write_count += 1
        self._history.append({
            "mean_weight": self._weights.mean().item(),
            "max_weight": self._weights.abs().max().item(),
        })

    def decay(self) -> None:
        """Apply decay and consolidation.

        Consolidated (strong) weights are protected from decay.
        """
        with torch.no_grad():
            consolidated = self._weights.abs() > self.consolidation_threshold
            decay_mask = (~consolidated).float()
            self._weights.data *= (1.0 - self.decay_rate * decay_mask)
            self._consolidation_mask = consolidated.float()

    def reset(self) -> None:
        with torch.no_grad():
            self._weights.data.zero_()
        self._consolidation_mask.zero_()
        self._access_counts.zero_()
        self._history.clear()
        self._read_count = 0
        self._write_count = 0

    def get_state(self) -> Dict[str, Any]:
        base = super().get_state()
        base["weights"] = self._weights.data.clone()
        base["consolidation_mask"] = self._consolidation_mask.clone()
        base["mean_weight"] = self._weights.mean().item()
        base["consolidated_fraction"] = self._consolidation_mask.mean().item()
        base["history"] = list(self._history[-100:])
        return base

    def extra_repr(self) -> str:
        return (
            f"input_size={self.input_size}, memory_size={self.memory_size}, "
            f"decay_rate={self.decay_rate}, threshold={self.consolidation_threshold}"
        )
