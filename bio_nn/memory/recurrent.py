"""Recurrent state memory — hidden state carried across time steps."""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from .base import BaseMemory


class RecurrentMemory(BaseMemory):
    """Maintains a hidden state vector updated each time step.

    Mimics the hidden state of an RNN cell::

        h_t = tanh(W_input @ x + W_hidden @ h_{t-1} + b)

    Parameters:
        input_size:  Dimensionality of the input signal.
        hidden_size: Dimensionality of the hidden state vector.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(config)
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.W_input = nn.Linear(input_size, hidden_size)
        self.W_hidden = nn.Linear(hidden_size, hidden_size, bias=False)

        self._hidden: Optional[torch.Tensor] = None
        self._history: list[torch.Tensor] = []

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def read(self, query: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Return the current hidden state.

        If *query* is provided it is **ignored** for this memory type (the
        hidden state is a fixed-size vector, not content-addressable).
        """
        if self._hidden is None:
            raise RuntimeError("Memory not initialised. Call write() first.")
        self._read_count += 1
        return self._hidden

    def write(self, input_data: torch.Tensor) -> None:
        """Advance the hidden state with *input_data*.

        Args:
            input_data: ``(batch, input_size)`` tensor.
        """
        batch = input_data.shape[0]
        if self._hidden is None or self._hidden.shape[0] != batch:
            self._hidden = torch.zeros(batch, self.hidden_size, device=input_data.device, dtype=input_data.dtype)

        self._hidden = torch.tanh(self.W_input(input_data) + self.W_hidden(self._hidden))
        self._write_count += 1
        self._history.append(self._hidden.detach().mean().item())

    def reset(self) -> None:
        self._hidden = None
        self._history.clear()
        self._read_count = 0
        self._write_count = 0

    def get_state(self) -> Dict[str, Any]:
        base = super().get_state()
        base["hidden"] = self._hidden
        base["hidden_size"] = self.hidden_size
        base["history"] = list(self._history[-100:])
        return base

    def extra_repr(self) -> str:
        return f"input_size={self.input_size}, hidden_size={self.hidden_size}"
