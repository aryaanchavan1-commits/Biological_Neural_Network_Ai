"""Working memory buffer — fixed-size circular store of recent patterns."""

from __future__ import annotations

from collections import deque
from typing import Any, Dict, Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseMemory


class WorkingMemory(BaseMemory):
    """Circular buffer that stores recent spike patterns.

    Read modes:
        * ``"all"``     — return every stored entry (concatenated).
        * ``"recent"``  — return only the *k* most-recent entries.
        * ``"attended"`` — soft-attention-weighted read over entries.

    Parameters:
        buffer_size:  Maximum number of stored entries.
        entry_size:   Dimensionality of each stored entry.
        read_mode:    One of ``"all"``, ``"recent"``, ``"attended"``.
        top_k:        Number of recent entries returned in ``"recent"`` mode.
    """

    def __init__(
        self,
        buffer_size: int,
        entry_size: int,
        read_mode: Literal["all", "recent", "attended"] = "all",
        top_k: int = 5,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(config)
        self.buffer_size = buffer_size
        self.entry_size = entry_size
        self.read_mode = read_mode
        self.top_k = top_k

        self.register_buffer(
            "_buffer", torch.zeros(buffer_size, entry_size)
        )
        self._position: int = 0
        self._count: int = 0

        self._attn_key = nn.Linear(entry_size, entry_size, bias=False)
        self._attn_query = nn.Linear(entry_size, entry_size, bias=False)

        self._history: list[float] = []

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def read(self, query: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Read contents of the buffer.

        Args:
            query: Optional ``(batch, entry_size)`` query.  Used only in
                   ``"attended"`` mode; ignored otherwise.

        Returns:
            Tensor whose leading dimension is the number of stored entries.
        """
        if self._count == 0:
            return torch.empty(0, self.entry_size, device=self._buffer.device)

        n = min(self._count, self.buffer_size)
        contents = self._buffer[:n]

        if self.read_mode == "recent":
            k = min(self.top_k, n)
            return contents[-k:]

        if self.read_mode == "attended" and query is not None:
            return self._attended_read(contents, query)

        return contents

    def write(self, input_data: torch.Tensor) -> None:
        """Append *input_data* to the circular buffer.

        Args:
            input_data: ``(batch, entry_size)`` — entries are stored
                        individually; only the first sample per batch is
                        kept (batch size must be 1 for deterministic
                        behaviour, or every sample is stored).
        """
        flat = input_data.reshape(-1, self.entry_size)
        for entry in flat:
            idx = self._position % self.buffer_size
            self._buffer[idx] = entry.detach()
            self._position += 1
            self._count += 1
        self._history.append(float(self._count))

    def reset(self) -> None:
        self._buffer.zero_()
        self._position = 0
        self._count = 0
        self._history.clear()

    def get_state(self) -> Dict[str, Any]:
        base = super().get_state()
        n = min(self._count, self.buffer_size)
        base["buffer"] = self._buffer[:n].clone()
        base["position"] = self._position
        base["count"] = self._count
        base["history"] = list(self._history[-100:])
        return base

    def extra_repr(self) -> str:
        return (
            f"buffer_size={self.buffer_size}, entry_size={self.entry_size}, "
            f"read_mode={self.read_mode}, top_k={self.top_k}"
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _attended_read(
        self, contents: torch.Tensor, query: torch.Tensor
    ) -> torch.Tensor:
        """Soft-attention read over buffer contents."""
        q = self._attn_query(query.mean(dim=0, keepdim=True))
        k = self._attn_key(contents)
        scores = torch.matmul(k, q.T).squeeze(-1) / (self.entry_size ** 0.5)
        weights = F.softmax(scores, dim=-1)
        return torch.sum(weights.unsqueeze(-1) * contents, dim=0, keepdim=True)
