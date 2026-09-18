"""Abstract base class for all memory modules."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import torch
import torch.nn as nn


class BaseMemory(nn.Module, ABC):
    """Base class for biologically-inspired memory modules.

    Every memory subclass must implement ``read``, ``write``, and ``reset``.
    ``get_state`` is provided as a default that returns the internal state dict.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__()
        self._config = config or {}
        self._read_count: int = 0
        self._write_count: int = 0
        self._history: list = []

    @abstractmethod
    def read(self, query: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Read from memory.

        Args:
            query: Optional query tensor.  Interpretation is
                   subclass-specific (e.g. key for content-addressable memory).

        Returns:
            Memory content tensor.
        """
        ...

    @abstractmethod
    def write(self, input_data: torch.Tensor) -> None:
        """Write to memory.

        Args:
            input_data: Data to store.
        """
        ...

    @abstractmethod
    def reset(self) -> None:
        """Clear all stored memory contents."""
        ...

    def forward(self, input_data: torch.Tensor, query: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Convenience: write then read."""
        self.write(input_data)
        return self.read(query)

    def get_state(self) -> Dict[str, Any]:
        """Return a serialisable snapshot of the memory state."""
        return {
            "config": self._config,
            "read_count": self._read_count,
            "write_count": self._write_count,
            "history_len": len(self._history),
        }
