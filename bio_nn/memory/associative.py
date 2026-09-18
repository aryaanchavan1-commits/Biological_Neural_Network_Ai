"""Associative (Hopfield-inspired) memory — content-addressable retrieval."""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseMemory


class AssociativeMemory(BaseMemory):
    """Modern Hopfield / dense associative memory.

    Stores key-value pairs and retrieves via content-addressable lookup.
    Hebbian-style outer-product storage with optional noise for
    stochastic retrieval.

    Parameters:
        capacity:         Maximum number of stored patterns.
        key_size:         Dimensionality of keys (and values).
        retrieval_noise:  Std-dev of Gaussian noise added during retrieval.
        beta:             Inverse temperature for the softmax read-out.
    """

    def __init__(
        self,
        capacity: int,
        key_size: int,
        retrieval_noise: float = 0.0,
        beta: float = 1.0,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(config)
        self.capacity = capacity
        self.key_size = key_size
        self.retrieval_noise = retrieval_noise
        self.beta = beta

        self.register_buffer("_keys", torch.zeros(capacity, key_size))
        self.register_buffer("_values", torch.zeros(capacity, key_size))
        self._count: int = 0

        self._history: list[dict] = []

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def read(self, query: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Content-addressable retrieval.

        Args:
            query: ``(batch, key_size)`` query vector(s).

        Returns:
            Retrieved pattern(s), same shape as *query*.
        """
        if query is None:
            raise ValueError("AssociativeMemory.read requires a query tensor.")
        if self._count == 0:
            return torch.zeros_like(query)

        n = self._count
        keys = self._keys[:n]
        values = self._values[:n]

        attn = torch.matmul(query, keys.T) * self.beta
        weights = F.softmax(attn, dim=-1)
        retrieved = torch.matmul(weights, values)

        if self.retrieval_noise > 0 and self.training:
            retrieved = retrieved + torch.randn_like(retrieved) * self.retrieval_noise

        self._read_count += 1
        return retrieved

    def write(self, input_data: torch.Tensor) -> None:
        """Store a key-value pair (value == key for auto-association).

        Args:
            input_data: ``(batch, key_size)`` pattern(s) to store.
        """
        flat = input_data.reshape(-1, self.key_size)
        for pattern in flat:
            idx = self._count % self.capacity
            self._keys[idx] = pattern.detach()
            self._values[idx] = pattern.detach()
            self._count += 1

        self._history.append({
            "count": min(self._count, self.capacity),
            "utilisation": min(self._count, self.capacity) / self.capacity,
        })

    def write_pair(
        self, key: torch.Tensor, value: torch.Tensor
    ) -> None:
        """Store an explicit key → value hetero-associative pair."""
        k_flat = key.reshape(-1, self.key_size)
        v_flat = value.reshape(-1, self.key_size)
        for k, v in zip(k_flat, v_flat):
            idx = self._count % self.capacity
            self._keys[idx] = k.detach()
            self._values[idx] = v.detach()
            self._count += 1

    def reset(self) -> None:
        self._keys.zero_()
        self._values.zero_()
        self._count = 0
        self._history.clear()
        self._read_count = 0
        self._write_count = 0

    def get_state(self) -> Dict[str, Any]:
        base = super().get_state()
        n = min(self._count, self.capacity)
        base["keys"] = self._keys[:n].clone()
        base["values"] = self._values[:n].clone()
        base["count"] = n
        base["capacity"] = self.capacity
        base["utilisation"] = n / self.capacity
        base["history"] = list(self._history[-100:])
        return base

    def extra_repr(self) -> str:
        return (
            f"capacity={self.capacity}, key_size={self.key_size}, "
            f"beta={self.beta}, noise={self.retrieval_noise}"
        )
