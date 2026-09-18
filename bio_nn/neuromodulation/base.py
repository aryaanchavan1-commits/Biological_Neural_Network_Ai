"""Abstract base class for neuromodulators."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import torch
import torch.nn as nn


class BaseNeuromodulator(nn.Module, ABC):
    """Base class for biologically-inspired neuromodulatory signals.

    Subclasses implement ``modulate`` which applies a global or local
    modulation to an input signal.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__()
        self._config = config or {}
        self._modulation_history: list[float] = []

    @abstractmethod
    def modulate(
        self, signal: torch.Tensor, context: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Apply neuromodulation to *signal*.

        Args:
            signal:  Tensor to modulate.
            context: Optional context (reward, novelty, arousal, etc.).

        Returns:
            Modulated signal (same shape as input).
        """
        ...

    def forward(
        self, signal: torch.Tensor, context: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Convenience alias for ``modulate``."""
        return self.modulate(signal, context)

    def get_state(self) -> Dict[str, Any]:
        return {
            "config": self._config,
            "history_len": len(self._modulation_history),
            "history": list(self._modulation_history[-100:]),
        }
