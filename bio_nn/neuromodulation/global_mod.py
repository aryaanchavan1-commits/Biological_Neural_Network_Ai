"""Global neuromodulatory signals — ACh, NE, 5-HT analogues."""

from __future__ import annotations

from typing import Any, Dict, Literal, Optional

import torch
import torch.nn as nn

from .base import BaseNeuromodulator


class GlobalModulator(BaseNeuromodulator):
    """Combines multiple neuromodulatory channels into a single output.

    Supported modulators:
        * ``"acetylcholine"``  — scales the learning-rate gain.
        * ``"norepinephrine"`` — scales signal gain / sharpens attention.
        * ``"serotonin"``      — biases toward exploration (adds noise).

    Parameters:
        signal_size:   Dimensionality of the input signal.
        modulators:    List of active modulator names.
        base_modulation: Default modulation value (before learned scaling).
    """

    def __init__(
        self,
        signal_size: int,
        modulators: Optional[list[str]] = None,
        base_modulation: float = 1.0,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(config)
        self.signal_size = signal_size
        self.active_modulators = modulators or ["acetylcholine", "norepinephrine"]
        self.base_modulation = base_modulation

        self._scalers = nn.ModuleDict()
        for mod in self.active_modulators:
            self._scalers[mod] = nn.Linear(signal_size, signal_size)

        self._levels: Dict[str, float] = {m: base_modulation for m in self.active_modulators}

    # ------------------------------------------------------------------

    def modulate(
        self, signal: torch.Tensor, context: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Apply all active modulatory channels.

        Args:
            signal:  ``(batch, signal_size)`` tensor.
            context: Optional context tensor; interpretation depends on
                     which modulators are active.

        Returns:
            Modulated signal.
        """
        modulated = signal

        if "acetylcholine" in self._scalers:
            ach = torch.sigmoid(self._scalers["acetylcholine"](modulated))
            self._levels["acetylcholine"] = ach.mean().item()
            modulated = modulated * ach

        if "norepinephrine" in self._scalers:
            ne = torch.sigmoid(self._scalers["norepinephrine"](modulated))
            gain = 1.0 + ne
            self._levels["norepinephrine"] = gain.mean().item()
            modulated = modulated * gain

        if "serotonin" in self._scalers:
            st = torch.sigmoid(self._scalers["serotonin"](modulated))
            noise_scale = 0.1 * st
            if self.training:
                modulated = modulated + torch.randn_like(modulated) * noise_scale
            self._levels["serotonin"] = st.mean().item()

        self._modulation_history.append(
            sum(self._levels.values()) / max(len(self._levels), 1)
        )
        return modulated

    def reset(self) -> None:
        self._levels = {m: self.base_modulation for m in self.active_modulators}
        self._modulation_history.clear()

    def get_state(self) -> Dict[str, Any]:
        base = super().get_state()
        base["levels"] = dict(self._levels)
        base["active_modulators"] = list(self.active_modulators)
        return base

    def extra_repr(self) -> str:
        return (
            f"signal_size={self.signal_size}, "
            f"active={self.active_modulators}"
        )
