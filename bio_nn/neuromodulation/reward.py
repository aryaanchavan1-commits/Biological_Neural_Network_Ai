"""Reward-based neuromodulation — dopamine-like signal."""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from .base import BaseNeuromodulator


class RewardModulator(BaseNeuromodulator):
    """Modulates learning via a global reward signal.

    Implements a dopamine-like mechanism::

        modulated = signal * (dopamine_level * eligibility_trace)

    where the eligibility trace is a decaying running average of recent
    activity.

    Parameters:
        signal_size:   Dimensionality of the signal to modulate.
        dopamine_level:   Gain factor for the reward signal.
        reward_decay:     Decay rate for the eligibility trace.
        baseline_reward:  Baseline ( tonic ) dopamine level.
    """

    def __init__(
        self,
        signal_size: int,
        dopamine_level: float = 1.0,
        reward_decay: float = 0.9,
        baseline_reward: float = 0.0,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(config)
        self.signal_size = signal_size
        self.dopamine_level = dopamine_level
        self.reward_decay = reward_decay
        self.baseline_reward = baseline_reward

        self.register_buffer("_eligibility", torch.zeros(signal_size))
        self._reward_signal: float = baseline_reward
        self._prediction_error: float = 0.0

    # ------------------------------------------------------------------

    def modulate(
        self, signal: torch.Tensor, context: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Apply reward-based modulation.

        Args:
            signal:  ``(batch, signal_size)`` tensor.
            context: Optional ``(batch, 1)`` reward signal.

        Returns:
            Modulated signal.
        """
        if context is not None:
            reward = context.mean().item()
        else:
            reward = self.baseline_reward

        self._prediction_error = reward - self._reward_signal
        self._reward_signal = reward

        with torch.no_grad():
            activity = signal.mean(dim=0)
            self._eligibility = (
                self.reward_decay * self._eligibility + (1 - self.reward_decay) * activity
            )

        modulation = self.dopamine_level * self._reward_signal * self._eligibility
        modulated = signal * modulation.unsqueeze(0)

        self._modulation_history.append(self._reward_signal)
        return modulated

    def reset(self) -> None:
        self._eligibility.zero_()
        self._reward_signal = self.baseline_reward
        self._prediction_error = 0.0
        self._modulation_history.clear()

    def get_state(self) -> Dict[str, Any]:
        base = super().get_state()
        base["eligibility"] = self._eligibility.clone()
        base["reward_signal"] = self._reward_signal
        base["prediction_error"] = self._prediction_error
        base["dopamine_level"] = self.dopamine_level
        return base

    def extra_repr(self) -> str:
        return (
            f"signal_size={self.signal_size}, dopamine_level={self.dopamine_level}, "
            f"decay={self.reward_decay}, baseline={self.baseline_reward}"
        )
