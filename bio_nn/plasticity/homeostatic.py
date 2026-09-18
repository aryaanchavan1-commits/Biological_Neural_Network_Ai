"""Homeostatic plasticity (synaptic scaling + intrinsic adaptation).

Synaptic scaling (Turrigiano et al. 1998):
    For each postsynaptic neuron j:
        scaling_factor = target_rate / (actual_rate_j + ε)
        w_ij *= scaling_factor

Intrinsic homeostatic plasticity:
    Adjust the neuron's firing threshold to maintain the target rate.

Scaling is applied every ``update_interval`` steps.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Deque, Dict, Optional

import torch

from bio_nn.plasticity.base import BasePlasticity


class HomeostaticPlasticity(BasePlasticity):
    """Synaptic scaling with intrinsic threshold adaptation."""

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)
        self.target_rate: float = config.get("target_rate", 0.05)
        self.rate_window: int = config.get("rate_window", 100)
        self.adaptation_rate: float = config.get("adaptation_rate", 0.01)
        self.threshold_adaptation: float = config.get("threshold_adaptation", 0.005)
        self.update_interval: int = config.get("update_interval", 50)
        self.epsilon: float = config.get("epsilon", 1e-8)
        self.w_min: float = config.get("w_min", 0.0)
        self.w_max: float = config.get("w_max", 1.0)

        # Per-postsynaptic-neuron spike history (circular buffer)
        self._spike_history: Optional[Deque[torch.Tensor]] = None
        self._n_post: int = 0
        self._steps_since_update: int = 0
        self._thresholds: Optional[torch.Tensor] = None

    def _init_state(self, n_post: int, device: torch.device) -> None:
        self._n_post = n_post
        self._spike_history = deque(maxlen=self.rate_window)
        self._thresholds = torch.zeros(n_post, device=device)

    def reset_state(self) -> None:
        self._spike_history = None
        self._thresholds = None

    # ------------------------------------------------------------------

    def forward(
        self,
        weights: torch.Tensor,
        pre_spikes: torch.Tensor,
        post_spikes: torch.Tensor,
        *,
        thresholds: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        n_pre, n_post = weights.shape
        device = weights.device

        if self._spike_history is None:
            self._init_state(n_post, device)

        # Record batch-averaged post-spike rate for this step
        step_rate = post_spikes.mean(dim=0)  # (n_post,)
        self._spike_history.append(step_rate.detach())

        self._steps_since_update += 1

        # Only apply scaling at scheduled intervals
        if self._steps_since_update >= self.update_interval:
            self._steps_since_update = 0
            self._apply_synaptic_scaling(weights)
            self._apply_threshold_adaptation()

        self._step_count += 1
        return weights

    # ------------------------------------------------------------------

    def _apply_synaptic_scaling(self, weights: torch.Tensor) -> None:
        if len(self._spike_history) == 0:
            return

        # Average firing rate over the window, shape (n_post,)
        rates = torch.stack(list(self._spike_history), dim=0).mean(dim=0)

        # Scaling factor: target / (actual + ε)
        scaling = self.target_rate / (rates + self.epsilon)

        # Only scale neurons that are too active or too silent
        # Use the adaptation_rate to smooth the scaling
        scaling = 1.0 + self.adaptation_rate * (scaling - 1.0)

        # Avoid pathological scaling
        scaling = scaling.clamp(0.5, 2.0)

        # Apply: w_ij *= scaling_j
        weights.data = weights.data * scaling.unsqueeze(0)
        weights.data = weights.data.clamp(self.w_min, self.w_max)

    def _apply_threshold_adaptation(self) -> None:
        if self._thresholds is None or len(self._spike_history) == 0:
            return

        rates = torch.stack(list(self._spike_history), dim=0).mean(dim=0)
        # Increase threshold if rate too high, decrease if too low
        self._thresholds = self._thresholds + self.threshold_adaptation * (rates - self.target_rate)

    # ------------------------------------------------------------------

    def get_thresholds(self) -> Optional[torch.Tensor]:
        return self._thresholds

    def get_state(self) -> Dict[str, Any]:
        base = super().get_state()
        base["thresholds"] = self._thresholds
        base["steps_since_update"] = self._steps_since_update
        return base

    def get_statistics(self) -> Dict[str, Any]:
        base = super().get_statistics()
        if self._spike_history and len(self._spike_history) > 0:
            rates = torch.stack(list(self._spike_history), dim=0).mean(dim=0)
            base["mean_firing_rate"] = rates.mean().item()
        base.update({
            "target_rate": self.target_rate,
            "rate_window": self.rate_window,
            "update_interval": self.update_interval,
        })
        return base
