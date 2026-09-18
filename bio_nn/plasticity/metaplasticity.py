"""Metaplasticity — plasticity of plasticity.

Tracks recent weight-change history and adjusts the effective learning rate
dynamically.  Implements a BCM-inspired sliding threshold:

    θ = activity_threshold * (1 + bcm_theta * mean_activity)

When postsynaptic activity is above θ, LTD dominates; below θ, LTP dominates.
The effective learning rate is also scaled down for weights that have changed
recently, preventing runaway potentiation/depression.

Reference: Bienenstock, Cooper & Munro (1982); Abraham & Bear (1996).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch

from bio_nn.plasticity.base import BasePlasticity


class MetaplasticityPlasticity(BasePlasticity):
    """BCM-inspired metaplastic STDP with adaptive learning rates."""

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)
        self.lr: float = config.get("learning_rate", 0.01)
        self.meta_lr: float = config.get("meta_learning_rate", 0.001)
        self.activity_threshold: float = config.get("activity_threshold", 0.1)
        self.bcm_theta: float = config.get("bcm_theta", 0.5)
        self.w_max: float = config.get("w_max", 1.0)
        self.w_min: float = config.get("w_min", 0.0)
        self.tau_pre: float = config.get("tau_pre", 20.0)
        self.tau_post: float = config.get("tau_post", 20.0)

        # Persistent state
        self._pre_trace: Optional[torch.Tensor] = None
        self._post_trace: Optional[torch.Tensor] = None
        self._weight_change_history: Optional[torch.Tensor] = None  # (n_pre, n_post)
        self._activity_history: Optional[torch.Tensor] = None  # (n_post,)
        self._bcm_threshold: Optional[torch.Tensor] = None  # (n_post,)

    def _init_state(self, n_pre: int, n_post: int, device: torch.device) -> None:
        self._pre_trace = torch.zeros(n_pre, device=device)
        self._post_trace = torch.zeros(n_post, device=device)
        self._weight_change_history = torch.zeros(n_pre, n_post, device=device)
        self._activity_history = torch.zeros(n_post, device=device)
        self._bcm_threshold = torch.full((n_post,), self.activity_threshold, device=device)

    def reset_state(self) -> None:
        self._pre_trace = None
        self._post_trace = None
        self._weight_change_history = None
        self._activity_history = None
        self._bcm_threshold = None

    # ------------------------------------------------------------------

    def forward(
        self,
        weights: torch.Tensor,
        pre_spikes: torch.Tensor,
        post_spikes: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        n_pre, n_post = weights.shape
        device = weights.device

        if self._pre_trace is None:
            self._init_state(n_pre, n_post, device)

        # Ensure correct device
        for t in (self._pre_trace, self._post_trace, self._weight_change_history,
                  self._activity_history, self._bcm_threshold):
            if t.device != device:
                return self._relocate_and_forward(weights, pre_spikes, post_spikes, device)

        decay_pre = torch.exp(torch.tensor(-1.0 / self.tau_pre, device=device))
        decay_post = torch.exp(torch.tensor(-1.0 / self.tau_post, device=device))

        # Decay and accumulate traces
        self._pre_trace = self._pre_trace * decay_pre + pre_spikes.mean(dim=0)
        self._post_trace = self._post_trace * decay_post + post_spikes.mean(dim=0)

        # Update BCM threshold: θ ← θ + meta_lr * mean_activity * (mean_activity - θ)
        mean_activity = self._post_trace.mean()
        self._bcm_threshold = self._bcm_threshold + self.meta_lr * mean_activity * (
            self._bcm_threshold - mean_activity
        )
        self._bcm_threshold = self._bcm_threshold.clamp(0.01, 2.0)

        # Compute metaplastic modulation per postsynaptic neuron
        # If activity > θ → strengthen LTD (reduce LTP)
        # If activity < θ → strengthen LTP (reduce LTD)
        # Modulation ∈ [0, 1] scales the effective learning rate
        diff = self._bcm_threshold - self._post_trace  # (n_post,)
        modulation = torch.sigmoid(diff * 10.0)  # smooth step: high when activity < θ

        # Adaptive LR: reduce for weights that changed recently
        recency_penalty = 1.0 / (1.0 + self._weight_change_history.abs().mean() * 5.0)

        # STDP-like update with BCM modulation
        ltp = torch.einsum("j,i->ij", self._post_trace, pre_spikes.mean(dim=0))
        ltd = torch.einsum("i,j->ij", self._pre_trace, post_spikes.mean(dim=0))

        # Apply per-postsynaptic modulation
        # modulation: (n_post,) → broadcast with (n_pre, n_post)
        effective_lr = self.lr * recency_penalty
        delta = effective_lr * (ltp * modulation.unsqueeze(0) - ltd * (1.0 - modulation).unsqueeze(0))

        weights = weights + delta
        weights = weights.clamp(self.w_min, self.w_max)

        # Track weight changes
        self._weight_change_history = self._weight_change_history * 0.9 + delta.abs() * 0.1

        self._step_count += 1
        return weights

    def _relocate_and_forward(
        self,
        weights: torch.Tensor,
        pre_spikes: torch.Tensor,
        post_spikes: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        n_pre, n_post = weights.shape
        self._pre_trace = self._pre_trace.to(device)
        self._post_trace = self._post_trace.to(device)
        self._weight_change_history = self._weight_change_history.to(device)
        self._activity_history = self._activity_history.to(device)
        self._bcm_threshold = self._bcm_threshold.to(device)
        return self.forward(weights, pre_spikes, post_spikes)

    # ------------------------------------------------------------------

    def get_state(self) -> Dict[str, Any]:
        base = super().get_state()
        base["bcm_threshold"] = self._bcm_threshold
        base["weight_change_history"] = self._weight_change_history
        return base

    def get_statistics(self) -> Dict[str, Any]:
        base = super().get_statistics()
        if self._bcm_threshold is not None:
            base["bcm_threshold_mean"] = self._bcm_threshold.mean().item()
        if self._weight_change_history is not None:
            base["weight_change_norm"] = self._weight_change_history.norm().item()
        base.update({
            "meta_learning_rate": self.meta_lr,
            "activity_threshold": self.activity_threshold,
        })
        return base
