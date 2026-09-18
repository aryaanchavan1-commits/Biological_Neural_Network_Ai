"""Reward-modulated STDP (R-STDP / three-factor learning).

    Δw = eligibility_trace * reward_signal

The eligibility trace is a trace-based STDP quantity that decays with
``tau_eligibility``.  The global reward signal is supplied via the
``reward`` keyword argument each step.

Reference: Frémaux & Gerstner (2016), "Neuromodulated Spike-Timing-
Dependent Plasticity".
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch

from bio_nn.plasticity.base import BasePlasticity


class RewardSTDPPlasticity(BasePlasticity):
    """Three-factor R-STDP with decaying eligibility trace."""

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)
        self.lr: float = config.get("learning_rate", 0.001)
        self.tau_pre: float = config.get("tau_pre", 20.0)
        self.tau_post: float = config.get("tau_post", 20.0)
        self.tau_elig: float = config.get("tau_eligibility", 50.0)
        self.w_max: float = config.get("w_max", 1.0)
        self.w_min: float = config.get("w_min", 0.0)

        # Persistent traces
        self._pre_trace: Optional[torch.Tensor] = None
        self._post_trace: Optional[torch.Tensor] = None
        self._elig_trace: Optional[torch.Tensor] = None

    def _init_traces(self, n_pre: int, n_post: int, device: torch.device) -> None:
        self._pre_trace = torch.zeros(n_pre, device=device)
        self._post_trace = torch.zeros(n_post, device=device)
        self._elig_trace = torch.zeros(n_pre, n_post, device=device)

    def reset_state(self) -> None:
        self._pre_trace = None
        self._post_trace = None
        self._elig_trace = None

    # ------------------------------------------------------------------

    def forward(
        self,
        weights: torch.Tensor,
        pre_spikes: torch.Tensor,
        post_spikes: torch.Tensor,
        *,
        reward: float = 0.0,
        **kwargs: Any,
    ) -> torch.Tensor:
        n_pre, n_post = weights.shape
        device = weights.device

        if self._pre_trace is None:
            self._init_traces(n_pre, n_post, device)

        if self._pre_trace.device != device:
            self._pre_trace = self._pre_trace.to(device)
            self._post_trace = self._post_trace.to(device)
            self._elig_trace = self._elig_trace.to(device)

        decay_pre = torch.exp(torch.tensor(-1.0 / self.tau_pre, device=device))
        decay_post = torch.exp(torch.tensor(-1.0 / self.tau_post, device=device))
        decay_elig = torch.exp(torch.tensor(-1.0 / self.tau_elig, device=device))

        # Decay traces
        self._pre_trace = self._pre_trace * decay_pre
        self._post_trace = self._post_trace * decay_post
        self._elig_trace = self._elig_trace * decay_elig

        # Accumulate spikes (batch-averaged)
        pre_avg = pre_spikes.mean(dim=0)   # (n_pre,)
        post_avg = post_spikes.mean(dim=0)  # (n_post,)

        self._pre_trace = self._pre_trace + pre_avg
        self._post_trace = self._post_trace + post_avg

        # STDP eligibility update:
        #   LTP:  post_trace ⊗ pre_spikes   (pre before post)
        #   LTD:  pre_trace ⊗ post_spikes   (post before pre)
        stdp_ltp = torch.einsum("j,i->ij", self._post_trace, pre_avg)
        stdp_ltd = torch.einsum("i,j->ij", self._pre_trace, post_avg)

        self._elig_trace = self._elig_trace + (stdp_ltp - stdp_ltd)

        # Weight update: eligibility * reward
        delta = self.lr * self._elig_trace * reward

        weights = weights + delta
        weights = weights.clamp(self.w_min, self.w_max)
        self._step_count += 1
        return weights

    # ------------------------------------------------------------------

    def get_state(self) -> Dict[str, Any]:
        base = super().get_state()
        base["pre_trace"] = self._pre_trace
        base["post_trace"] = self._post_trace
        base["elig_trace"] = self._elig_trace
        return base

    def get_statistics(self) -> Dict[str, Any]:
        base = super().get_statistics()
        base.update({
            "tau_pre": self.tau_pre,
            "tau_post": self.tau_post,
            "tau_eligibility": self.tau_elig,
        })
        if self._elig_trace is not None:
            base["elig_trace_norm"] = self._elig_trace.norm().item()
        return base
