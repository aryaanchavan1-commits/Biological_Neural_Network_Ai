"""Spike-Timing-Dependent Plasticity (trace-based).

For each synapse i → j:
  pre before post (Δt > 0)  →  LTP  (Δw > 0)
  post before pre (Δt < 0)  →  LTD  (Δw < 0)

Trace-based implementation (Gerstner & Kistler 2002):
  On pre-spike:  pre_trace  += 1
  On post-spike: post_trace += 1
  Both traces decay exponentially:  trace *= exp(-1/tau)

Weight update:
  Δw = η * (post_trace ⊗ pre_spikes − pre_trace ⊗ post_spikes)

This works across arbitrary batch dimensions because the outer products
are computed via ``torch.einsum``.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch

from bio_nn.plasticity.base import BasePlasticity


class STDPPlasticity(BasePlasticity):
    """Vanilla STDP with exponential eligibility traces."""

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)
        self.lr: float = config.get("learning_rate", 0.005)
        self.tau_pre: float = config.get("tau_pre", 20.0)
        self.tau_post: float = config.get("tau_post", 20.0)
        self.w_max: float = config.get("w_max", 1.0)
        self.w_min: float = config.get("w_min", 0.0)
        self.A_plus: float = config.get("A_plus", 1.0)
        self.A_minus: float = config.get("A_minus", 1.0)

        # Decay factors (per time-step)
        self.decay_pre: float = 0.0
        self.decay_post: float = 0.0

        # Persistent traces (initialised lazily on first forward)
        self._pre_trace: Optional[torch.Tensor] = None
        self._post_trace: Optional[torch.Tensor] = None

    def _init_traces(self, n_pre: int, n_post: int, device: torch.device) -> None:
        self._pre_trace = torch.zeros(n_pre, device=device)
        self._post_trace = torch.zeros(n_post, device=device)

    def reset_state(self) -> None:
        self._pre_trace = None
        self._post_trace = None

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

        # Lazy init
        if self._pre_trace is None:
            self._init_traces(n_pre, n_post, device)

        # Ensure traces live on the right device
        if self._pre_trace.device != device:
            self._pre_trace = self._pre_trace.to(device)
            self._post_trace = self._post_trace.to(device)

        # Compute decay factors from time constants
        # (tau is in units of time-steps)
        decay_pre = torch.exp(torch.tensor(-1.0 / self.tau_pre, device=device))
        decay_post = torch.exp(torch.tensor(-1.0 / self.tau_post, device=device))

        # Decay traces
        self._pre_trace = self._pre_trace * decay_pre
        self._post_trace = self._post_trace * decay_post

        # Accumulate spikes into traces (averaged over batch)
        pre_spikes_avg = pre_spikes.mean(dim=0)   # (n_pre,)
        post_spikes_avg = post_spikes.mean(dim=0)  # (n_post,)

        self._pre_trace = self._pre_trace + pre_spikes_avg
        self._post_trace = self._post_trace + post_spikes_avg

        # Weight update via trace-based STDP
        # LTP:  post_trace[j] * pre_spikes[i]   →  pre before post
        # LTD:  pre_trace[i] * post_spikes[j]   →  post before pre
        ltp = self.A_plus * torch.einsum("j,i->ij", self._post_trace, pre_spikes_avg)
        ltd = self.A_minus * torch.einsum("i,j->ij", self._pre_trace, post_spikes_avg)

        delta = self.lr * (ltp - ltd)

        weights = weights + delta
        weights = weights.clamp(self.w_min, self.w_max)
        self._step_count += 1
        return weights

    # ------------------------------------------------------------------

    def get_state(self) -> Dict[str, Any]:
        base = super().get_state()
        base["pre_trace"] = self._pre_trace
        base["post_trace"] = self._post_trace
        return base

    def get_statistics(self) -> Dict[str, Any]:
        base = super().get_statistics()
        base.update({
            "tau_pre": self.tau_pre,
            "tau_post": self.tau_post,
            "A_plus": self.A_plus,
            "A_minus": self.A_minus,
        })
        if self._pre_trace is not None:
            base["pre_trace_mean"] = self._pre_trace.mean().item()
            base["post_trace_mean"] = self._post_trace.mean().item()
        return base
