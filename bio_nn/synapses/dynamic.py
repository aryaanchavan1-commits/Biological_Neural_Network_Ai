import math

import torch
import torch.nn as nn

from .base import BaseSynapse


class DynamicSynapse(BaseSynapse):
    """Plastic synapse that maintains exponential traces for STDP-style learning.

    Tracks pre- and post-synaptic traces that decay exponentially each
    time step, enabling correlation-based plasticity rules.

    Config keys:
        tau_pre (float): Time constant for pre-synaptic trace decay. Default: 20.
        tau_post (float): Time constant for post-synaptic trace decay. Default: 20.
        w_max (float): Upper bound for weights. Default: 1.0.
        w_min (float): Lower bound for weights. Default: 0.0.
        init_weights (str): Initialisation scheme for weights. Default: "uniform".
    """

    def __init__(self, in_features: int, out_features: int, config: dict) -> None:
        super().__init__(in_features, out_features, config)

        self.tau_pre: float = config.get("tau_pre", 20.0)
        self.tau_post: float = config.get("tau_post", 20.0)
        self.w_max: float = config.get("w_max", 1.0)
        self.w_min: float = config.get("w_min", 0.0)

        # Pre-compute decay factors
        self.register_buffer(
            "_decay_pre",
            torch.tensor(math.exp(-1.0 / self.tau_pre)),
        )
        self.register_buffer(
            "_decay_post",
            torch.tensor(math.exp(-1.0 / self.tau_post)),
        )

        # Trace buffers (not parameters – updated in-place each step)
        self.register_buffer("pre_trace", torch.zeros(in_features))
        self.register_buffer("post_trace", torch.zeros(out_features))

        # Weight initialisation
        init_scheme = config.get("init_weights", "uniform")
        if init_scheme == "uniform":
            nn.init.uniform_(self.weights, self.w_min, self.w_max)
        else:
            nn.init.xavier_uniform_(self.weights)
            with torch.no_grad():
                self.weights.clamp_(self.w_min, self.w_max)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, pre_spikes: torch.Tensor) -> torch.Tensor:
        """Compute post-synaptic currents and update traces.

        Args:
            pre_spikes: (batch, in_features) binary spikes.

        Returns:
            post_input: (batch, out_features)
        """
        self._update_traces(pre_spikes)
        return pre_spikes @ self.weights.t()

    # ------------------------------------------------------------------
    # Trace management
    # ------------------------------------------------------------------

    def _update_traces(self, pre_spikes: torch.Tensor) -> None:
        """Exponential moving average of spike history."""
        # pre_spikes: (batch, in_features) → average across batch for trace
        batch_mean_pre = pre_spikes.detach().mean(dim=0)  # (in_features,)
        self.pre_trace.mul_(self._decay_pre).add_(batch_mean_pre, alpha=1.0 - self._decay_pre)

        # post_trace is updated externally via update_post_trace() after the
        # post-neuron fires; here we just decay it.
        self.post_trace.mul_(self._decay_post)

    def update_post_trace(self, post_spikes: torch.Tensor) -> None:
        """Update the post-synaptic trace with current post-neuron activity.

        Call this from the neuron layer after it computes its output.

        Args:
            post_spikes: (batch, out_features) binary spikes.
        """
        batch_mean_post = post_spikes.detach().mean(dim=0)  # (out_features,)
        self.post_trace.mul_(self._decay_post).add_(batch_mean_post, alpha=1.0 - self._decay_post)

    # ------------------------------------------------------------------
    # Weight update (called by learning rule)
    # ------------------------------------------------------------------

    def apply_stdp(self, lr: float = 1e-3) -> None:
        """Apply a basic STDP-like weight update using the current traces.

        This is a simplified Hebbian rule: Δw = lr * post_trace ⊗ pre_trace.
        Call this once per time-step after traces are updated.

        The weights are clamped to [w_min, w_max] after the update.
        """
        with torch.no_grad():
            # outer product: (out_features, in_features)
            delta = torch.outer(self.post_trace, self.pre_trace)
            self.weights.add_(delta * lr)
            self.weights.clamp_(self.w_min, self.w_max)

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset_traces(self) -> None:
        """Zero out both traces (call at the start of each sequence)."""
        self.pre_trace.zero_()
        self.post_trace.zero_()

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def get_statistics(self) -> dict:
        stats = super().get_statistics()
        stats.update({
            "tau_pre": self.tau_pre,
            "tau_post": self.tau_post,
            "w_max": self.w_max,
            "w_min": self.w_min,
            "pre_trace_mean": self.pre_trace.mean().item(),
            "post_trace_mean": self.post_trace.mean().item(),
        })
        return stats
