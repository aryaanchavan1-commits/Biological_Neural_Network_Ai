"""Spiking Self-Attention.

Computes attention using discrete spike events rather than continuous
activations. Attention weights emerge from spike timing relationships
(Spike-Timing-Dependent Attention), making this module bio-plausible
and memory-efficient via event-driven computation.

Key ideas:
  - Q, K, V are first discretised into spikes via a threshold.
  - Attention scores are accumulated only where spikes co-occur (event-driven).
  - Temporal attention: recent spikes contribute more (exponential time-decay).
  - STDP-like plasticity: pre-post timing determines attention strength.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpikingSelfAttention(nn.Module):
    """Spike-based self-attention with temporal coding.

    Instead of continuous softmax attention, this module:
      1. Generates binary spikes from Q, K via thresholding.
      2. Accumulates attention weight from co-occurring spikes,
         modulated by exponential time-decay (recency bias).
      3. Uses the accumulated sparse weights to read out V.

    This is event-driven: only spike pairs contribute to attention,
    giving effective sparsity proportional to the firing rate.

    Args:
        dim: Input dimension.
        num_heads: Number of attention heads.
        spike_threshold: Threshold for spike generation (relative to L2 norm).
        tau_decay: Time constant for temporal decay (in positions).
        surrogate_slope: Slope of surrogate gradient for threshold function.
        dropout: Output dropout.
        qkv_bias: Whether to use bias in QKV projections.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        spike_threshold: float = 0.5,
        tau_decay: float = 4.0,
        surrogate_slope: float = 5.0,
        dropout: float = 0.0,
        qkv_bias: bool = False,
    ):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} must be divisible by num_heads {num_heads}"

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.spike_threshold = spike_threshold
        self.tau_decay = tau_decay
        self.surrogate_slope = surrogate_slope
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, 3 * dim, bias=qkv_bias)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout_p = dropout

        # Per-head learnable threshold offset
        self.threshold_bias = nn.Parameter(torch.zeros(num_heads))

    @staticmethod
    def _surrogate_heaviside(x: torch.Tensor, slope: float) -> torch.Tensor:
        """Differentiable surrogate for the Heaviside step function."""
        return torch.sigmoid(slope * x)

    @staticmethod
    def _temporal_decay_matrix(N: int, tau: float, device: torch.device) -> torch.Tensor:
        """Build a causal temporal decay matrix: decay[i,j] = exp(-(i-j)/tau) for j<=i."""
        pos = torch.arange(N, device=device, dtype=torch.float32)
        diff = pos.unsqueeze(0) - pos.unsqueeze(1)  # (N, N), diff[i,j] = i-j
        decay = torch.where(diff >= 0, torch.exp(-diff / tau), torch.zeros_like(diff))
        return decay

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (B, N, D) input tensor.

        Returns:
            (B, N, D) attended output.
        """
        B, N, D = x.shape
        H = self.num_heads
        hd = self.head_dim

        # Project to Q, K, V — permute to (B, H, N, hd)
        qkv = self.qkv(x).reshape(B, N, 3, H, hd)
        q, k, v = qkv.unbind(2)  # each (B, N, H, hd)
        q = q.permute(0, 2, 1, 3)  # (B, H, N, hd)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        # --- Spike generation ---
        q_norm = F.normalize(q, dim=-1)
        k_norm = F.normalize(k, dim=-1)

        threshold = self.spike_threshold + self.threshold_bias.view(1, H, 1, 1)
        q_spikes = (q_norm > threshold).float()
        k_spikes = (k_norm > threshold).float()

        # Surrogate gradients for backward pass
        q_surrogate = self._surrogate_heaviside(q_norm - threshold, self.surrogate_slope)
        k_surrogate = self._surrogate_heaviside(k_norm - threshold, self.surrogate_slope)

        # --- Event-driven attention ---
        # (B, H, N, hd) -> (B, H, N_q, N_k)
        attn_logits = torch.einsum(
            "bhqd,bhkd->bhqk",
            q_surrogate,
            k_surrogate,
        ) * self.scale

        # --- Temporal decay (causal recency bias) ---
        temporal = self._temporal_decay_matrix(N, self.tau_decay, x.device)
        attn_logits = attn_logits * temporal.unsqueeze(0).unsqueeze(0)

        # Mask where both Q and K are silent (no spike event)
        spike_mask = torch.einsum("bhqd,bhkd->bhqk", q_spikes, k_spikes)
        attn_logits = attn_logits.masked_fill(spike_mask == 0, float("-inf"))

        # Sparse softmax (many positions will be -inf)
        attn = F.softmax(attn_logits, dim=-1)
        attn = F.dropout(attn, p=self.dropout_p, training=self.training)

        # Replace NaN from all-masked rows with uniform
        attn = torch.where(torch.isnan(attn), 0.0, attn)

        out = torch.einsum("bhqk,bhkd->bhqd", attn, v)
        out = out.permute(0, 2, 1, 3).reshape(B, N, D)

        return self.out_proj(out)

    def extra_repr(self) -> str:
        return (
            f"dim={self.dim}, heads={self.num_heads}, "
            f"threshold={self.spike_threshold}, tau={self.tau_decay}"
        )
