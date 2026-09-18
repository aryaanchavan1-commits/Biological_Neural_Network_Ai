"""Biological Attention Module.

Combines Dual-Space Sparse Attention (DSSA), Spiking Self-Attention,
and Memory-Gated Routing into a single configurable module.

Design principles:
  - LayerNorm-free: uses RMSNorm where normalization is needed (biological).
  - Configurable composition: choose which sub-attention mechanisms to use.
  - Residual connections follow biological signal flow (dendritic integration).
"""

from __future__ import annotations

from typing import List, Literal

import torch
import torch.nn as nn

from .dssa import DualSpaceSparseAttention
from .memory_routing import MemoryGatedRouting
from .spiking_attention import SpikingSelfAttention


class BiologicalAttention(nn.Module):
    """Composite biological attention layer.

    Stacks configurable sub-attention mechanisms in parallel or series,
    with residual connections and gating.

    Args:
        dim: Input/output dimension.
        num_heads: Number of attention heads.
        components: List of sub-modules to include. Each is one of:
            "dssa" — Dual-Space Sparse Attention
            "spiking" — Spiking Self-Attention
            "memory_routing" — Memory-Gated Routing
        composition: How to combine components:
            "parallel" — run all in parallel, gate-fuse outputs
            "serial" — run sequentially (DSSA -> spiking -> memory_routing)
        block_size: Block size for DSSA.
        global_tokens: Number of global tokens for DSSA.
        compress_dim: SSM compression dim for DSSA.
        spike_threshold: Spike threshold for spiking attention.
        tau_decay: Temporal decay for spiking attention.
        memory_slots: Number of memory slots.
        slot_dim: Dimension of memory slots.
        routing_ratio: Fraction of tokens routed to memory.
        dropout: Global dropout rate.
        causal: Whether attention is causal.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        components: List[Literal["dssa", "spiking", "memory_routing"]] | None = None,
        composition: Literal["parallel", "serial"] = "parallel",
        block_size: int = 64,
        global_tokens: int = 8,
        compress_dim: int = 64,
        spike_threshold: float = 0.5,
        tau_decay: float = 4.0,
        memory_slots: int = 64,
        slot_dim: int = 64,
        routing_ratio: float = 0.5,
        dropout: float = 0.0,
        causal: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.composition = composition
        self.component_names = components or ["dssa"]

        modules = nn.ModuleDict()

        if "dssa" in self.component_names:
            modules["dssa"] = DualSpaceSparseAttention(
                dim=dim,
                num_heads=num_heads,
                block_size=block_size,
                num_global_tokens=global_tokens,
                compress_dim=compress_dim,
                dropout=dropout,
                causal=causal,
            )

        if "spiking" in self.component_names:
            modules["spiking"] = SpikingSelfAttention(
                dim=dim,
                num_heads=num_heads,
                spike_threshold=spike_threshold,
                tau_decay=tau_decay,
                dropout=dropout,
            )

        if "memory_routing" in self.component_names:
            modules["memory_routing"] = MemoryGatedRouting(
                dim=dim,
                num_heads=num_heads,
                memory_slots=memory_slots,
                slot_dim=slot_dim,
                routing_ratio=routing_ratio,
                dropout=dropout,
            )

        self.modules_dict = modules
        self.num_components = len(modules)

        # Parallel composition: learned gate per component
        if composition == "parallel" and self.num_components > 1:
            self.component_gates = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(dim, dim // 4),
                    nn.GELU(),
                    nn.Linear(dim // 4, 1),
                    nn.Sigmoid(),
                )
                for _ in range(self.num_components)
            ])
        else:
            self.component_gates = None

        # Serial composition: layer norm between stages (RMSNorm — biological)
        if composition == "serial":
            self.inter_norms = nn.ModuleList([
                nn.RMSNorm(dim) for _ in range(self.num_components - 1)
            ])
        else:
            self.inter_norms = None

        # Final residual projection
        self.residual_scale = nn.Parameter(torch.ones(1) * 0.1)
        self.out_norm = nn.RMSNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (B, N, D) input tensor.

        Returns:
            (B, N, D) attended output.
        """
        residual = x

        if self.composition == "serial":
            h = x
            for i, name in enumerate(self.component_names):
                h = self.modules_dict[name](h)
                if i < len(self.inter_norms):
                    h = self.inter_norms[i](h)
            return self.out_norm(h + residual * self.residual_scale)

        # Parallel composition
        outputs = []
        for i, name in enumerate(self.component_names):
            out_i = self.modules_dict[name](x)
            if self.component_gates is not None:
                gate = self.component_gates[i](x)  # (B, N, 1)
                out_i = gate * out_i
            outputs.append(out_i)

        if len(outputs) == 1:
            combined = outputs[0]
        else:
            combined = sum(outputs)

        return self.out_norm(combined + residual * self.residual_scale)

    def extra_repr(self) -> str:
        return (
            f"dim={self.dim}, composition={self.composition}, "
            f"components={self.component_names}"
        )
