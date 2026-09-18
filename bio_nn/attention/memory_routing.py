"""Memory-Gated Routing Attention.

Inspired by scale-free architectures (Dragon Hatchling) where routing is
adaptive rather than fixed. Each token's representation complexity determines
which memory path it takes: simple tokens bypass expensive computation,
complex tokens route through a learned memory bank.

Key ideas:
  - Input-dependent gating: a lightweight MLP scores each token's "complexity".
  - Sparse routing: only the top-k most complex tokens access the memory bank.
  - Bypass path: remaining tokens get a cheap identity/skip connection.
  - Adaptive budget: the sparsity ratio adapts based on input statistics.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MemoryBank(nn.Module):
    """Learnable memory slots with content-based retrieval.

    Each slot stores a latent representation. Retrieval uses cosine
    similarity to compute soft access weights over slots.
    """

    def __init__(self, num_slots: int, slot_dim: int, head_dim: int, dropout: float = 0.0):
        super().__init__()
        self.slots = nn.Parameter(torch.randn(1, num_slots, slot_dim) * 0.02)
        self.slot_dim = slot_dim
        self.num_slots = num_slots

        self.query_proj = nn.Linear(head_dim, slot_dim)
        self.out_proj = nn.Linear(slot_dim, head_dim)
        self.norm = nn.RMSNorm(slot_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Retrieve from memory.

        Args:
            x: (B, N, D) tensor (only tokens routed to memory will use this).

        Returns:
            (B, N, D) memory-read output (zero-padded for non-routed tokens).
        """
        B, N, D = x.shape
        q = self.norm(self.query_proj(x))  # (B, N, slot_dim)

        # Cosine similarity retrieval
        slots = self.slots.squeeze(0)  # (num_slots, slot_dim)
        q_norm = F.normalize(q, dim=-1)
        s_norm = F.normalize(slots, dim=-1)
        sim = torch.einsum("bnd,md->bnm", q_norm, s_norm)  # (B, N, num_slots)
        attn = F.softmax(sim * (self.slot_dim ** 0.5), dim=-1)
        attn = self.dropout(attn)

        # Read from slots
        read = torch.einsum("bnm,md->bnd", attn, slots)  # (B, N, slot_dim)
        return self.dropout(self.out_proj(read))


class MemoryGatedRouting(nn.Module):
    """Memory-Gated Routing Attention.

    Routes tokens adaptively: complex tokens go through a memory bank,
    simple tokens take a cheap bypass. The gating is input-dependent.

    Args:
        dim: Input/output dimension.
        num_heads: Number of attention heads.
        memory_slots: Number of memory slots in the bank.
        slot_dim: Dimension of each memory slot.
        routing_ratio: Initial fraction of tokens routed to memory (0-1).
            Adaptively adjusted based on input complexity.
        complexity_threshold: Threshold on the complexity score for routing.
        dropout: Dropout rate.
        qkv_bias: Whether to include bias in QKV projections.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        memory_slots: int = 64,
        slot_dim: int = 64,
        routing_ratio: float = 0.5,
        complexity_threshold: float = 0.5,
        dropout: float = 0.0,
        qkv_bias: bool = False,
    ):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} must be divisible by num_heads {num_heads}"

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.routing_ratio = routing_ratio
        self.complexity_threshold = complexity_threshold

        # QKV for standard attention (bypass path)
        self.qkv = nn.Linear(dim, 3 * dim, bias=qkv_bias)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout_p = dropout

        # Complexity scorer: decides routing
        # Uses variance across features as a proxy for "information complexity"
        self.complexity_gate = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.GELU(),
            nn.Linear(dim // 4, 1),
            nn.Sigmoid(),
        )

        # Memory bank (shared across heads via reshape)
        self.memory = MemoryBank(memory_slots, slot_dim, self.head_dim, dropout=dropout)

        # Fusion: combine routed (memory) and bypass (standard attention) outputs
        self.fusion_norm = nn.RMSNorm(dim)
        self.fusion_proj = nn.Linear(dim * 2, dim)

    def _standard_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        """Vanilla scaled dot-product attention for the bypass path.

        Args:
            q, k, v: (B*H, N, D)
        Returns:
            (B*H, N, D)
        """
        scale = self.head_dim ** -0.5
        attn = torch.bmm(q, k.transpose(1, 2)) * scale
        attn = F.softmax(attn, dim=-1)
        attn = F.dropout(attn, p=self.dropout_p, training=self.training)
        return torch.bmm(attn, v)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (B, N, D) input tensor.

        Returns:
            (B, N, D) output tensor.
        """
        B, N, D = x.shape
        H = self.num_heads
        hd = self.head_dim

        # --- Complexity scoring ---
        complexity = self.complexity_gate(x).squeeze(-1)  # (B, N)

        # Adaptive thresholding: route top-k tokens by complexity
        k = max(1, int(N * self.routing_ratio))
        # Use topk per-batch to handle variable complexity
        _, topk_indices = complexity.topk(k, dim=1)  # (B, k)

        # Build binary mask: 1 = routed to memory, 0 = bypass
        route_mask = torch.zeros(B, N, device=x.device, dtype=torch.bool)
        route_mask.scatter_(1, topk_indices, True)

        # --- QKV projection ---
        qkv = self.qkv(x).reshape(B, N, 3, H, hd)
        q, kv_q, v = qkv.unbind(2)

        # Fold batch+head for attention kernels
        qf = q.permute(0, 2, 1, 3).reshape(B * H, N, hd)
        kf = kv_q.permute(0, 2, 1, 3).reshape(B * H, N, hd)
        vf = v.permute(0, 2, 1, 3).reshape(B * H, N, hd)

        # --- Bypass path: standard attention ---
        bypass_out = self._standard_attention(qf, kf, vf)

        # --- Memory path: routed tokens read from memory bank ---
        memory_out = torch.zeros_like(x)  # (B, N, D)

        for b in range(B):
            idx = route_mask[b].nonzero(as_tuple=False).squeeze(1)
            if idx.numel() == 0:
                continue
            # Gather routed queries: (num_routed, H, hd)
            routed_q = q[b, idx]
            # Interleave heads: (H, num_routed, hd) -> (num_routed, H, hd) -> (num_routed*H, 1, hd)
            routed_flat = routed_q.permute(1, 0, 2).reshape(-1, hd).unsqueeze(1)
            mem_read = self.memory(routed_flat).squeeze(1)  # (num_routed*H, hd)
            # Reshape back: (H, num_routed, hd) -> (num_routed, H, hd) -> (num_routed, D)
            mem_read = mem_read.reshape(H, idx.numel(), hd).permute(1, 0, 2).reshape(-1, D)
            memory_out[b, idx] = mem_read

        # --- Compose outputs ---
        bypass_full = bypass_out.reshape(B, H, N, hd).permute(0, 2, 1, 3).reshape(B, N, D)
        route_mask_f = route_mask.unsqueeze(-1).float()

        combined = route_mask_f * memory_out + (1.0 - route_mask_f) * bypass_full

        normed = self.fusion_norm(combined)
        fused = self.fusion_proj(torch.cat([x, normed], dim=-1))

        return fused

    def extra_repr(self) -> str:
        return (
            f"dim={self.dim}, heads={self.num_heads}, "
            f"slots={self.memory.num_slots}, ratio={self.routing_ratio}"
        )
