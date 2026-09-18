"""Dual-Space Sparse Attention (DSSA).

Combines MoBA-style block-sparse attention with state-space compression
to achieve O(N) complexity while preserving long-range dependency capture.

Biological motivation: two parallel pathways for information routing -
  - Local pathway: block-sparse attention (receptive field approximation)
  - Global pathway: state-space compression (sustained global context)

The block-sparse path attends within local blocks plus a configurable
number of global summary tokens. The state-space path compresses the
input via a learned SSM kernel, then expands back, giving O(N) cost
for the global component.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class StateSpaceCompression(nn.Module):
    """Lightweight SSM-style compression: input -> low-rank -> causal decay -> expand.

    Complexity: O(N * d_compress) where d_compress << N.
    """

    def __init__(self, dim: int, compress_dim: int, dropout: float = 0.0):
        super().__init__()
        self.proj_down = nn.Linear(dim, compress_dim)
        self.proj_up = nn.Linear(compress_dim, dim)
        self.decay = nn.Parameter(torch.zeros(compress_dim))
        self.norm = nn.RMSNorm(compress_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, _ = x.shape
        compressed = self.proj_down(x)
        compressed = self.norm(compressed)

        # Causal cumulative decay: state[i] = sum_{j<=i} x[j] * exp(decay)^(i-j)
        decay = torch.exp(self.decay)
        positions = torch.arange(N, device=x.device, dtype=decay.dtype).unsqueeze(1)
        decay_matrix = decay.unsqueeze(0) ** positions
        accumulated = torch.cumsum(compressed * decay_matrix, dim=1)

        return self.dropout(self.proj_up(accumulated))


class DualSpaceSparseAttention(nn.Module):
    """Dual-Space Sparse Attention: block-sparse local + SSM global.

    Args:
        dim: Input and output dimension.
        num_heads: Number of attention heads.
        block_size: Size of each local attention block.
        num_global_tokens: Number of learnable summary tokens that provide
            global context to every block.
        compress_dim: Internal dimension of the SSM compression.
        dropout: Attention and residual dropout.
        causal: Whether to apply causal masking within blocks.
        qkv_bias: Whether to include bias in QKV projections.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        block_size: int = 64,
        num_global_tokens: int = 8,
        compress_dim: int = 64,
        dropout: float = 0.0,
        causal: bool = False,
        qkv_bias: bool = False,
    ):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} must be divisible by num_heads {num_heads}"

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.block_size = block_size
        self.num_global_tokens = num_global_tokens
        self.causal = causal
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, 3 * dim, bias=qkv_bias)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout_p = dropout

        # Learnable global summary tokens
        self.global_tokens = nn.Parameter(torch.randn(1, num_global_tokens, dim) * 0.02)

        # Global SSM pathway
        self.ssm = StateSpaceCompression(dim, compress_dim, dropout=dropout)

        # Gating to fuse local and global
        self.gate_proj = nn.Linear(dim * 2, dim)

    def _block_sparse_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        N: int,
    ) -> torch.Tensor:
        """Block-sparse attention: each block attends to itself + global tokens.

        Args:
            q, k, v: (B*H, N, D) — already folded batch+head dims.
            N: Sequence length.

        Returns:
            (B*H, N, D) attended output.
        """
        BH = q.shape[0]
        bs = self.block_size
        nd = self.head_dim
        n_glob = self.num_global_tokens

        # Separate global keys/values from the first n_glob positions
        glob_k = k[:, :n_glob, :]  # (BH, n_glob, D)
        glob_v = v[:, :n_glob, :]

        out = q.new_zeros(BH, N, nd)

        num_blocks = math.ceil(N / bs)
        for i in range(num_blocks):
            s = i * bs
            e = min(s + bs, N)

            q_block = q[:, s:e, :]  # (BH, block_len, D)

            # Keys/values: current block + global tokens
            k_cat = torch.cat([k[:, s:e, :], glob_k], dim=1)
            v_cat = torch.cat([v[:, s:e, :], glob_v], dim=1)

            # Attention scores
            attn = torch.bmm(q_block, k_cat.transpose(1, 2)) * self.scale  # (BH, block_len, block_len+n_glob)

            if self.causal:
                block_len = e - s
                causal_mask = torch.triu(
                    torch.ones(block_len, block_len + n_glob, device=q.device, dtype=torch.bool),
                    diagonal=block_len,
                )
                attn.masked_fill_(causal_mask.unsqueeze(0), float("-inf"))

            attn = F.softmax(attn, dim=-1)
            attn = F.dropout(attn, p=self.dropout_p, training=self.training)

            out[:, s:e, :] = torch.bmm(attn, v_cat)

        return out

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

        # --- Path 1: Block-sparse local attention ---
        qkv = self.qkv(x).reshape(B, N, 3, H, hd)
        q, k, v = qkv.unbind(2)  # each (B, N, H, hd)

        # Fold batch and head dims for the block-sparse kernel
        qf = q.permute(0, 2, 1, 3).reshape(B * H, N, hd)
        kf = k.permute(0, 2, 1, 3).reshape(B * H, N, hd)
        vf = v.permute(0, 2, 1, 3).reshape(B * H, N, hd)

        sparse_out = self._block_sparse_attention(qf, kf, vf, N)
        sparse_out = sparse_out.reshape(B, H, N, hd).permute(0, 2, 1, 3).reshape(B, N, D)

        # --- Path 2: SSM global compression ---
        ssm_out = self.ssm(x)

        # --- Gated fusion ---
        gate = torch.sigmoid(self.gate_proj(torch.cat([sparse_out, ssm_out], dim=-1)))
        fused = gate * sparse_out + (1.0 - gate) * ssm_out

        return self.out_proj(fused)

    def extra_repr(self) -> str:
        return (
            f"dim={self.dim}, heads={self.num_heads}, block_size={self.block_size}, "
            f"global_tokens={self.num_global_tokens}, causal={self.causal}"
        )
