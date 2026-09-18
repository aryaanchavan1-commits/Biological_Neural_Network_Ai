"""Biologically-inspired attention mechanisms for the BIO-NN framework.

Provides three core attention variants and a composite module:

- DualSpaceSparseAttention (DSSA): MoBA block-sparse + SSM compression
- SpikingSelfAttention: spike-timing-driven attention
- MemoryGatedRouting: adaptive routing through learned memory slots
- BiologicalAttention: configurable composite of the above

Usage::

    from bio_nn.attention import DualSpaceSparseAttention, BiologicalAttention

    attn = DualSpaceSparseAttention(dim=512, num_heads=8, block_size=64)
    out = attn(x)  # (B, N, 512)

    bio = BiologicalAttention(
        dim=512, num_heads=8,
        components=["dssa", "spiking", "memory_routing"],
        composition="parallel",
    )
    out = bio(x)
"""

from .bio_attention import BiologicalAttention
from .dssa import DualSpaceSparseAttention, StateSpaceCompression
from .memory_routing import MemoryBank, MemoryGatedRouting
from .spiking_attention import SpikingSelfAttention

__all__ = [
    "DualSpaceSparseAttention",
    "StateSpaceCompression",
    "SpikingSelfAttention",
    "MemoryBank",
    "MemoryGatedRouting",
    "BiologicalAttention",
]
