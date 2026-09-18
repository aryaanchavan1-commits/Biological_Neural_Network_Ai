"""Memory-profiling utilities (CPU and CUDA)."""

from __future__ import annotations

import gc
from typing import Optional

import torch


def get_peak_memory(device: Optional[torch.device] = None) -> int:
    """Return peak GPU memory allocated (bytes) via ``torch.cuda``.

    Args:
        device: CUDA device to query.  Defaults to the current device.

    Returns:
        Peak memory in bytes.  Returns ``0`` when CUDA is unavailable.
    """
    if not torch.cuda.is_available():
        return 0
    if device is None:
        device = torch.device("cuda")
    return torch.cuda.max_memory_allocated(device)


def get_current_memory(device: Optional[torch.device] = None) -> int:
    """Return current GPU memory allocated (bytes).

    Args:
        device: CUDA device to query.  Defaults to the current device.

    Returns:
        Current allocated memory in bytes.  Returns ``0`` when CUDA is
        unavailable.
    """
    if not torch.cuda.is_available():
        return 0
    if device is None:
        device = torch.device("cuda")
    return torch.cuda.memory_allocated(device)


def format_bytes(n: int) -> str:
    """Convert *n* bytes into a human-readable string.

    >>> format_bytes(1536)
    '1.50 KB'
    """
    if n < 0:
        raise ValueError(f"n must be non-negative, got {n}")
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(n)
    for unit in units:
        if abs(value) < 1024.0:
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{value:.2f} PB"


def clear_gpu_cache() -> None:
    """Force garbage collection and clear the CUDA cache."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
