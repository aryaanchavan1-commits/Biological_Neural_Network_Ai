"""Reproducibility utilities — seeding for torch, numpy, and stdlib random."""

from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np
import torch


def set_seed(seed: int, *, cuda_deterministic: bool = True) -> None:
    """Set random seeds for reproducibility.

    Args:
        seed:              The seed value.
        cuda_deterministic: If ``True`` (default), enable deterministic CUDA
                            ops.  This may hurt performance.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if cuda_deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_rng(
    seed: Optional[int] = None,
) -> tuple[random.Random, np.random.Generator]:
    """Return seeded ``random.Random`` and ``numpy.random.Generator`` instances.

    Useful when you need independent RNG streams that won't interfere with
    the global state.

    Args:
        seed: Optional seed value.

    Returns:
        ``(py_random, np_generator)`` tuple.
    """
    py_random = random.Random(seed)
    np_generator = np.random.default_rng(seed)
    return py_random, np_generator
