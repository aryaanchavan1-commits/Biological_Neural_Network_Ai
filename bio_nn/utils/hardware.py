"""Hardware detection utilities."""

from __future__ import annotations

import os
import platform
from typing import Any, Dict

import torch


def detect_device() -> torch.device:
    """Detect the best available device.

    Returns:
        ``torch.device("cuda")`` if CUDA is available, else CPU.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_hardware_info() -> Dict[str, Any]:
    """Gather basic hardware information.

    Returns:
        Dictionary with ``cpu``, ``gpu``, and ``ram`` keys.
    """
    info: Dict[str, Any] = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu": {
            "count": os.cpu_count(),
            "processor": platform.processor(),
        },
        "gpu": None,
    }

    if torch.cuda.is_available():
        info["gpu"] = {
            "name": torch.cuda.get_device_name(0),
            "count": torch.cuda.device_count(),
            "capability": torch.cuda.get_device_capability(0),
            "total_memory": torch.cuda.get_device_properties(0).total_memory,
        }

    return info
