"""Resource manager for BIO-NN.

Auto-detects hardware, estimates memory, suggests batch sizes,
and monitors usage during training.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

try:
    import psutil
except ImportError:  # pragma: no cover
    psutil = None  # type: ignore[assignment]

from ..utils.hardware import detect_device, get_hardware_info
from ..utils.memory import clear_gpu_cache, format_bytes, get_peak_memory


def _process_rss_mb() -> float:
    if psutil is None:
        return 0.0
    try:
        return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    except Exception:
        return 0.0


class HardwareInfo:
    """Snapshot of available compute resources."""

    def __init__(self) -> None:
        self.device = detect_device()
        self._raw = get_hardware_info()
        self.cpu_count = self._raw["cpu"]["count"]
        self.gpu_count = self._raw["gpu"]["count"] if self._raw["gpu"] else 0
        self.gpu_name = self._raw["gpu"]["name"] if self._raw["gpu"] else None
        self.gpu_memory_bytes = (
            self._raw["gpu"]["total_memory"] if self._raw["gpu"] else 0
        )
        self.gpu_capability = (
            self._raw["gpu"]["capability"] if self._raw["gpu"] else None
        )
        self.platform = self._raw["platform"]
        self.python = self._raw["python"]

    @property
    def has_gpu(self) -> bool:
        return self.gpu_count > 0

    @property
    def supports_bf16(self) -> bool:
        if not self.has_gpu or self.gpu_capability is None:
            return False
        # BF16 requires compute capability >= 8.0 (Ampere+)
        return self.gpu_capability[0] >= 8

    @property
    def supports_amp(self) -> bool:
        return self.has_gpu

    @property
    def gpu_memory_mb(self) -> float:
        return self.gpu_memory_bytes / (1024 * 1024)

    def summary(self) -> Dict[str, Any]:
        return {
            "device": str(self.device),
            "cpu_count": self.cpu_count,
            "gpu_count": self.gpu_count,
            "gpu_name": self.gpu_name,
            "gpu_memory_mb": round(self.gpu_memory_mb, 1),
            "supports_bf16": self.supports_bf16,
            "supports_amp": self.supports_amp,
            "platform": self.platform,
        }


def estimate_model_memory(
    model: nn.Module,
    input_shape: Tuple[int, ...],
    dtype: torch.dtype = torch.float32,
) -> Dict[str, float]:
    """Estimate memory requirements for a model.

    Args:
        model:      The model.
        input_shape: Batch-less input shape, e.g. (3, 224, 224).
        dtype:      Parameter dtype.

    Returns:
        Dict with parameter_bytes, input_bytes, gradient_bytes,
        total_estimate_mb.
    """
    param_bytes = sum(
        p.nelement() * p.element_size() for p in model.parameters()
    )
    buf_bytes = sum(
        b.nelement() * b.element_size() for b in model.buffers()
    )
    total_param = param_bytes + buf_bytes

    input_elements = 1
    for s in input_shape:
        input_elements *= s
    element_size = 2 if dtype in (torch.float16, torch.bfloat16) else 4
    input_bytes = input_elements * element_size

    # Rough estimate: gradients = params, optimizer state = 2x params (Adam)
    gradient_bytes = param_bytes
    optimizer_bytes = param_bytes * 2

    total = total_param + input_bytes + gradient_bytes + optimizer_bytes
    return {
        "parameter_bytes": total_param,
        "input_bytes": input_bytes,
        "gradient_bytes": gradient_bytes,
        "optimizer_bytes": optimizer_bytes,
        "total_estimate_mb": total / (1024 * 1024),
    }


def suggest_batch_size(
    model: nn.Module,
    input_shape: Tuple[int, ...],
    device: Optional[torch.device] = None,
    max_memory_fraction: float = 0.8,
    dtype: torch.dtype = torch.float32,
) -> Dict[str, Any]:
    """Suggest a safe batch size based on available VRAM.

    Uses a binary search heuristic to find the largest batch size
    that fits within the memory budget.

    Returns dict with suggested batch_size, estimated_memory_mb, and
    available_memory_mb.
    """
    hw = HardwareInfo()
    if device is None:
        device = hw.device

    if device.type != "cuda":
        # On CPU, suggest a reasonable default
        return {
            "suggested_batch_size": 16,
            "estimated_memory_mb": 0.0,
            "available_memory_mb": 0.0,
            "reason": "CPU-only: using default batch size 16",
        }

    total_vram = torch.cuda.get_device_properties(device).total_memory
    budget = total_vram * max_memory_fraction

    # Binary search: try batch sizes 1, 2, 4, 8, 16, 32, 64, ...
    # Fit within budget
    lo, hi = 1, 512
    best = 1

    # Quick upper bound: estimate memory per sample
    mem_est = estimate_model_memory(model, input_shape, dtype)
    per_sample_mb = mem_est["total_estimate_mb"]  # at batch=1
    if per_sample_mb > 0:
        hi = min(hi, int(budget / (1024 * 1024 * per_sample_mb)))

    # Simple linear scan (safe, fast enough for this range)
    for bs in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]:
        est = per_sample_mb * bs
        if est <= budget / (1024 * 1024):
            best = bs
        else:
            break

    return {
        "suggested_batch_size": best,
        "estimated_memory_mb": round(per_sample_mb * best, 1),
        "available_memory_mb": round(budget / (1024 * 1024), 1),
        "reason": f"Estimated {per_sample_mb:.1f} MB/sample",
    }


class ResourceMonitor:
    """Monitor resource usage during training.

    Call :meth:`sample` periodically to record snapshots.
    """

    def __init__(self, warning_threshold_fraction: float = 0.9) -> None:
        self.warning_threshold = warning_threshold_fraction
        self.snapshots: List[Dict[str, Any]] = []
        self._peak_cpu = 0.0
        self._peak_gpu = 0.0

    def sample(self, label: str = "") -> Dict[str, Any]:
        """Record a resource snapshot."""
        cpu_mb = _process_rss_mb()
        gpu_mb = 0.0
        gpu_total = 0.0

        if torch.cuda.is_available():
            gpu_mb = torch.cuda.memory_allocated() / (1024 * 1024)
            gpu_total = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)

        self._peak_cpu = max(self._peak_cpu, cpu_mb)
        self._peak_gpu = max(self._peak_gpu, gpu_mb)

        snapshot = {
            "label": label,
            "cpu_mb": round(cpu_mb, 1),
            "gpu_mb": round(gpu_mb, 1),
            "gpu_total_mb": round(gpu_total, 1),
            "gpu_percent": round(gpu_mb / max(gpu_total, 1) * 100, 1),
        }
        self.snapshots.append(snapshot)
        return snapshot

    @property
    def warning(self) -> bool:
        """True if any resource exceeded the warning threshold."""
        if not self.snapshots:
            return False
        last = self.snapshots[-1]
        if last["gpu_total_mb"] > 0:
            if last["gpu_percent"] / 100 > self.warning_threshold:
                return True
        return False

    @property
    def summary(self) -> Dict[str, Any]:
        return {
            "peak_cpu_mb": round(self._peak_cpu, 1),
            "peak_gpu_mb": round(self._peak_gpu, 1),
            "num_samples": len(self.snapshots),
            "last_snapshot": self.snapshots[-1] if self.snapshots else None,
        }

    def clear(self) -> None:
        self.snapshots.clear()
        self._peak_cpu = 0.0
        self._peak_gpu = 0.0
        clear_gpu_cache()


class ResourceManager:
    """High-level resource manager coordinating detection, estimation,
    and monitoring.

    Usage::

        rm = ResourceManager()
        print(rm.hardware.summary())
        suggestion = rm.suggest_batch_size(model, (3, 64, 64))
    """

    def __init__(self) -> None:
        self.hardware = HardwareInfo()
        self.monitor = ResourceMonitor()

    def suggest_batch_size(
        self,
        model: nn.Module,
        input_shape: Tuple[int, ...],
        dtype: torch.dtype = torch.float32,
    ) -> Dict[str, Any]:
        return suggest_batch_size(model, input_shape, self.hardware.device, dtype=dtype)

    def estimate_memory(
        self,
        model: nn.Module,
        input_shape: Tuple[int, ...],
        dtype: torch.dtype = torch.float32,
    ) -> Dict[str, float]:
        return estimate_model_memory(model, input_shape, dtype)

    def cleanup(self) -> None:
        """Clear caches and reset monitors."""
        self.monitor.clear()
        clear_gpu_cache()

    def status(self) -> Dict[str, Any]:
        """Full status report."""
        return {
            "hardware": self.hardware.summary(),
            "monitor": self.monitor.summary,
            "current_process_memory_mb": round(_process_rss_mb(), 1),
        }
