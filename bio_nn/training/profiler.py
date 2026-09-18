"""Performance profiler for SNN models.

Measures wall-clock time, CPU time, peak memory usage, and
per-forward / per-training-step costs.
"""

from __future__ import annotations

import time
import os
from typing import Any, Dict, Optional

import torch
import psutil


def _get_peak_memory_mb() -> float:
    """Return current process peak memory in MB."""
    try:
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        # On Windows RSS is the best available proxy; peak may differ
        return mem_info.rss / (1024 * 1024)
    except Exception:
        return 0.0


class Profiler:
    """Lightweight wall-clock + memory profiler.

    Usage::

        p = Profiler()
        p.start()
        # ... do work ...
        stats = p.stop()
        print(stats)
    """

    def __init__(self) -> None:
        self._start_time: float = 0.0
        self._start_cpu: float = 0.0
        self._start_mem: float = 0.0

    def start(self) -> None:
        """Mark the start point."""
        self._start_time = time.perf_counter()
        self._start_cpu = time.process_time()
        self._start_mem = _get_peak_memory_mb()

    def stop(self) -> Dict[str, float]:
        """Return elapsed wall time, CPU time, and peak memory delta (MB)."""
        wall = time.perf_counter() - self._start_time
        cpu = time.process_time() - self._start_cpu
        current_peak = _get_peak_memory_mb()
        return {
            "wall_time": wall,
            "cpu_time": cpu,
            "peak_memory_mb": current_peak,
            "memory_delta_mb": current_peak - self._start_mem,
        }

    # ------------------------------------------------------------------
    # Convenience wrappers
    # ------------------------------------------------------------------

    @staticmethod
    def profile_forward(
        model: torch.nn.Module,
        input_tensor: torch.Tensor,
        num_repeats: int = 10,
    ) -> Dict[str, float]:
        """Profile a single forward pass (averaged over ``num_repeats``).

        Returns:
            avg_time_ms, std_time_ms, peak_memory_mb.
        """
        model.eval()
        device = next(model.parameters()).device
        data = input_tensor.to(device)

        # Warm-up
        with torch.no_grad():
            for _ in range(3):
                model(data)

        times: list = []
        for _ in range(num_repeats):
            if torch.cuda.is_available() and device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                model(data)
            if torch.cuda.is_available() and device.type == "cuda":
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)

        import numpy as np
        arr = np.array(times) * 1000  # ms
        return {
            "avg_time_ms": float(arr.mean()),
            "std_time_ms": float(arr.std()),
            "peak_memory_mb": _get_peak_memory_mb(),
        }

    @staticmethod
    def profile_training(
        model: torch.nn.Module,
        dataloader,
        epochs: int = 1,
        device: Optional[torch.device] = None,
        loss_fn=None,
    ) -> Dict[str, float]:
        """Profile ``epochs`` of training, returning aggregate timing.

        Returns:
            avg_epoch_time_s, total_time_s, peak_memory_mb, samples_per_second.
        """
        if device is None:
            device = next(model.parameters()).device
        if loss_fn is None:
            loss_fn = torch.nn.CrossEntropyLoss()

        import torch.nn as nn

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        p = Profiler()
        p.start()

        total_samples = 0
        for epoch in range(epochs):
            model.train()
            for data, targets in dataloader:
                data, targets = data.to(device), targets.to(device)
                optimizer.zero_grad()
                out = model(data)
                if isinstance(out, dict):
                    output = out.get("output", out.get("spikes"))
                else:
                    output = out
                loss = loss_fn(output, targets)
                loss.backward()
                optimizer.step()
                total_samples += data.size(0)

        stats = p.stop()
        stats["avg_epoch_time_s"] = stats["wall_time"] / max(epochs, 1)
        stats["total_time_s"] = stats["wall_time"]
        stats["samples_per_second"] = total_samples / max(stats["wall_time"], 1e-9)
        return stats
