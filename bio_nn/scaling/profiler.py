"""Resource profiler for BIO-NN scaling.

Tracks FLOPs, memory, throughput, and per-component costs.
Extends the existing training.profiler for scaling-specific needs.
"""

from __future__ import annotations

import os
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

try:
    import psutil
except ImportError:  # pragma: no cover
    psutil = None  # type: ignore[assignment]


def _process_memory_mb() -> float:
    if psutil is None:
        return 0.0
    try:
        return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    except Exception:
        return 0.0


def _gpu_memory_mb(device: Optional[torch.device] = None) -> float:
    if not torch.cuda.is_available():
        return 0.0
    if device is None:
        device = torch.device("cuda")
    return torch.cuda.memory_allocated(device) / (1024 * 1024)


class FLOPsCounter:
    """Count multiply-accumulate operations via PyTorch hooks.

    Usage::

        counter = FLOPsCounter(model)
        # run forward pass
        print(counter.total_flops)
        counter.reset()
    """

    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self.total_flops = 0
        self._hooks: List[torch.Tensor] = []
        self._counters: Dict[str, int] = {}
        self._enabled = False

    def enable(self) -> None:
        if self._enabled:
            return
        self._enabled = True
        self.reset()
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                h = module.register_forward_hook(self._make_hook(name, module))
                self._hooks.append(h)

    def disable(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        self._enabled = False

    def reset(self) -> None:
        self.total_flops = 0
        self._counters.clear()

    def _make_hook(self, name: str, module: nn.Module):
        def hook(mod, inp, out):
            flops = 0
            if isinstance(mod, nn.Linear):
                batch = inp[0].shape[0] if inp[0].dim() > 1 else 1
                flops = 2 * batch * mod.in_features * mod.out_features
            elif isinstance(mod, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                batch = inp[0].shape[0]
                out_elements = out.shape[0]
                flops = 2 * out_elements * mod.in_channels
                kernel = 1
                for k in mod.kernel_size:
                    kernel *= k
                flops *= kernel
            self.total_flops += flops
            self._counters[name] = self._counters.get(name, 0) + flops
        return hook

    @property
    def per_module(self) -> Dict[str, int]:
        return dict(self._counters)

    def summary(self) -> Dict[str, Any]:
        return {
            "total_flops": self.total_flops,
            "per_module": dict(self._counters),
        }


class ThroughputTracker:
    """Measure samples/second and spikes/second during training.

    Usage::

        tracker = ThroughputTracker()
        tracker.start()
        for batch in dataloader:
            ...
            tracker.update(batch_size, total_spikes)
        stats = tracker.stop()
    """

    def __init__(self) -> None:
        self._start_time: float = 0.0
        self._samples: int = 0
        self._spikes: int = 0
        self._batches: int = 0
        self._wall_start: float = 0.0

    def start(self) -> None:
        self._start_time = time.perf_counter()
        self._wall_start = time.time()
        self._samples = 0
        self._spikes = 0
        self._batches = 0

    def update(self, batch_size: int, spikes: int = 0) -> None:
        self._samples += batch_size
        self._spikes += spikes
        self._batches += 1

    def stop(self) -> Dict[str, float]:
        elapsed = time.perf_counter() - self._start_time
        elapsed = max(elapsed, 1e-9)
        return {
            "samples_per_second": self._samples / elapsed,
            "spikes_per_second": self._spikes / elapsed,
            "batches_per_second": self._batches / elapsed,
            "total_samples": self._samples,
            "total_spikes": self._spikes,
            "total_batches": self._batches,
            "wall_time_s": time.time() - self._wall_start,
        }


class ResourceProfiler:
    """Full resource profiler combining FLOPs, memory, and throughput.

    Usage::

        profiler = ResourceProfiler(model)
        profiler.start()
        # ... training ...
        report = profiler.stop()
    """

    def __init__(
        self,
        model: Optional[nn.Module] = None,
        track_flops: bool = True,
        track_gpu: bool = True,
    ) -> None:
        self.model = model
        self.track_flops = track_flops and model is not None
        self.track_gpu = track_gpu

        self._counter = FLOPsCounter(model) if self.track_flops else None
        self._throughput = ThroughputTracker()
        self._start_mem: float = 0.0
        self._start_gpu: float = 0.0
        self._start_time: float = 0.0
        self._snapshots: List[Dict[str, float]] = []

    def start(self) -> None:
        self._start_time = time.perf_counter()
        self._start_mem = _process_memory_mb()
        self._start_gpu = _gpu_memory_mb()
        self._throughput.start()
        if self._counter:
            self._counter.enable()

    def snapshot(self, label: str = "") -> None:
        """Take an in-flight memory snapshot (useful between epochs)."""
        self._snapshots.append({
            "label": label,
            "cpu_mb": _process_memory_mb(),
            "gpu_mb": _gpu_memory_mb(),
            "flops": self._counter.total_flops if self._counter else 0,
        })

    def update(self, batch_size: int, spikes: int = 0) -> None:
        self._throughput.update(batch_size, spikes)

    def stop(self) -> Dict[str, Any]:
        self._throughput.stop()
        throughput = self._throughput.stop()
        if self._counter:
            self._counter.disable()

        elapsed = time.perf_counter() - self._start_time
        current_mem = _process_memory_mb()
        current_gpu = _gpu_memory_mb()

        report: Dict[str, Any] = {
            "wall_time_s": elapsed,
            "cpu_memory": {
                "start_mb": self._start_mem,
                "current_mb": current_mem,
                "delta_mb": current_mem - self._start_mem,
            },
            "throughput": throughput,
            "snapshots": self._snapshots,
        }

        if self.track_gpu and torch.cuda.is_available():
            report["gpu_memory"] = {
                "start_mb": self._start_gpu,
                "current_mb": current_gpu,
                "delta_mb": current_gpu - self._start_gpu,
                "peak_mb": torch.cuda.max_memory_allocated() / (1024 * 1024),
            }

        if self._counter:
            report["flops"] = self._counter.summary()

        return report

    def reset_peak_memory(self) -> None:
        """Reset CUDA peak memory stats."""
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()


def generate_report(
    profiles: List[Dict[str, Any]],
    title: str = "BIO-NN Profiling Report",
) -> str:
    """Format a list of profiling reports into a human-readable string."""
    lines = [f"{'=' * 60}", f"  {title}", f"{'=' * 60}", ""]

    for i, p in enumerate(profiles):
        lines.append(f"--- Profile {i + 1} ---")
        lines.append(f"  Wall time:      {p.get('wall_time_s', 0):.3f}s")

        cpu = p.get("cpu_memory", {})
        lines.append(f"  CPU memory:     {cpu.get('delta_mb', 0):+.1f} MB")

        gpu = p.get("gpu_memory")
        if gpu:
            lines.append(f"  GPU memory:     {gpu.get('delta_mb', 0):+.1f} MB (peak: {gpu.get('peak_mb', 0):.1f} MB)")

        tp = p.get("throughput", {})
        lines.append(f"  Throughput:     {tp.get('samples_per_second', 0):.1f} samples/s")

        flops = p.get("flops", {})
        if flops:
            total = flops.get("total_flops", 0)
            if total > 1e9:
                lines.append(f"  FLOPs:          {total / 1e9:.2f} GFLOPs")
            else:
                lines.append(f"  FLOPs:          {total / 1e6:.1f} MFLOPs")
        lines.append("")

    lines.append(f"{'=' * 60}")
    return "\n".join(lines)
