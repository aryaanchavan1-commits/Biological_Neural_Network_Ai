"""Neuromorphic hardware benchmarks and performance estimation.

Provides spike-based energy estimation, event-driven latency estimation,
hardware utilization metrics, and comparison with CPU/GPU baselines.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Hardware profile constants (approximate, based on published specs)
# ---------------------------------------------------------------------------

# Intel Loihi 2
LOIHI2_JOULES_PER_SPIKE = 0.1e-12  # ~0.1 pJ per synaptic event
LOIHI2_NEURONS_PER_CORE = 1024
LOIHI2_SYNAPSES_PER_CORE = 128 * 1024

# BrainScaleS (analog accelerators)
BRAINSCALES_JOULES_PER_SPIKE = 0.5e-12  # ~0.5 pJ per synaptic event
BRAINSCALES_NEURONS_PER_WAFER = 512 * 1024

# SpiNNaker2
SPINNAKER2_JOULES_PER_SPIKE = 1.0e-12  # ~1 pJ per synaptic event
SPINNAKER2_NEURONS_PER_CHIP = 128

# CPU baseline (ARM Cortex-A72, typical mobile)
CPU_JOULES_PER_MAC = 100e-12  # ~100 pJ per multiply-accumulate

# GPU baseline (NVIDIA A100, typical)
GPU_JOULES_PER_MAC = 0.5e-12  # ~0.5 pJ per MAC (amortized over large batches)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class BenchmarkResult:
    """Result of a single benchmark measurement.

    Attributes:
        metric_name: Name of the measured metric.
        value: Measured value.
        unit: Unit of measurement.
        hardware: Target hardware name.
        details: Additional detailed breakdown.
    """

    metric_name: str
    value: float
    unit: str
    hardware: str = "generic"
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HardwareProfile:
    """Characteristics of a neuromorphic hardware target.

    Attributes:
        name: Hardware identifier.
        energy_per_spike: Energy per synaptic event (Joules).
        neurons_per_core: Neurons per processing core.
        synapses_per_core: Synapses per core.
        clock_frequency_hz: Core clock frequency in Hz.
        max_spike_rate: Maximum spike rate per neuron (spikes/sec).
    """

    name: str
    energy_per_spike: float = 0.1e-12
    neurons_per_core: int = 1024
    synapses_per_core: int = 128 * 1024
    clock_frequency_hz: float = 1e6
    max_spike_rate: float = 1e3


# Pre-defined profiles
LOIHI2_PROFILE = HardwareProfile(
    name="Intel Loihi 2",
    energy_per_spike=LOIHI2_JOULES_PER_SPIKE,
    neurons_per_core=LOIHI2_NEURONS_PER_CORE,
    synapses_per_core=LOIHI2_SYNAPSES_PER_CORE,
    clock_frequency_hz=1e6,
    max_spike_rate=1e3,
)

BRAINSCALES_PROFILE = HardwareProfile(
    name="BrainScaleS",
    energy_per_spike=BRAINSCALES_JOULES_PER_SPIKE,
    neurons_per_core=128,
    synapses_per_core=64 * 1024,
    clock_frequency_hz=1e9,
    max_spike_rate=1e4,
)

SPINNAKER2_PROFILE = HardwareProfile(
    name="SpiNNaker2",
    energy_per_spike=SPINNAKER2_JOULES_PER_SPIKE,
    neurons_per_core=SPINNAKER2_NEURONS_PER_CHIP,
    synapses_per_core=64 * 1024,
    clock_frequency_hz=200e6,
    max_spike_rate=500,
)


# ---------------------------------------------------------------------------
# Energy estimation
# ---------------------------------------------------------------------------


def compute_energy_estimate(
    spike_history: Union[List[torch.Tensor], torch.Tensor],
    weights: Optional[torch.Tensor] = None,
    profile: Optional[HardwareProfile] = None,
    include_leak_energy: bool = True,
    leak_energy_per_neuron: float = 0.01e-12,
) -> BenchmarkResult:
    """Estimate energy consumption from spike activity.

    Args:
        spike_history: Spike tensors (list per timestep or stacked).
        weights: Optional weight matrix for synaptic event counting.
        profile: Hardware profile. Defaults to Loihi 2.
        include_leak_energy: Whether to include leakage current energy.
        leak_energy_per_neuron: Energy per neuron per timestep for leakage.

    Returns:
        BenchmarkResult with energy estimate in Joules.
    """
    profile = profile or LOIHI2_PROFILE

    if isinstance(spike_history, list):
        spike_tensor = torch.stack(spike_history)  # (T, batch, neurons)
    else:
        spike_tensor = spike_history

    T = spike_tensor.shape[0]
    batch_size = spike_tensor.shape[1] if spike_tensor.dim() > 1 else 1
    n_neurons = spike_tensor.numel() // max(T * batch_size, 1)

    # Total spike count
    total_spikes = spike_tensor.float().sum().item()

    # Synaptic events: each spike triggers connections to downstream neurons
    if weights is not None:
        # Count non-zero connections per post-synaptic neuron
        n_connections = (weights.abs() > 1e-6).float().sum().item()
        avg_connections = n_connections / max(weights.shape[0], 1)
    else:
        avg_connections = min(n_neurons, profile.synapses_per_core // max(n_neurons, 1))

    synaptic_events = total_spikes * avg_connections
    synaptic_energy = synaptic_events * profile.energy_per_spike

    # Leakage energy
    leak_energy = 0.0
    if include_leak_energy:
        leak_energy = n_neurons * batch_size * T * leak_energy_per_neuron

    total_energy = synaptic_energy + leak_energy

    return BenchmarkResult(
        metric_name="energy",
        value=total_energy,
        unit="joules",
        hardware=profile.name,
        details={
            "total_spikes": total_spikes,
            "synaptic_events": synaptic_events,
            "synaptic_energy": synaptic_energy,
            "leak_energy": leak_energy,
            "n_neurons": n_neurons,
            "timesteps": T,
            "batch_size": batch_size,
            "avg_connections": avg_connections,
        },
    )


# ---------------------------------------------------------------------------
# Latency estimation
# ---------------------------------------------------------------------------


def compute_latency_estimate(
    spike_history: Union[List[torch.Tensor], torch.Tensor],
    profile: Optional[HardwareProfile] = None,
    output_layer_idx: Optional[int] = None,
) -> BenchmarkResult:
    """Estimate event-driven inference latency.

    In event-driven hardware, latency is determined by when the output
    layer produces its first meaningful spike, not by a fixed clock.

    Args:
        spike_history: Spike tensors (list per timestep or stacked).
        profile: Hardware profile. Defaults to Loihi 2.
        output_layer_idx: Index of the output layer in spike_history.
                          If None, uses the last layer.

    Returns:
        BenchmarkResult with latency estimate in seconds.
    """
    profile = profile or LOIHI2_PROFILE

    if isinstance(spike_history, list):
        spike_tensor = torch.stack(spike_history)  # (T, batch, neurons)
    else:
        spike_tensor = spike_history

    T = spike_tensor.shape[0]

    # Find the first timestep with significant output activity
    if output_layer_idx is not None and spike_tensor.dim() > 2:
        layer_spikes = spike_tensor[:, output_layer_idx, :]
    else:
        layer_spikes = spike_tensor

    # Per-timestep spike counts
    step_spikes = layer_spikes.float().sum(dim=tuple(range(1, layer_spikes.dim())))

    # Threshold: first timestep where spike count exceeds 10% of neurons
    threshold = layer_spikes[0].numel() * 0.1 if T > 0 else 1.0
    first_active_step = T  # default: all steps needed
    for t in range(T):
        if step_spikes[t] > threshold:
            first_active_step = t + 1
            break

    # Hardware clock period
    clock_period = 1.0 / max(profile.clock_frequency_hz, 1.0)

    # Propagation latency: time for spike to traverse layers
    n_layers = spike_tensor.shape[1] if spike_tensor.dim() > 2 else 1
    propagation_latency = first_active_step * clock_period * n_layers

    # Queueing/buffering latency (empirical: ~5% of propagation)
    buffering_latency = propagation_latency * 0.05

    total_latency = propagation_latency + buffering_latency

    return BenchmarkResult(
        metric_name="latency",
        value=total_latency,
        unit="seconds",
        hardware=profile.name,
        details={
            "first_active_step": first_active_step,
            "total_timesteps": T,
            "propagation_latency": propagation_latency,
            "buffering_latency": buffering_latency,
            "clock_period": clock_period,
            "n_layers": n_layers,
            "step_spike_counts": step_spikes.tolist(),
        },
    )


# ---------------------------------------------------------------------------
# Hardware utilization metrics
# ---------------------------------------------------------------------------


def compute_hardware_utilization(
    spike_history: Union[List[torch.Tensor], torch.Tensor],
    weights: Optional[torch.Tensor] = None,
    profile: Optional[HardwareProfile] = None,
) -> BenchmarkResult:
    """Compute hardware utilization metrics.

    Measures core utilization, synaptic efficiency, and sparsity utilization.

    Args:
        spike_history: Spike tensors.
        weights: Optional weight matrix.
        profile: Hardware profile. Defaults to Loihi 2.

    Returns:
        BenchmarkResult with utilization metrics.
    """
    profile = profile or LOIHI2_PROFILE

    if isinstance(spike_history, list):
        spike_tensor = torch.stack(spike_history)
    else:
        spike_tensor = spike_history

    T = spike_tensor.shape[0]
    batch_size = spike_tensor.shape[1] if spike_tensor.dim() > 1 else 1
    n_neurons = spike_tensor.numel() // max(T * batch_size, 1)

    # Core utilization: fraction of cores with active neurons
    cores_needed = max(n_neurons // profile.neurons_per_core, 1)
    active_per_step = []
    for t in range(T):
        step = spike_tensor[t] if spike_tensor.dim() > 1 else spike_tensor
        n_active = (step.float().sum(dim=tuple(range(1, step.dim()))) > 0).float().mean().item()
        active_per_step.append(n_active)

    avg_active_fraction = np.mean(active_per_step) if active_per_step else 0.0
    core_utilization = min(avg_active_fraction * cores_needed / max(cores_needed, 1), 1.0)

    # Synaptic efficiency: useful synaptic events / total possible
    total_spikes = spike_tensor.float().sum().item()
    if weights is not None:
        total_synapses = weights.numel()
        active_synapses = (weights.abs() > 1e-6).float().sum().item()
        synaptic_efficiency = active_synapses / max(total_synapses, 1)
    else:
        synaptic_efficiency = avg_active_fraction

    # Sparsity utilization: benefit of event-driven over clock-driven
    sparsity = 1.0 - avg_active_fraction
    event_driven_speedup = 1.0 / max(1.0 - sparsity, 0.01)

    # Memory utilization
    if weights is not None:
        weight_bytes = weights.numel() * weights.element_size()
        memory_per_core = weight_bytes / max(cores_needed, 1)
        memory_utilization = memory_per_core / max(
            profile.synapses_per_core * 4, 1  # 4 bytes per synapse (float32)
        )
    else:
        memory_utilization = 0.0

    return BenchmarkResult(
        metric_name="utilization",
        value=core_utilization,
        unit="fraction",
        hardware=profile.name,
        details={
            "core_utilization": core_utilization,
            "synaptic_efficiency": synaptic_efficiency,
            "sparsity": sparsity,
            "event_driven_speedup": event_driven_speedup,
            "memory_utilization": memory_utilization,
            "cores_needed": cores_needed,
            "avg_active_fraction": avg_active_fraction,
            "total_spikes": total_spikes,
        },
    )


# ---------------------------------------------------------------------------
# CPU/GPU baseline comparison
# ---------------------------------------------------------------------------


def compare_with_baseline(
    model: nn.Module,
    spike_history: Union[List[torch.Tensor], torch.Tensor],
    weights: Optional[torch.Tensor] = None,
    profile: Optional[HardwareProfile] = None,
    cpu_macs_per_inference: Optional[int] = None,
    gpu_macs_per_inference: Optional[int] = None,
    batch_size: int = 1,
) -> BenchmarkResult:
    """Compare neuromorphic efficiency against CPU/GPU baselines.

    Args:
        model: The SNN model.
        spike_history: Spike tensors from inference.
        weights: Optional weight matrix.
        profile: Neuromorphic hardware profile.
        cpu_macs_per_inference: MAC operations for CPU baseline.
        gpu_macs_per_inference: MAC operations for GPU baseline.
        batch_size: Batch size for amortization.

    Returns:
        BenchmarkResult with comparison metrics.
    """
    profile = profile or LOIHI2_PROFILE

    # Neuromorphic energy
    neuro_energy = compute_energy_estimate(spike_history, weights, profile)

    # Count model operations
    total_params = sum(p.numel() for p in model.parameters() if p.dim() >= 2)

    if cpu_macs_per_inference is None:
        cpu_macs_per_inference = total_params * batch_size
    if gpu_macs_per_inference is None:
        gpu_macs_per_inference = total_params * batch_size

    cpu_energy = cpu_macs_per_inference * CPU_JOULES_PER_MAC
    gpu_energy = gpu_macs_per_inference * GPU_JOULES_PER_MAC

    # Speedup calculations
    neuro_vs_cpu = cpu_energy / max(neuro_energy.value, 1e-20)
    neuro_vs_gpu = gpu_energy / max(neuro_energy.value, 1e-20)

    # Latency comparison
    neuro_latency = compute_latency_estimate(spike_history, profile)

    # CPU/GPU latency estimates (clock-driven, fixed timesteps)
    T = (
        spike_history.shape[0]
        if isinstance(spike_history, torch.Tensor)
        else len(spike_history)
    )
    cpu_clock_ghz = 3.0
    cpu_latency = T * total_params / (cpu_clock_ghz * 1e9 * 0.1)  # ~10% utilization
    gpu_latency = T * total_params / (1e12 * 0.3)  # ~30% utilization on A100

    return BenchmarkResult(
        metric_name="comparison",
        value=neuro_vs_cpu,
        unit="speedup_vs_cpu",
        hardware=profile.name,
        details={
            "neuromorphic_energy_j": neuro_energy.value,
            "cpu_energy_j": cpu_energy,
            "gpu_energy_j": gpu_energy,
            "neuro_vs_cpu_energy": neuro_vs_cpu,
            "neuro_vs_gpu_energy": neuro_vs_gpu,
            "neuromorphic_latency_s": neuro_latency.value,
            "cpu_latency_s": cpu_latency,
            "gpu_latency_s": gpu_latency,
            "total_params": total_params,
            "timesteps": T,
        },
    )


# ---------------------------------------------------------------------------
# High-level benchmark suite
# ---------------------------------------------------------------------------


class NeuromorphicBenchmark:
    """Comprehensive benchmark suite for neuromorphic model evaluation.

    Runs energy, latency, utilization, and comparison benchmarks against
    multiple hardware profiles.

    Args:
        profiles: List of hardware profiles to benchmark against.
    """

    def __init__(
        self,
        profiles: Optional[Sequence[HardwareProfile]] = None,
    ):
        self.profiles = list(profiles or [LOIHI2_PROFILE, BRAINSCALES_PROFILE, SPINNAKER2_PROFILE])
        self._results: List[BenchmarkResult] = []

    def run(
        self,
        model: nn.Module,
        spike_history: Union[List[torch.Tensor], torch.Tensor],
        weights: Optional[torch.Tensor] = None,
        batch_size: int = 1,
    ) -> Dict[str, Dict[str, BenchmarkResult]]:
        """Run the full benchmark suite.

        Args:
            model: The SNN model.
            spike_history: Spike tensors from inference.
            weights: Optional weight matrix.
            batch_size: Batch size.

        Returns:
            Nested dict: {profile_name: {metric_name: BenchmarkResult}}.
        """
        all_results: Dict[str, Dict[str, BenchmarkResult]] = {}

        for profile in self.profiles:
            profile_results: Dict[str, BenchmarkResult] = {}

            # Energy
            energy = compute_energy_estimate(spike_history, weights, profile)
            profile_results["energy"] = energy
            self._results.append(energy)

            # Latency
            latency = compute_latency_estimate(spike_history, profile)
            profile_results["latency"] = latency
            self._results.append(latency)

            # Utilization
            utilization = compute_hardware_utilization(spike_history, weights, profile)
            profile_results["utilization"] = utilization
            self._results.append(utilization)

            # Comparison
            comparison = compare_with_baseline(
                model, spike_history, weights, profile, batch_size=batch_size
            )
            profile_results["comparison"] = comparison
            self._results.append(comparison)

            all_results[profile.name] = profile_results

        self._log_summary(all_results)
        return all_results

    def _log_summary(self, results: Dict[str, Dict[str, BenchmarkResult]]) -> None:
        """Log a human-readable summary of benchmark results."""
        logger.info("=" * 60)
        logger.info("Neuromorphic Benchmark Summary")
        logger.info("=" * 60)

        for profile_name, metrics in results.items():
            logger.info("\n--- %s ---", profile_name)

            if "energy" in metrics:
                e = metrics["energy"]
                logger.info(
                    "  Energy: %.4e %s (spikes=%d, events=%d)",
                    e.value,
                    e.unit,
                    e.details.get("total_spikes", 0),
                    e.details.get("synaptic_events", 0),
                )

            if "latency" in metrics:
                l = metrics["latency"]
                logger.info(
                    "  Latency: %.4e %s (first_active_step=%d)",
                    l.value,
                    l.unit,
                    l.details.get("first_active_step", 0),
                )

            if "utilization" in metrics:
                u = metrics["utilization"]
                logger.info(
                    "  Core util: %.2f%%, Synaptic eff: %.2f%%, Speedup: %.1fx",
                    u.details.get("core_utilization", 0) * 100,
                    u.details.get("synaptic_efficiency", 0) * 100,
                    u.details.get("event_driven_speedup", 1),
                )

            if "comparison" in metrics:
                c = metrics["comparison"]
                logger.info(
                    "  vs CPU: %.1fx energy savings, vs GPU: %.1fx energy savings",
                    c.details.get("neuro_vs_cpu_energy", 1),
                    c.details.get("neuro_vs_gpu_energy", 1),
                )

        logger.info("=" * 60)

    @property
    def results(self) -> List[BenchmarkResult]:
        """All benchmark results from the last run."""
        return list(self._results)

    def summary_table(self) -> Dict[str, Dict[str, float]]:
        """Return a flat summary table suitable for printing or export."""
        table: Dict[str, Dict[str, float]] = {}
        for r in self._results:
            key = f"{r.hardware}|{r.metric_name}"
            table[key] = {"value": r.value, **{k: v for k, v in r.details.items() if isinstance(v, (int, float))}}
        return table
