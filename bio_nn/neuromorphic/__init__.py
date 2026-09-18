"""Neuromorphic hardware deployment and optimization for BIO-NN models.

Provides export, quantization, hardware-aware optimization, and benchmarking
for deploying spiking neural networks on neuromorphic chips (Intel Loihi,
BrainScaleS, SpiNNaker2) and conventional hardware.
"""

from .benchmarks import (
    NeuromorphicBenchmark,
    compute_energy_estimate,
    compute_latency_estimate,
    compute_hardware_utilization,
    compare_with_baseline,
)
from .exporter import (
    NeuromorphicExporter,
    export_snntorch,
    export_onnx_temporal,
    export_lava,
    export_pynn,
)
from .optimizer import (
    HardwareAwareOptimizer,
    EnergyAwarePenalty,
    LatencyPenalty,
    ThroughputPenalty,
)
from .quantizer import (
    WeightQuantizer,
    quantize_int,
    quantize_ternary,
    quantize_binary,
    post_training_quantize,
    quantization_aware_hook,
    accuracy_efficiency_tradeoff,
)

__all__ = [
    # Benchmarks
    "NeuromorphicBenchmark",
    "compute_energy_estimate",
    "compute_latency_estimate",
    "compute_hardware_utilization",
    "compare_with_baseline",
    # Exporter
    "NeuromorphicExporter",
    "export_snntorch",
    "export_onnx_temporal",
    "export_lava",
    "export_pynn",
    # Optimizer
    "HardwareAwareOptimizer",
    "EnergyAwarePenalty",
    "LatencyPenalty",
    "ThroughputPenalty",
    # Quantizer
    "WeightQuantizer",
    "quantize_int",
    "quantize_ternary",
    "quantize_binary",
    "post_training_quantize",
    "quantization_aware_hook",
    "accuracy_efficiency_tradeoff",
]
