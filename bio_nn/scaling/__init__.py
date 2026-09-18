"""Scaling infrastructure for BIO-NN.

Provides mixed precision training, distributed training support,
checkpoint management, resource profiling, and hardware resource management.
"""

from .mixed_precision import MixedPrecisionTrainer
from .distributed import (
    GradientAccumulator,
    cleanup_distributed,
    create_distributed_sampler,
    get_rank,
    get_world_size,
    is_distributed,
    is_main_process,
    setup_distributed,
    wrap_model,
)
from .checkpointing import CheckpointManager
from .profiler import (
    FLOPsCounter,
    ResourceProfiler,
    ThroughputTracker,
    generate_report,
)
from .resource_manager import (
    HardwareInfo,
    ResourceManager,
    ResourceMonitor,
    estimate_model_memory,
    suggest_batch_size,
)

__all__ = [
    # Mixed precision
    "MixedPrecisionTrainer",
    # Distributed
    "GradientAccumulator",
    "cleanup_distributed",
    "create_distributed_sampler",
    "get_rank",
    "get_world_size",
    "is_distributed",
    "is_main_process",
    "setup_distributed",
    "wrap_model",
    # Checkpointing
    "CheckpointManager",
    # Profiling
    "FLOPsCounter",
    "ResourceProfiler",
    "ThroughputTracker",
    "generate_report",
    # Resources
    "HardwareInfo",
    "ResourceManager",
    "ResourceMonitor",
    "estimate_model_memory",
    "suggest_batch_size",
]
