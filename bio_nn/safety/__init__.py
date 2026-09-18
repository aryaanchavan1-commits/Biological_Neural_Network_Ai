"""Safety and alignment verification for biological neural networks.

This module provides tools for interpreting, controlling, and monitoring
spiking neural networks to ensure they behave as intended. It covers
network interpretability, controllability analysis, alignment verification,
and runtime safety monitoring.

Typical usage::

    from bio_nn.safety import (
        InterpretabilityAnalyzer,
        ControllabilityAnalyzer,
        AlignmentVerifier,
        SafetyMonitor,
    )

    # After training a SNN model...
    interp = InterpretabilityAnalyzer(model)
    acts = interp.activation_analysis(test_input)
    concepts = interp.concept_discovery(hidden_spikes)

    ctrl = ControllabilityAnalyzer(model)
    bounds = ctrl.estimate_safety_bounds(network_params)
    shutdown = ctrl.detect_emergency_shutdown(obs_sequence)

    align = AlignmentVerifier(model)
    metrics = align.calibration_analysis(test_loader)
    drift = align.distribution_shift_detect(train_stats, test_stats)

    monitor = SafetyMonitor(config={"anomaly_threshold": 3.0})
    alerts = monitor.checkimestep(activation_snapshot)
"""

from .alignment import AlignmentVerifier
from .controllability import ControllabilityAnalyzer
from .interpretability import InterpretabilityAnalyzer
from .monitoring import SafetyMonitor

__all__ = [
    "InterpretabilityAnalyzer",
    "ControllabilityAnalyzer",
    "AlignmentVerifier",
    "SafetyMonitor",
]
