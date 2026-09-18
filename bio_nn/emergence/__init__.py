"""Emergent behavior detection for BIO-NN.

Modules for detecting and analyzing emergent phenomena in spiking neural
networks, including criticality, functional specialization, memory routing,
complexity, and real-time monitoring.
"""

from .criticality import (
    CriticalityReport,
    CriticalityDetector,
    analyze_criticality,
    compute_branching_ratio,
    compute_susceptibility,
    compute_order_parameter,
    estimate_correlation_length,
)
from .complexity import (
    ComplexityReport,
    analyze_complexity,
    neural_complexity,
    compute_metastability,
    compute_entropy_rate,
    estimate_lyapunov_exponents,
    compute_fisher_information,
    compute_information_capacity,
)
from .functional_zones import (
    FunctionalZoneReport,
    analyze_functional_zones,
    cluster_neurons,
    compute_pairwise_correlation,
    compute_mutual_information,
    compute_transfer_entropy,
    compute_modularity,
    compute_participation_ratio,
)
from .memory_routing import (
    RoutingReport,
    analyze_memory_routing,
    compute_information_flow,
    compute_adaptive_index,
    compute_path_entropy,
    compute_consolidation_ratio,
    compute_bottleneck_scores,
    compute_capacity_growth,
    detect_rerouting_events,
    compute_path_stability,
)
from .monitor import (
    AlertConfig,
    EmergenceSnapshot,
    EmergenceState,
    EmergenceMonitor,
)

__all__ = [
    # Criticality
    "CriticalityReport",
    "CriticalityDetector",
    "analyze_criticality",
    "compute_branching_ratio",
    "compute_susceptibility",
    "compute_order_parameter",
    "estimate_correlation_length",
    # Complexity
    "ComplexityReport",
    "analyze_complexity",
    "neural_complexity",
    "compute_metastability",
    "compute_entropy_rate",
    "estimate_lyapunov_exponents",
    "compute_fisher_information",
    "compute_information_capacity",
    # Functional zones
    "FunctionalZoneReport",
    "analyze_functional_zones",
    "cluster_neurons",
    "compute_pairwise_correlation",
    "compute_mutual_information",
    "compute_transfer_entropy",
    "compute_modularity",
    "compute_participation_ratio",
    # Memory routing
    "RoutingReport",
    "analyze_memory_routing",
    "compute_information_flow",
    "compute_adaptive_index",
    "compute_path_entropy",
    "compute_consolidation_ratio",
    "compute_bottleneck_scores",
    "compute_capacity_growth",
    "detect_rerouting_events",
    "compute_path_stability",
    # Monitor
    "AlertConfig",
    "EmergenceSnapshot",
    "EmergenceState",
    "EmergenceMonitor",
]
