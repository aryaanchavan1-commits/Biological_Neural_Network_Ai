"""Static visualization functions for BIO-NN."""

from .spike_raster import plot_spike_raster
from .membrane import plot_membrane_potential
from .weights import plot_weight_heatmap, plot_weight_distribution
from .firing_rates import plot_firing_rates, plot_firing_rate_over_time
from .sparsity import plot_sparsity_over_time
from .plasticity import plot_plasticity_over_time, plot_plasticity_heatmap
from .structure import plot_connectivity_over_time, plot_network_topology
from .continual import plot_continual_accuracy, plot_forgetting_curve, plot_bwt
from .comparison import plot_baseline_comparison, plot_ablation_comparison, plot_training_curves

__all__ = [
    "plot_spike_raster",
    "plot_membrane_potential",
    "plot_weight_heatmap",
    "plot_weight_distribution",
    "plot_firing_rates",
    "plot_firing_rate_over_time",
    "plot_sparsity_over_time",
    "plot_plasticity_over_time",
    "plot_plasticity_heatmap",
    "plot_connectivity_over_time",
    "plot_network_topology",
    "plot_continual_accuracy",
    "plot_forgetting_curve",
    "plot_bwt",
    "plot_baseline_comparison",
    "plot_ablation_comparison",
    "plot_training_curves",
]
