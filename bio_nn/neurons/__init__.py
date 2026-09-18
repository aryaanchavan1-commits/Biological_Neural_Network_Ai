"""Neuron model implementations for the BIO-NN framework.

Provides a unified interface for different spiking neuron models,
each implementing the BaseNeuron abstract class. Use create_neuron()
for dynamic instantiation based on configuration.
"""

from typing import Any, Dict, Type

from .base import BaseNeuron
from .lif import LIFNeuron
from .adaptive_lif import AdaptiveLIFNeuron
from .izhikevich import IzhikevichNeuron, IZHIKEVICH_PRESETS
from .dual_lif import DualLIFNeuron
from .spiking_brain import SpikingBrainNeuron
from .resonate_fire import ResonateAndFireNeuron
from .adex import AdExNeuron

NEURON_REGISTRY: Dict[str, Type[BaseNeuron]] = {
    "lif": LIFNeuron,
    "adaptive_lif": AdaptiveLIFNeuron,
    "izhikevich": IzhikevichNeuron,
    "dual_lif": DualLIFNeuron,
    "spiking_brain": SpikingBrainNeuron,
    "resonate_fire": ResonateAndFireNeuron,
    "adex": AdExNeuron,
}


def create_neuron(model_type: str, size: int, config: Dict[str, Any] = None) -> BaseNeuron:
    """Factory function to create a neuron model by name.

    Args:
        model_type: Key in NEURON_REGISTRY (e.g., "lif", "adaptive_lif", "izhikevich").
        size: Number of neurons.
        config: Model-specific configuration dictionary.

    Returns:
        Instantiated BaseNeuron subclass.

    Raises:
        ValueError: If model_type is not in the registry.
    """
    if model_type not in NEURON_REGISTRY:
        available = ", ".join(NEURON_REGISTRY.keys())
        raise ValueError(
            f"Unknown neuron model '{model_type}'. "
            f"Available models: {available}"
        )

    return NEURON_REGISTRY[model_type](size=size, config=config or {})


__all__ = [
    "BaseNeuron",
    "LIFNeuron",
    "AdaptiveLIFNeuron",
    "IzhikevichNeuron",
    "IZHIKEVICH_PRESETS",
    "DualLIFNeuron",
    "SpikingBrainNeuron",
    "ResonateAndFireNeuron",
    "AdExNeuron",
    "NEURON_REGISTRY",
    "create_neuron",
]
