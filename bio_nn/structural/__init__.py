"""Structural plasticity package.

Provides connection/neuron growth, pruning, and adaptive topology modules
that modify network connectivity during training.

Usage::

    from bio_nn.structural import create_structural, STRUCTURAL_REGISTRY

    topology = create_structural("adaptive", {
        "growth": {"strategy": "activity", "growth_rate": 0.02},
        "pruning": {"strategy": "magnitude", "pruning_rate": 0.01},
    })
    weights = topology.modify(weights, activity=activity, epoch=1)
"""

from __future__ import annotations

from typing import Any, Dict, Type

from .adaptive_topology import AdaptiveTopology
from .base import BaseStructuralPlasticity
from .growth import ConnectionGrowth
from .pruning import ConnectionPruning

STRUCTURAL_REGISTRY: Dict[str, Type[BaseStructuralPlasticity]] = {
    "growth": ConnectionGrowth,
    "pruning": ConnectionPruning,
    "adaptive": AdaptiveTopology,
}


def create_structural(rule_type: str, config: Dict[str, Any] = None) -> BaseStructuralPlasticity:
    """Instantiate a structural-plasticity module by name.

    Args:
        rule_type: Key into ``STRUCTURAL_REGISTRY``.
        config:    Configuration dict forwarded to the constructor.

    Returns:
        An initialised structural-plasticity module.

    Raises:
        KeyError: If *rule_type* is not registered.
    """
    if rule_type not in STRUCTURAL_REGISTRY:
        raise KeyError(
            f"Unknown structural rule '{rule_type}'. "
            f"Available: {sorted(STRUCTURAL_REGISTRY)}"
        )
    return STRUCTURAL_REGISTRY[rule_type](config)

__all__ = [
    "BaseStructuralPlasticity",
    "ConnectionGrowth",
    "ConnectionPruning",
    "AdaptiveTopology",
    "STRUCTURAL_REGISTRY",
    "create_structural",
]
