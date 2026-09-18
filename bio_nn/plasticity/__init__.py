"""Plasticity rules registry and factory."""

from __future__ import annotations

from typing import Any, Dict, Type

from bio_nn.plasticity.base import BasePlasticity
from bio_nn.plasticity.hebbian import HebbianPlasticity
from bio_nn.plasticity.stdp import STDPPlasticity
from bio_nn.plasticity.reward_stdp import RewardSTDPPlasticity
from bio_nn.plasticity.homeostatic import HomeostaticPlasticity
from bio_nn.plasticity.metaplasticity import MetaplasticityPlasticity

PLASTICITY_REGISTRY: Dict[str, Type[BasePlasticity]] = {
    "hebbian": HebbianPlasticity,
    "stdp": STDPPlasticity,
    "reward_stdp": RewardSTDPPlasticity,
    "homeostatic": HomeostaticPlasticity,
    "metaplasticity": MetaplasticityPlasticity,
}


def create_plasticity(rule_type: str, config: Dict[str, Any]) -> BasePlasticity:
    """Instantiate a plasticity rule by name.

    Args:
        rule_type: Key in ``PLASTICITY_REGISTRY``.
        config:    Rule-specific configuration dictionary.

    Returns:
        An initialised ``BasePlasticity`` instance.

    Raises:
        ValueError: If ``rule_type`` is not registered.
    """
    if rule_type not in PLASTICITY_REGISTRY:
        raise ValueError(
            f"Unknown plasticity rule '{rule_type}'. "
            f"Available: {list(PLASTICITY_REGISTRY.keys())}"
        )
    return PLASTICITY_REGISTRY[rule_type](config)
