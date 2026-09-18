"""Topology package — network connectivity pattern generators.

Provides a registry and factory for building topology masks that enforce
sparse or structured connectivity in spiking neural networks.
"""

from __future__ import annotations

from typing import Any, Dict, Type

from .base import BaseTopology
from .random import RandomTopology
from .small_world import SmallWorldTopology

_TOPOLOGY_REGISTRY: Dict[str, Type[BaseTopology]] = {
    "random": RandomTopology,
    "small_world": SmallWorldTopology,
}


def register_topology(name: str, cls: Type[BaseTopology]) -> None:
    """Register a custom topology class under *name*."""
    _TOPOLOGY_REGISTRY[name] = cls


def build_topology(config: Dict[str, Any]) -> BaseTopology:
    """Instantiate a topology from a configuration dictionary.

    Args:
        config: Must contain a ``"type"`` key whose value matches a
                registered topology name.

    Returns:
        An initialised ``BaseTopology`` instance.

    Raises:
        ValueError: If ``"type"`` is missing or unrecognised.
    """
    topo_type = config.get("type", "random")
    if topo_type not in _TOPOLOGY_REGISTRY:
        raise ValueError(
            f"Unknown topology type '{topo_type}'. "
            f"Available: {sorted(_TOPOLOGY_REGISTRY)}"
        )
    return _TOPOLOGY_REGISTRY[topo_type](config)


__all__ = [
    "BaseTopology",
    "RandomTopology",
    "SmallWorldTopology",
    "register_topology",
    "build_topology",
]
