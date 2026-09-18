"""Dendrites module — registry and factory for all dendrite types."""

from __future__ import annotations

from typing import Any, Dict, Type

from .base import BaseDendrite
from .compartment import CompartmentDendrite
from .nonlinear import NonlinearDendrite

__all__ = [
    "BaseDendrite",
    "CompartmentDendrite",
    "NonlinearDendrite",
    "get_dendrite",
    "list_dendrite_types",
]

_REGISTRY: Dict[str, Type[BaseDendrite]] = {
    "compartment": CompartmentDendrite,
    "nonlinear": NonlinearDendrite,
}


def get_dendrite(name: str, **kwargs: Any) -> BaseDendrite:
    """Instantiate a dendrite module by name.

    Args:
        name: One of the registered dendrite type names.
        **kwargs: Keyword arguments forwarded to the constructor.

    Raises:
        KeyError: If *name* is not in the registry.
    """
    if name not in _REGISTRY:
        raise KeyError(f"Unknown dendrite type '{name}'. Available: {list(_REGISTRY)}")
    return _REGISTRY[name](**kwargs)


def list_dendrite_types() -> list[str]:
    """Return the names of all registered dendrite types."""
    return list(_REGISTRY)
