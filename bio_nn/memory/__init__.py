"""Memory module — registry and factory for all memory types."""

from __future__ import annotations

from typing import Any, Dict, Type

from .associative import AssociativeMemory
from .base import BaseMemory
from .recurrent import RecurrentMemory
from .synaptic import SynapticMemory
from .working_memory import WorkingMemory

__all__ = [
    "BaseMemory",
    "RecurrentMemory",
    "WorkingMemory",
    "AssociativeMemory",
    "SynapticMemory",
    "get_memory",
    "list_memory_types",
]

_REGISTRY: Dict[str, Type[BaseMemory]] = {
    "recurrent": RecurrentMemory,
    "working": WorkingMemory,
    "associative": AssociativeMemory,
    "synaptic": SynapticMemory,
}


def get_memory(name: str, **kwargs: Any) -> BaseMemory:
    """Instantiate a memory module by name.

    Args:
        name: One of the registered memory type names.
        **kwargs: Keyword arguments forwarded to the constructor.

    Raises:
        KeyError: If *name* is not in the registry.
    """
    if name not in _REGISTRY:
        raise KeyError(f"Unknown memory type '{name}'. Available: {list(_REGISTRY)}")
    return _REGISTRY[name](**kwargs)


def list_memory_types() -> list[str]:
    """Return the names of all registered memory types."""
    return list(_REGISTRY)
