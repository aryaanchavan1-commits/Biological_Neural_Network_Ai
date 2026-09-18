"""Neuromodulation module — registry and factory."""

from __future__ import annotations

from typing import Any, Dict, Type

from .base import BaseNeuromodulator
from .global_mod import GlobalModulator
from .reward import RewardModulator

__all__ = [
    "BaseNeuromodulator",
    "RewardModulator",
    "GlobalModulator",
    "get_neuromodulator",
    "list_neuromodulator_types",
]

_REGISTRY: Dict[str, Type[BaseNeuromodulator]] = {
    "reward": RewardModulator,
    "global": GlobalModulator,
}


def get_neuromodulator(name: str, **kwargs: Any) -> BaseNeuromodulator:
    """Instantiate a neuromodulator by name.

    Args:
        name: One of the registered neuromodulator type names.
        **kwargs: Keyword arguments forwarded to the constructor.

    Raises:
        KeyError: If *name* is not in the registry.
    """
    if name not in _REGISTRY:
        raise KeyError(
            f"Unknown neuromodulator type '{name}'. Available: {list(_REGISTRY)}"
        )
    return _REGISTRY[name](**kwargs)


def list_neuromodulator_types() -> list[str]:
    """Return the names of all registered neuromodulator types."""
    return list(_REGISTRY)
