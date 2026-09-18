"""Prediction module — registry and factory for all prediction types."""

from __future__ import annotations

from typing import Any, Dict, Type

from .base import BasePrediction
from .error import PredictionError
from .predictor import Predictor

__all__ = [
    "BasePrediction",
    "Predictor",
    "PredictionError",
    "get_predictor",
    "list_predictor_types",
]

_REGISTRY: Dict[str, Type[BasePrediction]] = {
    "predictor": Predictor,
    "error": PredictionError,
    "linear": Predictor,
    "mlp": Predictor,
    "temporal": Predictor,
}


def get_predictor(name: str, **kwargs: Any) -> BasePrediction:
    """Instantiate a prediction module by name.

    Args:
        name: One of the registered prediction type names.
        **kwargs: Keyword arguments forwarded to the constructor.

    Raises:
        KeyError: If *name* is not in the registry.
    """
    if name not in _REGISTRY:
        raise KeyError(f"Unknown predictor type '{name}'. Available: {list(_REGISTRY)}")
    return _REGISTRY[name](**kwargs)


def list_predictor_types() -> list[str]:
    """Return the names of all registered predictor types."""
    return list(_REGISTRY)
