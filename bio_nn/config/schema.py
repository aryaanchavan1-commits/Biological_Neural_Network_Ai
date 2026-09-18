"""Configuration validation helpers."""

from __future__ import annotations

from typing import Any, Dict, List

from bio_nn.config.defaults import DEFAULTS


def _check_required_keys(
    config: Dict[str, Any],
    required: List[str],
    path: str = "",
) -> List[str]:
    """Return error messages for any missing required keys."""
    errors: List[str] = []
    for key in required:
        if key not in config:
            errors.append(f"Missing required key: '{path}{key}'")
    return errors


def _check_types(
    config: Dict[str, Any],
    spec: Dict[str, type],
    path: str = "",
) -> List[str]:
    """Return error messages for value type mismatches."""
    errors: List[str] = []
    for key, expected in spec.items():
        if key in config and not isinstance(config[key], expected):
            actual = type(config[key]).__name__
            errors.append(
                f"Type mismatch for '{path}{key}': expected "
                f"{expected.__name__}, got {actual}"
            )
    return errors


def validate_config(config: Dict[str, Any]) -> List[str]:
    """Validate a general configuration dict.

    Checks:
    - ``"experiment"`` section exists.
    - ``"model"`` section exists.
    - ``"data"`` section exists.
    - No unknown top-level keys (warns, does not error).

    Args:
        config: Configuration dictionary to validate.

    Returns:
        List of human-readable error strings (empty if valid).
    """
    errors: List[str] = []

    errors.extend(_check_required_keys(config, ["experiment", "model", "data"]))
    errors.extend(_check_types(config, {"experiment": dict, "model": dict, "data": dict}))

    known_sections = set(DEFAULTS.keys()) | {"experiment", "model", "data"}
    for key in config:
        if key not in known_sections:
            errors.append(f"Unknown top-level config section: '{key}'")

    return errors


def validate_experiment_config(config: Dict[str, Any]) -> List[str]:
    """Validate the ``experiment`` sub-section.

    Expected keys:
    - ``name`` (str)
    - ``seed`` (int, optional)
    - ``epochs`` (int, optional)

    Args:
        config: The ``experiment`` dict.

    Returns:
        List of error strings (empty if valid).
    """
    errors: List[str] = []
    errors.extend(_check_required_keys(config, ["name"], path="experiment."))
    errors.extend(_check_types(
        config,
        {"name": str, "seed": int, "epochs": int},
        path="experiment.",
    ))
    return errors


def validate_model_config(config: Dict[str, Any]) -> List[str]:
    """Validate the ``model`` sub-section.

    Expected keys:
    - ``type`` (str) — component registry name
    - ``n_neurons`` (int, optional)
    - ``dt`` (float, optional)

    Args:
        config: The ``model`` dict.

    Returns:
        List of error strings (empty if valid).
    """
    errors: List[str] = []
    errors.extend(_check_required_keys(config, ["type"], path="model."))
    errors.extend(_check_types(
        config,
        {"type": str, "n_neurons": int, "dt": (int, float)},
        path="model.",
    ))
    if "n_neurons" in config and config["n_neurons"] <= 0:
        errors.append("model.n_neurons must be positive")
    return errors
