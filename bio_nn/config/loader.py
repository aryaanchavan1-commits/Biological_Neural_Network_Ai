"""YAML config loader with deep-merge and dot-access."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml


class Config(dict):
    """Dictionary subclass that supports attribute-style access.

    ``cfg.model.tau_mem`` is equivalent to ``cfg["model"]["tau_mem"]``.
    Nested dicts are also wrapped in ``Config``.
    """

    def __getattr__(self, key: str) -> Any:
        try:
            value = self[key]
        except KeyError:
            raise AttributeError(f"Config has no attribute '{key}'")
        return Config._wrap(value) if isinstance(value, dict) else value

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    def __delattr__(self, key: str) -> None:
        try:
            del self[key]
        except KeyError:
            raise AttributeError(f"Config has no attribute '{key}'")

    @staticmethod
    def _wrap(obj: Any) -> Any:
        """Recursively wrap dicts as ``Config``."""
        if isinstance(obj, dict):
            return Config(obj)
        if isinstance(obj, list):
            return [Config._wrap(v) for v in obj]
        return obj

    def to_dict(self) -> Dict[str, Any]:
        """Convert back to a plain nested dict."""
        out: Dict[str, Any] = {}
        for k, v in self.items():
            if isinstance(v, Config):
                out[k] = v.to_dict()
            else:
                out[k] = v
        return out

    def get_nested(self, dotted_key: str, default: Any = None) -> Any:
        """Retrieve a value via a dotted path like ``"model.tau_mem"``."""
        keys = dotted_key.split(".")
        current: Any = self
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return default
        return current


def load_config(path: Union[str, Path]) -> Config:
    """Load a YAML config file and return a :class:`Config`.

    Args:
        path: Path to the YAML file.

    Returns:
        Parsed configuration wrapped in ``Config``.

    Raises:
        FileNotFoundError: If the file does not exist.
        yaml.YAMLError: On parse failure.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if raw is None:
        return Config()
    return Config(raw)


def merge_configs(
    base: Dict[str, Any], override: Dict[str, Any]
) -> Dict[str, Any]:
    """Deep-merge *override* into *base* (mutates *base* in-place).

    Nested dicts are merged recursively; scalar values in *override*
    replace those in *base*.

    Args:
        base:    Base configuration (will be modified).
        override: Override configuration.

    Returns:
        The merged dictionary (same reference as *base*).
    """
    for key, value in override.items():
        if (
            key in base
            and isinstance(base[key], dict)
            and isinstance(value, dict)
        ):
            merge_configs(base[key], value)
        else:
            base[key] = copy.deepcopy(value)
    return base
