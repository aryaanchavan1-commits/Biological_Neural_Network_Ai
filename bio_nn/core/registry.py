"""Component registry for dynamic instantiation from config.

Usage::

    from bio_nn.core.registry import ComponentRegistry

    # Register
    ComponentRegistry.register("lif_neuron", LIFNeuron)

    # Instantiate from a config dict
    neuron = ComponentRegistry.build({"type": "lif_neuron", "tau_mem": 20.0})
"""

from __future__ import annotations

import importlib
import logging
from typing import Any, Dict, Optional, Type

logger = logging.getLogger(__name__)


class ComponentRegistry:
    """Global registry that maps string names to component classes.

    Components can be registered explicitly via ``register`` or discovered
    via ``import_path`` entries in config dicts.
    """

    _registry: Dict[str, Type] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @classmethod
    def register(cls, name: str, component_class: Type) -> None:
        """Register a component class under *name*.

        Args:
            name:            Lookup key (e.g. ``"lif_neuron"``).
            component_class: The class to associate with *name*.

        Raises:
            TypeError: If *component_class* is not a type / callable.
        """
        if not isinstance(component_class, type):
            raise TypeError(
                f"Expected a class, got {type(component_class).__name__}"
            )
        if name in cls._registry:
            logger.debug("Overwriting registry entry '%s'", name)
        cls._registry[name] = component_class
        logger.debug("Registered '%s' -> %s", name, component_class.__qualname__)

    @classmethod
    def get(cls, name: str) -> Type:
        """Look up a component class by name.

        Args:
            name: The registered name.

        Returns:
            The component class.

        Raises:
            KeyError: If *name* is not in the registry.
        """
        if name not in cls._registry:
            raise KeyError(
                f"Component '{name}' is not registered. "
                f"Available: {sorted(cls._registry)}"
            )
        return cls._registry[name]

    @classmethod
    def build(cls, config: Dict[str, Any]) -> Any:
        """Instantiate a component from a config dict.

        The dict **must** contain a ``"type"`` key whose value is a
        registered name or a fully-qualified ``"import_path"``.

        Extra keys are forwarded as ``**kwargs`` to the constructor.

        Args:
            config: Config dict with at least a ``"type"`` key.

        Returns:
            An instance of the resolved component class.

        Raises:
            ValueError: If ``"type"`` is missing.
            KeyError:   If the type is not registered and no import path
                        is provided.
        """
        if "type" not in config:
            raise ValueError("Config must contain a 'type' key for ComponentRegistry.build")

        name = config["type"]
        kwargs = {k: v for k, v in config.items() if k != "type"}

        # Try direct lookup first
        try:
            component_cls = cls.get(name)
        except KeyError:
            # Fall back to import_path if supplied
            import_path = kwargs.pop("import_path", None) or config.get("import_path")
            if import_path is None:
                raise
            component_cls = cls._import_from_path(import_path)
            # Register for future lookups
            cls._registry[name] = component_cls

        return component_cls(**kwargs)

    @classmethod
    def list_registered(cls) -> list[str]:
        """Return sorted list of registered component names."""
        return sorted(cls._registry)

    @classmethod
    def clear(cls) -> None:
        """Remove all registrations.  Intended for testing only."""
        cls._registry.clear()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _import_from_path(dotted_path: str) -> Type:
        """Import a class from a dotted module path like ``pkg.mod.Class``.

        Args:
            dotted_path: Fully qualified attribute path.

        Returns:
            The resolved class / callable.

        Raises:
            ImportError: If the module cannot be imported.
            AttributeError: If the attribute does not exist on the module.
        """
        module_path, _, attr_name = dotted_path.rpartition(".")
        if not module_path:
            raise ImportError(
                f"Invalid import path '{dotted_path}': expected 'module.ClassName'"
            )
        module = importlib.import_module(module_path)
        return getattr(module, attr_name)
