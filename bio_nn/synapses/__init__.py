from .base import BaseSynapse
from .static import StaticSynapse
from .dynamic import DynamicSynapse


_REGISTRY = {
    "static": StaticSynapse,
    "dynamic": DynamicSynapse,
}


def create_synapse(synapse_type: str, in_features: int, out_features: int, config: dict | None = None) -> BaseSynapse:
    """Factory for synapse creation.

    Args:
        synapse_type: One of {"static", "dynamic"}.
        in_features: Number of pre-synaptic neurons.
        out_features: Number of post-synaptic neurons.
        config: Optional configuration dict forwarded to the synapse.

    Returns:
        A BaseSynapse instance.
    """
    if synapse_type not in _REGISTRY:
        raise ValueError(
            f"Unknown synapse type '{synapse_type}'. "
            f"Available: {list(_REGISTRY.keys())}"
        )
    return _REGISTRY[synapse_type](in_features, out_features, config or {})
