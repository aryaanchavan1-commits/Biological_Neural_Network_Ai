from .base import BaseEncoder
from .rate import RateEncoder
from .temporal import TemporalEncoder
from .population import PopulationEncoder


_REGISTRY = {
    "rate": RateEncoder,
    "temporal": TemporalEncoder,
    "population": PopulationEncoder,
}


def create_encoder(encoder_type: str, config: dict | None = None) -> BaseEncoder:
    """Factory for encoder creation.

    Args:
        encoder_type: One of {"rate", "temporal", "population"}.
        config: Optional configuration dict forwarded to the encoder.

    Returns:
        A BaseEncoder instance.
    """
    if encoder_type not in _REGISTRY:
        raise ValueError(
            f"Unknown encoder type '{encoder_type}'. "
            f"Available: {list(_REGISTRY.keys())}"
        )
    return _REGISTRY[encoder_type](config or {})
