from .base import BaseDecoder
from .rate_decoder import RateDecoder
from .spike_decoder import SpikeDecoder


_REGISTRY = {
    "rate": RateDecoder,
    "spike": SpikeDecoder,
}


def create_decoder(decoder_type: str, config: dict | None = None) -> BaseDecoder:
    """Factory for decoder creation.

    Args:
        decoder_type: One of {"rate", "spike"}.
        config: Optional configuration dict forwarded to the decoder.

    Returns:
        A BaseDecoder instance.
    """
    if decoder_type not in _REGISTRY:
        raise ValueError(
            f"Unknown decoder type '{decoder_type}'. "
            f"Available: {list(_REGISTRY.keys())}"
        )
    return _REGISTRY[decoder_type](config or {})
