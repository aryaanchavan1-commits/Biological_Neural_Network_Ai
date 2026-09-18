import torch

from .base import BaseDecoder


class SpikeDecoder(BaseDecoder):
    """Spike-based decoder.

    Extracts information directly from spike timing or presence.

    Config keys:
        mode (str): How to aggregate across time.
                    "last"  – use only the last time step's spikes.
                    "max"   – take the maximum value across time per neuron.
                    Default: "last".
    """

    _MODES = ("last", "max")

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self.mode: str = config.get("mode", "last")
        if self.mode not in self._MODES:
            raise ValueError(
                f"Unknown mode '{self.mode}'. Must be one of {self._MODES}"
            )

    def decode(self, spike_traces: torch.Tensor) -> torch.Tensor:
        """Decode spike trains via the configured strategy.

        Args:
            spike_traces: (time_steps, batch, features)

        Returns:
            (batch, features) decoded tensor.
        """
        if self.mode == "last":
            return spike_traces[-1]  # (batch, features)
        else:  # "max"
            return spike_traces.max(dim=0).values  # (batch, features)
