from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class BaseDecoder(nn.Module, ABC):
    """Abstract base class for spike decoders.

    Decoders convert spike trains back into continuous-valued outputs.
    """

    def __init__(self, config: dict) -> None:
        super().__init__()
        self.config = config

    @abstractmethod
    def decode(self, spike_traces: torch.Tensor) -> torch.Tensor:
        """Decode spike trains into a continuous output.

        Args:
            spike_traces: (time_steps, batch, features) binary spike tensor.

        Returns:
            output: (batch, features) continuous tensor.
        """
        ...

    def forward(self, spike_traces: torch.Tensor) -> torch.Tensor:
        return self.decode(spike_traces)
