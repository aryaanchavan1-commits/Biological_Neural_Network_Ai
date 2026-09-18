from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class BaseEncoder(nn.Module, ABC):
    """Abstract base class for spike encoders.

    Encoders convert continuous-valued data into binary spike trains
    suitable for spiking neural networks.
    """

    def __init__(self, config: dict) -> None:
        super().__init__()
        self.config = config

    @abstractmethod
    def encode(self, data: torch.Tensor, time_steps: int) -> torch.Tensor:
        """Encode static data into spike trains.

        Args:
            data: (batch, features) continuous-valued tensor, typically in [0, 1].
            time_steps: Number of simulation time steps.

        Returns:
            spikes: (time_steps, batch, features) binary spike tensor.
        """
        ...

    def forward(self, data: torch.Tensor, time_steps: int) -> torch.Tensor:
        return self.encode(data, time_steps)
