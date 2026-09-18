import torch

from .base import BaseEncoder


class RateEncoder(BaseEncoder):
    """Rate coding encoder.

    Converts analog values in [0, 1] to spike trains where each time step
    fires independently with probability proportional to the input value.

    spikes[t] = Bernoulli(data * max_rate * dt)

    Config keys:
        max_rate (float): Maximum firing rate in Hz. Default: 100.
        dt (float): Simulation time step in seconds. Default: 0.001.
    """

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self.max_rate: float = config.get("max_rate", 100.0)
        self.dt: float = config.get("dt", 0.001)

    def encode(self, data: torch.Tensor, time_steps: int) -> torch.Tensor:
        """Generate Poisson-like spike trains from analog values.

        Args:
            data: (batch, features) values in [0, 1].
            time_steps: Number of simulation time steps.

        Returns:
            spikes: (time_steps, batch, features) binary tensor.
        """
        batch_size, features = data.shape

        # Probability of firing at each time step
        prob = data.clamp(0.0, 1.0) * self.max_rate * self.dt  # (batch, features)
        prob = prob.clamp(0.0, 1.0)

        # Draw independent Bernoulli samples for every time step
        spikes = torch.bernoulli(
            prob.unsqueeze(0).expand(time_steps, -1, -1)
        )  # (time_steps, batch, features)

        return spikes
