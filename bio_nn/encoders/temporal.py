import torch

from .base import BaseEncoder


class TemporalEncoder(BaseEncoder):
    """Temporal (latency) coding encoder.

    Higher input values produce earlier spikes.  A value of 1.0 spikes
    at the first time step; a value of 0.0 never spikes.  Intermediate
    values spike at a time step proportional to (1 - value).

    Config keys:
        threshold (float): Minimum value to produce a spike. Default: 0.01.
    """

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self.threshold: float = config.get("threshold", 0.01)

    def encode(self, data: torch.Tensor, time_steps: int) -> torch.Tensor:
        """Generate one spike per feature at a time proportional to (1 - value).

        Args:
            data: (batch, features) values in [0, 1].
            time_steps: Number of simulation time steps.

        Returns:
            spikes: (time_steps, batch, features) binary tensor.
        """
        batch_size, features = data.shape
        spikes = torch.zeros(time_steps, batch_size, features, device=data.device)

        # Spike time: t = round((1 - value) * (time_steps - 1))
        # Clamped so that very small values may not spike at all.
        values = data.clamp(0.0, 1.0)
        spike_times = ((1.0 - values) * (time_steps - 1)).round().long()  # (batch, features)

        # Create mask for neurons that should spike
        mask = values > self.threshold  # (batch, features)

        # Scatter spikes into the time dimension
        for b in range(batch_size):
            for f in range(features):
                if mask[b, f]:
                    t = spike_times[b, f].item()
                    spikes[t, b, f] = 1.0

        return spikes
