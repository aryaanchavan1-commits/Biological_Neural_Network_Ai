import torch

from .base import BaseDecoder


class RateDecoder(BaseDecoder):
    """Rate-based decoder.

    Computes the average firing rate by summing spikes over time and
    dividing by the number of time steps.

        output = sum(spikes, dim=0) / time_steps

    This produces a value in [0, 1] representing the proportion of
    time steps in which each neuron spiked.
    """

    def decode(self, spike_traces: torch.Tensor) -> torch.Tensor:
        """Compute firing rates from spike trains.

        Args:
            spike_traces: (time_steps, batch, features)

        Returns:
            (batch, features) firing rates in [0, 1].
        """
        time_steps = spike_traces.shape[0]
        return spike_traces.sum(dim=0) / time_steps
