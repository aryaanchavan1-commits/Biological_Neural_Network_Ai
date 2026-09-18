import torch

from .base import BaseEncoder


class PopulationEncoder(BaseEncoder):
    """Population coding encoder.

    Each input feature is represented by a population of neurons with
    Gaussian tuning curves evenly spaced across [0, 1].  A neuron
    fires at a rate proportional to its tuning curve evaluated at the
    input value.

    Config keys:
        n_neurons_per_feature (int): Number of neurons in the population for each feature.
                                     Default: 10.
        max_rate (float): Peak firing rate at the centre of the tuning curve.
                          Default: 100.
        dt (float): Simulation time step in seconds. Default: 0.001.
        sigma (float | None): Width of the Gaussian tuning curves.
                              If None, defaults to 1 / n_neurons_per_feature.
    """

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self.n_neurons: int = config.get("n_neurons_per_feature", 10)
        self.max_rate: float = config.get("max_rate", 100.0)
        self.dt: float = config.get("dt", 0.001)
        self.sigma: float = config.get("sigma") or (1.0 / self.n_neurons)

        # Centres of the tuning curves: evenly spaced in [0, 1]
        # (n_neurons,) – stored as a buffer (not a parameter)
        centres = torch.linspace(0.0, 1.0, self.n_neurons)
        self.register_buffer("centres", centres)

    def encode(self, data: torch.Tensor, time_steps: int) -> torch.Tensor:
        """Encode each feature via a population of Gaussian-tuned neurons.

        Args:
            data: (batch, features) values in [0, 1].
            time_steps: Number of simulation time steps.

        Returns:
            spikes: (time_steps, batch, features * n_neurons_per_feature) binary tensor.
        """
        batch_size, features = data.shape

        # Reshape data for broadcasting: (batch, features, 1)
        x = data.unsqueeze(-1)  # (batch, features, 1)

        # Tuning curve activations: Gaussian distance to each centre
        # centres: (n_neurons,) → (1, 1, n_neurons)
        centres = self.centres.view(1, 1, -1)
        activations = torch.exp(-0.5 * ((x - centres) / self.sigma) ** 2)  # (batch, features, n_neurons)

        # Spike probability per time step
        prob = (activations * self.max_rate * self.dt).clamp(0.0, 1.0)

        # Sample spikes
        spikes = torch.bernoulli(
            prob.unsqueeze(0).expand(time_steps, -1, -1, -1)
        )  # (time_steps, batch, features, n_neurons)

        # Flatten the last two dims → (time_steps, batch, features * n_neurons)
        return spikes.reshape(time_steps, batch_size, features * self.n_neurons)
