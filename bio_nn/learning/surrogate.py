import torch

from .base import BaseLearningRule


class SurrogateGradient(BaseLearningRule):
    """Surrogate gradient learning for SNNs.

    Uses a surrogate function to approximate the gradient of the spike
    threshold function. Common choices: fast sigmoid, sigmoid, triangular.
    """

    def __init__(self, config):
        super().__init__(config)
        self.slope = config.get("slope", 5.0)
        self.surrogate_type = config.get("type", "fast_sigmoid")

    def surrogate_function(self, membrane_potential, threshold):
        """Compute surrogate gradient.

        fast_sigmoid: 1 / (1 + slope * |x - threshold|)^2
        sigmoid: torch.sigmoid(slope * (x - threshold))
        triangular: max(0, 1 - slope * |x - threshold|)
        """
        x = membrane_potential - threshold
        if self.surrogate_type == "fast_sigmoid":
            return 1.0 / (1.0 + self.slope * x.abs()).pow(2)
        elif self.surrogate_type == "sigmoid":
            return torch.sigmoid(self.slope * x)
        elif self.surrogate_type == "triangular":
            return torch.clamp(1.0 - self.slope * x.abs(), min=0.0)
        else:
            raise ValueError(f"Unknown surrogate type: {self.surrogate_type}")

    def compute_update(self, weights, pre_spikes, post_spikes, **kwargs):
        """Standard Hebbian-like update via surrogate gradients."""
        batch_size = pre_spikes.shape[0]
        update = torch.einsum('bi,bj->ij', pre_spikes, post_spikes) / batch_size
        return update

    def apply_update(self, weights, update):
        lr = self.config.get("learning_rate", 0.01)
        w = weights + lr * update
        w_min = self.config.get("w_min", 0.0)
        w_max = self.config.get("w_max", 1.0)
        return torch.clamp(w, min=w_min, max=w_max)
