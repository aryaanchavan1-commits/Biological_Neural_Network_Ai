import torch

from .base import BaseLearningRule


class HybridLearning(BaseLearningRule):
    """Hybrid: global error-driven learning + local Hebbian/STDP component."""

    def __init__(self, config):
        super().__init__(config)
        self.global_weight = config.get("global_weight", 0.7)
        self.local_weight = config.get("local_weight", 0.3)

    def compute_update(self, weights, pre_spikes, post_spikes, **kwargs):
        batch = pre_spikes.shape[0]
        local_update = torch.einsum('bi,bj->ij', pre_spikes, post_spikes) / batch
        global_update = kwargs.get("global_update", torch.zeros_like(weights))
        return self.global_weight * global_update + self.local_weight * local_update

    def apply_update(self, weights, update):
        lr = self.config.get("learning_rate", 0.01)
        w = weights + lr * update
        return torch.clamp(w, min=0.0, max=1.0)
