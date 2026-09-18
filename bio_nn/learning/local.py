import torch

from .base import BaseLearningRule


class LocalLearning(BaseLearningRule):
    """Local learning rules: Hebbian, competitive, anti-Hebbian."""

    def __init__(self, config):
        super().__init__(config)
        self.rule_type = config.get("local_rule", "hebbian")
        self.learning_rate = config.get("learning_rate", 0.01)

    def compute_update(self, weights, pre_spikes, post_spikes, **kwargs):
        if self.rule_type == "hebbian":
            batch = pre_spikes.shape[0]
            return torch.einsum('bi,bj->ij', pre_spikes, post_spikes) / batch
        elif self.rule_type == "competitive":
            batch = pre_spikes.shape[0]
            update = torch.einsum('bi,bj->ij', pre_spikes, post_spikes) / batch
            threshold = update.quantile(0.7)
            update[update < threshold] = 0
            return update
        elif self.rule_type == "anti_hebbian":
            batch = pre_spikes.shape[0]
            hebb = torch.einsum('bi,bj->ij', pre_spikes, post_spikes) / batch
            anti = torch.einsum('bi,bj->ij', pre_spikes, (1 - post_spikes)) / batch
            return hebb - 0.5 * anti
        else:
            raise ValueError(f"Unknown local rule: {self.rule_type}")

    def apply_update(self, weights, update):
        w = weights + self.learning_rate * update
        return torch.clamp(w, min=0.0, max=1.0)
