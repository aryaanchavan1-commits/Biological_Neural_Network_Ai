from abc import ABC, abstractmethod
import torch


class BaseLearningRule(ABC):
    """Base class for learning rules that compute weight updates."""

    def __init__(self, config):
        self.config = config

    @abstractmethod
    def compute_update(self, weights, pre_spikes, post_spikes, **kwargs) -> torch.Tensor:
        """Compute weight update.
        Returns: weight delta tensor of same shape as weights
        """
        ...

    @abstractmethod
    def apply_update(self, weights, update) -> torch.Tensor:
        """Apply update to weights with optional clipping.
        Returns: updated weights
        """
        ...

    def get_statistics(self) -> dict:
        return {}
