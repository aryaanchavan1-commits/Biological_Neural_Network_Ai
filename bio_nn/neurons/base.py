"""Abstract base class for all neuron models in the BIO-NN framework."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn


class BaseNeuron(nn.Module, ABC):
    """Abstract base class for biologically-inspired neuron models.

    All neuron models must implement forward() and maintain membrane
    potential, spike history, and state dictionaries. This interface
    ensures consistent behavior across different neuron implementations
    for training, simulation, and visualization.

    Args:
        size: Number of neurons in this layer.
        config: Dictionary of neuron-specific configuration parameters.
    """

    def __init__(self, size: int, config: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.size = size
        self.config = config or {}

        self._membrane_history: List[torch.Tensor] = []
        self._spike_history: List[torch.Tensor] = []
        self._step_count: int = 0

        self._device = torch.device("cpu")
        self._dtype = torch.float32

    def _init_state(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> Dict[str, torch.Tensor]:
        """Initialize state tensors. Override in subclasses for model-specific state."""
        return {}

    def _get_initial_state(self, batch_size: int, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None) -> Dict[str, torch.Tensor]:
        device = device or self._device
        dtype = dtype or self._dtype
        return self._init_state(batch_size, device, dtype)

    @abstractmethod
    def forward(self, input_current: torch.Tensor, state: Dict[str, torch.Tensor]):
        """Process input current for one time step.

        Args:
            input_current: Tensor of shape (batch, size) representing
                the input current to each neuron.
            state: Dictionary of previous state tensors. Keys depend on
                the specific neuron model.

        Returns:
            Tuple of (spikes, membrane_potential, new_state):
                - spikes: Binary tensor (batch, size) indicating which neurons fired.
                - membrane_potential: Tensor (batch, size) of current membrane potentials.
                - new_state: Dictionary of updated state tensors.
        """
        ...

    def reset_state(self):
        """Clear all recorded history and step count."""
        self._membrane_history.clear()
        self._spike_history.clear()
        self._step_count = 0

    def get_state(self) -> Dict[str, torch.Tensor]:
        """Return the most recent state dict, or empty if no steps have run."""
        return getattr(self, "_last_state", {})

    def get_spike_history(self) -> List[torch.Tensor]:
        """Return list of spike tensors from each simulated time step."""
        return self._spike_history

    def get_membrane_history(self) -> List[torch.Tensor]:
        """Return list of membrane potential tensors from each simulated time step."""
        return self._membrane_history

    def get_statistics(self) -> Dict[str, Any]:
        """Return summary statistics over the recorded history.

        Includes mean/std firing rate, mean membrane potential over time,
        and total step count.
        """
        stats: Dict[str, Any] = {
            "step_count": self._step_count,
            "size": self.size,
        }

        if self._spike_history:
            all_spikes = torch.stack(self._spike_history)  # (T, batch, size)
            stats["mean_firing_rate"] = all_spikes.mean().item()
            stats["total_spikes"] = all_spikes.sum().item()

        if self._membrane_history:
            all_membrane = torch.stack(self._membrane_history)
            stats["mean_membrane_potential"] = all_membrane.mean().item()
            stats["std_membrane_potential"] = all_membrane.std().item()

        return stats

    def record_step(self, spikes: torch.Tensor, membrane_potential: torch.Tensor, state: Dict[str, torch.Tensor]):
        """Record a single time step for history tracking.

        Call this inside forward() after computing spikes and membrane potential.
        """
        self._spike_history.append(spikes.detach().cpu())
        self._membrane_history.append(membrane_potential.detach().cpu())
        self._step_count += 1
        self._last_state = state

    def extra_repr(self) -> str:
        return f"size={self.size}, config={self.config}"
