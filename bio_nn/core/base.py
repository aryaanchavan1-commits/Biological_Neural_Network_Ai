"""Abstract base classes for all BIO-NN components.

Every module in the framework inherits from one of these ABCs so that
the registry, config loader, and experiment runner can treat them uniformly.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Neuron / Synapse / Plasticity primitives
# ---------------------------------------------------------------------------


class NeuronModel(ABC, nn.Module):
    """Base class for all neuron models.

    Subclasses must implement ``forward`` which advances the neuron state
    by one time-step.
    """

    @abstractmethod
    def forward(
        self, state: Dict[str, torch.Tensor], input: torch.Tensor
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        """Advance the neuron state by one time-step.

        Args:
            state:  Dictionary of internal state tensors (membrane potential,
                    threshold, trace, etc.).  The exact keys depend on the
                    concrete neuron model.
            input:  Current input tensor (post-synaptic currents).

        Returns:
            A 3-tuple ``(new_state, spikes, membrane_potential)``.
        """
        ...


class SynapseModel(ABC, nn.Module):
    """Base class for synaptic transmission models."""

    @abstractmethod
    def forward(
        self, pre_spikes: torch.Tensor, weights: torch.Tensor
    ) -> torch.Tensor:
        """Compute the post-synaptic input from pre-synaptic spikes.

        Args:
            pre_spikes: Spike train from the pre-synaptic population.
            weights:    Synaptic weight matrix.

        Returns:
            Post-synaptic current / input tensor.
        """
        ...


class PlasticityRule(ABC, nn.Module):
    """Base class for weight-plasticity rules (STDP, etc.)."""

    @abstractmethod
    def forward(
        self,
        weights: torch.Tensor,
        pre_spikes: torch.Tensor,
        post_spikes: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Compute updated weights for one time-step.

        Args:
            weights:     Current weight matrix.
            pre_spikes:  Pre-synaptic spike train.
            post_spikes: Post-synaptic spike train.

        Returns:
            Updated weight tensor.
        """
        ...


# ---------------------------------------------------------------------------
# Dendrites / Structural plasticity
# ---------------------------------------------------------------------------


class DendriticModule(ABC, nn.Module):
    """Models dendritic computation (branching, non-linear integration)."""

    @abstractmethod
    def forward(self, input: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Process input through dendritic branches.

        Args:
            input: Current input tensor.

        Returns:
            Dendritically processed output.
        """
        ...


class StructuralPlasticity(ABC, nn.Module):
    """Handles synapse formation / elimination during training."""

    @abstractmethod
    def forward(
        self,
        weights: torch.Tensor,
        activity: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Apply structural plasticity modifications.

        Args:
            weights:  Current weight matrix.
            activity: Neuron activity tensor (e.g. firing rates).

        Returns:
            Modified weight matrix.
        """
        ...


# ---------------------------------------------------------------------------
# Memory / Prediction
# ---------------------------------------------------------------------------


class MemoryModule(ABC, nn.Module):
    """Working / short-term / long-term memory module."""

    @abstractmethod
    def read(self, state: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Read from memory.

        Args:
            state: Internal memory state.

        Returns:
            Read output tensor.
        """
        ...

    @abstractmethod
    def write(
        self, state: Dict[str, torch.Tensor], input: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Write to memory and return updated state.

        Args:
            state: Current memory state.
            input: Data to write.

        Returns:
            Updated state dictionary.
        """
        ...

    def forward(self, state: Dict[str, torch.Tensor], input: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Read then write convenience wrapper."""
        output = self.read(state)
        new_state = self.write(state, input)
        return output, new_state


class PredictionModule(ABC, nn.Module):
    """Generative / predictive coding module."""

    @abstractmethod
    def predict(self, input: torch.Tensor) -> torch.Tensor:
        """Produce a prediction from the current input.

        Args:
            input: Current sensory input.

        Returns:
            Predicted next-state / reconstruction.
        """
        ...

    @abstractmethod
    def compute_error(
        self, input: torch.Tensor, prediction: torch.Tensor
    ) -> torch.Tensor:
        """Compute prediction error.

        Args:
            input:      Ground-truth input.
            prediction: Model prediction.

        Returns:
            Prediction error tensor.
        """
        ...


# ---------------------------------------------------------------------------
# Neuromodulation / Encoding / Decoding
# ---------------------------------------------------------------------------


class Neuromodulator(ABC, nn.Module):
    """Models neuromodulatory signals (dopamine, serotonin, etc.)."""

    @abstractmethod
    def forward(
        self, signal: torch.Tensor, **kwargs: Any
    ) -> torch.Tensor:
        """Apply neuromodulation to a signal.

        Args:
            signal: Input signal tensor.

        Returns:
            Modulated signal.
        """
        ...


class Encoder(ABC, nn.Module):
    """Encodes continuous data into spike trains."""

    @abstractmethod
    def forward(self, data: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Encode data into spikes.

        Args:
            data: Continuous input data.

        Returns:
            Spike tensor.
        """
        ...


class Decoder(ABC, nn.Module):
    """Decodes spike trains back into continuous values."""

    @abstractmethod
    def forward(self, spikes: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Decode spikes into output.

        Args:
            spikes: Spike tensor.

        Returns:
            Decoded continuous output.
        """
        ...


# ---------------------------------------------------------------------------
# Topology / Learning rules
# ---------------------------------------------------------------------------


class TopologyRule(ABC, nn.Module):
    """Generates network topology / connectivity matrices."""

    @abstractmethod
    def forward(
        self, n_neurons: int, **kwargs: Any
    ) -> torch.Tensor:
        """Generate an adjacency / connectivity matrix.

        Args:
            n_neurons: Number of neurons in the layer.

        Returns:
            Adjacency matrix tensor.
        """
        ...


class LearningRule(ABC, nn.Module):
    """Generic learning-rule interface (abstracts over STDP, R-stdp, etc.)."""

    @abstractmethod
    def forward(self, **kwargs: Any) -> torch.Tensor:
        """Compute the weight update tensor.

        Returns:
            Delta-weights tensor (same shape as the weight matrix).
        """
        ...


# ---------------------------------------------------------------------------
# Metrics / Visualization
# ---------------------------------------------------------------------------


class Metric(ABC, nn.Module):
    """Evaluation metric base class."""

    @abstractmethod
    def forward(
        self, predictions: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute metric value.

        Args:
            predictions: Model predictions.
            targets:     Ground-truth targets.

        Returns:
            Scalar metric tensor.
        """
        ...


class Visualizer(ABC):
    """Base class for rendering figures / plots."""

    @abstractmethod
    def render(self, data: Any, **kwargs: Any) -> Any:
        """Render a figure from data.

        Args:
            data: Arbitrary data to visualize.

        Returns:
            A matplotlib Figure or similar object.
        """
        ...
