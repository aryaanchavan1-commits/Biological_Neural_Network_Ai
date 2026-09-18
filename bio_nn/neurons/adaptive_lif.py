"""Adaptive Leaky Integrate-and-Fire neuron model."""

from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from .base import BaseNeuron


class AdaptiveLIFNeuron(BaseNeuron):
    """Adaptive LIF neuron with dynamic threshold that increases with spiking activity.

    Implements the adaptive threshold model:
        tau_mem * dV/dt = -(V - V_rest) + R * I
        if V >= V_thresh + theta_adapt: spike, V = V - V_thresh
        tau_adapt * dtheta_adapt/dt = -theta_adapt + b * spike

    The adaptation current increases the effective threshold after each spike,
    implementing spike-frequency adaptation (reduces firing rate over time).

    Args:
        size: Number of neurons.
        config: Dictionary with keys:
            - tau_mem (float): Membrane time constant. Default 20.0.
            - threshold (float): Base spike threshold. Default 1.0.
            - V_rest (float): Resting membrane potential. Default 0.0.
            - tau_adapt (float): Adaptation time constant. Default 100.0.
            - threshold_adapt_rate (float): Adaptation increment b. Default 0.01.
            - reset_mechanism (str): "subtract" or "zero". Default "subtract".
            - refractory_period (int): Refractory steps after spike. Default 0.
            - spike_grad (str): Surrogate gradient type. Default "fast_sigmoid".
    """

    def __init__(self, size: int, config: Optional[Dict[str, Any]] = None):
        super().__init__(size, config)

        self.tau_mem: float = self.config.get("tau_mem", 20.0)
        self.threshold: float = self.config.get("threshold", 1.0)
        self.V_rest: float = self.config.get("V_rest", 0.0)
        self.tau_adapt: float = self.config.get("tau_adapt", 100.0)
        self.threshold_adapt_rate: float = self.config.get("threshold_adapt_rate", 0.01)
        self.reset_mechanism: str = self.config.get("reset_mechanism", "subtract")
        self.refractory_period: int = self.config.get("refractory_period", 0)

        import math
        self.beta = math.exp(-1.0 / self.tau_mem)
        self.beta_adapt = math.exp(-1.0 / self.tau_adapt)

        spike_grad_type = self.config.get("spike_grad", "fast_sigmoid")
        self.surrogate_grad = self._build_surrogate_grad(spike_grad_type)

    @staticmethod
    def _build_surrogate_grad(name: str):
        if name == "fast_sigmoid":
            return lambda x: torch.sigmoid(25.0 * x)
        elif name == "straight_through":
            return lambda x: (x > 0).float()
        raise ValueError(f"Unknown surrogate gradient: {name}")

    def _init_state(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> Dict[str, torch.Tensor]:
        return {
            "membrane": torch.full((batch_size, self.size), self.V_rest, device=device, dtype=dtype),
            "adaptation": torch.zeros(batch_size, self.size, device=device, dtype=dtype),
            "spike_count": torch.zeros(batch_size, self.size, device=device, dtype=dtype),
            "refractory_counter": torch.zeros(batch_size, self.size, device=device, dtype=torch.long),
        }

    def forward(self, input_current: torch.Tensor, state: Dict[str, torch.Tensor]):
        """Single-step adaptive LIF forward pass.

        Args:
            input_current: (batch, size) input current.
            state: Must contain "membrane" and "adaptation" keys.

        Returns:
            Tuple of (spikes, membrane_potential, new_state).
        """
        v = state["membrane"]
        theta_adapt = state["adaptation"]
        spike_count = state.get("spike_count", torch.zeros_like(v))
        refractory_counter = state.get("refractory_counter", torch.zeros(v.shape[0], self.size, device=v.device, dtype=torch.long))

        refractory_mask = refractory_counter > 0

        # Compute effective threshold (base + adaptation)
        effective_threshold = self.threshold + theta_adapt

        # Update membrane potential (Euler step)
        v_new = self.beta * (v - self.V_rest) + input_current * (1.0 - self.beta) + self.V_rest
        v_new = torch.where(refractory_mask, v, v_new)

        # Spike generation
        spike_input = self.surrogate_grad(v_new - effective_threshold)
        spikes = (spike_input > 0.5).float()

        # Apply reset
        if self.reset_mechanism == "subtract":
            v_new = v_new - spikes * self.threshold
        elif self.reset_mechanism == "zero":
            v_new = torch.where(spikes > 0, torch.zeros_like(v_new), v_new)

        # Update adaptation variable: theta_adapt decays toward b * spike_count
        theta_adapt_new = (
            self.beta_adapt * theta_adapt
            + (1.0 - self.beta_adapt) * self.threshold_adapt_rate * spikes
        )

        # Update refractory counter
        refractory_counter = torch.where(
            spikes > 0,
            torch.full_like(refractory_counter, self.refractory_period),
            torch.clamp(refractory_counter - 1, min=0),
        )

        spike_count = spike_count + spikes

        new_state = {
            "membrane": v_new,
            "adaptation": theta_adapt_new,
            "spike_count": spike_count,
            "refractory_counter": refractory_counter,
            "effective_threshold": effective_threshold,
        }

        self.record_step(spikes, v_new, new_state)
        return spikes, v_new, new_state

    def extra_repr(self) -> str:
        return (
            f"size={self.size}, tau_mem={self.tau_mem}, threshold={self.threshold}, "
            f"tau_adapt={self.tau_adapt}, adapt_rate={self.threshold_adapt_rate}, "
            f"reset={self.reset_mechanism}"
        )
