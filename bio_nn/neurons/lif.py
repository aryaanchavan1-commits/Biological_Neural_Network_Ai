"""Leaky Integrate-and-Fire (LIF) neuron model."""

from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from .base import BaseNeuron


class LIFNeuron(BaseNeuron):
    """Leaky Integrate-and-Fire neuron with configurable reset mechanism.

    Implements the standard LIF dynamics:
        tau_mem * dV/dt = -(V - V_rest) + R * I
        if V >= V_thresh: spike, V = reset(V)

    Supports three reset mechanisms:
        - "subtract": V = V - V_thresh (hard subtract)
        - "zero": V = 0
        - "none": no reset (integrate-and-fire without reset)

    The discrete-time update uses Euler integration:
        V[t+1] = beta * (V[t] - V_rest) + I * (1 - beta) + V_rest
    where beta = exp(-dt / tau_mem) and dt=1 (single step).

    Args:
        size: Number of neurons.
        config: Dictionary with keys:
            - tau_mem (float): Membrane time constant. Default 20.0.
            - threshold (float): Spike threshold. Default 1.0.
            - V_rest (float): Resting membrane potential. Default 0.0.
            - reset_mechanism (str): "subtract", "zero", or "none". Default "subtract".
            - refractory_period (int): Number of steps a neuron is refractory
              after spiking. Default 0.
            - spike_grad (str): Surrogate gradient type. Default "fast_sigmoid".
    """

    # Surrogate gradient functions for the spike threshold function
    _surrogate_grads = {}

    def __init__(self, size: int, config: Optional[Dict[str, Any]] = None):
        super().__init__(size, config)

        self.tau_mem: float = self.config.get("tau_mem", 20.0)
        self.threshold: float = self.config.get("threshold", 1.0)
        self.V_rest: float = self.config.get("V_rest", 0.0)
        self.reset_mechanism: str = self.config.get("reset_mechanism", "subtract")
        self.refractory_period: int = self.config.get("refractory_period", 0)

        self.beta = self._compute_decay(self.tau_mem)

        # Surrogate gradient for spike generation
        spike_grad_type = self.config.get("spike_grad", "fast_sigmoid")
        self.surrogate_grad = self._get_surrogate_grad(spike_grad_type)

        # Register persistent state as buffers so they move with .to(device)
        self.register_buffer("_v", torch.zeros(1))
        self.register_buffer("_spike_count", torch.zeros(1))
        self.register_buffer("_refractory_counter", torch.zeros(1, dtype=torch.long))

    @staticmethod
    def _compute_decay(tau_mem: float) -> float:
        """Compute membrane decay factor beta = exp(-1/tau_mem) for dt=1."""
        import math
        return math.exp(-1.0 / tau_mem)

    @staticmethod
    def _get_surrogate_grad(name: str):
        """Return a surrogate gradient function for the Heaviside step."""
        if name == "fast_sigmoid":
            def fast_sigmoid(x: torch.Tensor, slope: float = 25.0) -> torch.Tensor:
                return torch.sigmoid(slope * x)
            return fast_sigmoid
        elif name == "straight_through":
            def straight_through(x: torch.Tensor) -> torch.Tensor:
                return (x > 0).float()
            return straight_through
        else:
            raise ValueError(f"Unknown surrogate gradient: {name}")

    def _init_state(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> Dict[str, torch.Tensor]:
        return {
            "membrane": torch.full((batch_size, self.size), self.V_rest, device=device, dtype=dtype),
            "spike_count": torch.zeros(batch_size, self.size, device=device, dtype=dtype),
            "refractory_counter": torch.zeros(batch_size, self.size, device=device, dtype=torch.long),
        }

    def forward(self, input_current: torch.Tensor, state: Dict[str, torch.Tensor]):
        """Single-step LIF forward pass.

        Args:
            input_current: (batch, size) input current.
            state: Must contain "membrane" key; optionally "refractory_counter".

        Returns:
            Tuple of (spikes, membrane_potential, new_state).
        """
        v = state["membrane"]
        spike_count = state.get("spike_count", torch.zeros_like(v))
        refractory_counter = state.get("refractory_counter", torch.zeros(v.shape[0], self.size, device=v.device, dtype=torch.long))

        # Mask neurons in refractory period: they receive no input and don't update
        refractory_mask = refractory_counter > 0

        # Update membrane potential (Euler step) for non-refractory neurons
        v_new = self.beta * (v - self.V_rest) + input_current * (1.0 - self.beta) + self.V_rest
        v_new = torch.where(refractory_mask, v, v_new)

        # Generate spikes via surrogate gradient
        spike_input = self.surrogate_grad(v_new - self.threshold)
        spikes = (spike_input > 0.5).float()

        # Apply reset mechanism
        if self.reset_mechanism == "subtract":
            v_new = v_new - spikes * self.threshold
        elif self.reset_mechanism == "zero":
            v_new = torch.where(spikes > 0, torch.zeros_like(v_new), v_new)

        # Update refractory counter
        refractory_counter = torch.where(
            spikes > 0,
            torch.full_like(refractory_counter, self.refractory_period),
            torch.clamp(refractory_counter - 1, min=0),
        )

        # Track cumulative spike count
        spike_count = spike_count + spikes

        new_state = {
            "membrane": v_new,
            "spike_count": spike_count,
            "refractory_counter": refractory_counter,
        }

        self.record_step(spikes, v_new, new_state)
        return spikes, v_new, new_state

    def extra_repr(self) -> str:
        return (
            f"size={self.size}, tau_mem={self.tau_mem}, threshold={self.threshold}, "
            f"V_rest={self.V_rest}, reset={self.reset_mechanism}, "
            f"refractory={self.refractory_period}"
        )
