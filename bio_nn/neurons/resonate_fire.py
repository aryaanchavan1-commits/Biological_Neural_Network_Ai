"""Resonate-and-Fire neuron model."""

import math
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from .base import BaseNeuron


class ResonateAndFireNeuron(BaseNeuron):
    """Resonate-and-Fire neuron with oscillatory membrane dynamics.

    Unlike LIF which exponentially decays to rest, the Resonate-and-Fire
    neuron has a damped oscillatory response, naturally resonating at a
    preferred frequency. This makes it sensitive to input temporal structure
    and capable of frequency-selective filtering.

    Dynamics (two-variable formulation):
        dz/dt = -lambda * z + w * x + I
        dx/dt = -lambda * x - w * z
        if x >= threshold: spike, x = x - 2*threshold

    where z is the primary state, x is the orthogonal component, w is the
    resonant angular frequency, and lambda controls damping.

    The natural frequency is f = w / (2*pi), and the quality factor is
    Q = w / (2*lambda).

    Args:
        size: Number of neurons.
        config: Dictionary with keys:
            - w (float): Resonant angular frequency. Default 2*pi*0.05 (~0.314 rad/step).
            - lambda_damp (float): Damping coefficient. Default 0.1.
            - threshold (float): Spike threshold on x. Default 1.0.
            - z_init (float): Initial z value. Default 0.0.
            - x_init (float): Initial x value. Default 0.0.
            - refractory_period (int): Refractory steps. Default 0.
            - spike_grad (str): Surrogate gradient. Default "fast_sigmoid".
    """

    def __init__(self, size: int, config: Optional[Dict[str, Any]] = None):
        super().__init__(size, config)

        self.w: float = self.config.get("w", 2.0 * math.pi * 0.05)
        self.lambda_damp: float = self.config.get("lambda_damp", 0.1)
        self.threshold: float = self.config.get("threshold", 1.0)
        self.z_init: float = self.config.get("z_init", 0.0)
        self.x_init: float = self.config.get("x_init", 0.0)
        self.refractory_period: int = self.config.get("refractory_period", 0)

        self.natural_freq = self.w / (2.0 * math.pi)
        self.quality_factor = self.w / (2.0 * self.lambda_damp) if self.lambda_damp > 0 else float("inf")

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
            "z": torch.full((batch_size, self.size), self.z_init, device=device, dtype=dtype),
            "x": torch.full((batch_size, self.size), self.x_init, device=device, dtype=dtype),
            "spike_count": torch.zeros(batch_size, self.size, device=device, dtype=dtype),
            "refractory_counter": torch.zeros(batch_size, self.size, device=device, dtype=torch.long),
        }

    def forward(self, input_current: torch.Tensor, state: Dict[str, torch.Tensor]):
        """Single-step Resonate-and-Fire forward pass.

        Args:
            input_current: (batch, size) input current.
            state: Must contain "z" and "x".

        Returns:
            Tuple of (spikes, membrane_potential, new_state).
        """
        z = state["z"]
        x = state["x"]
        spike_count = state.get("spike_count", torch.zeros_like(z))
        refractory_counter = state.get(
            "refractory_counter",
            torch.zeros(z.shape[0], self.size, device=z.device, dtype=torch.long),
        )

        refractory_mask = refractory_counter > 0

        # Coupled oscillatory dynamics (Euler integration)
        # dz/dt = -lambda * z + w * x + I
        # dx/dt = -lambda * x - w * z
        lam = self.lambda_damp
        w = self.w

        z_new = z + (-lam * z + w * x + input_current)
        x_new = x + (-lam * x - w * z)

        # Apply refractory mask
        z_new = torch.where(refractory_mask, z, z_new)
        x_new = torch.where(refractory_mask, x, x_new)

        # Spike generation based on x crossing threshold
        spike_input = self.surrogate_grad(x_new - self.threshold)
        spikes = (spike_input > 0.5).float()

        # Reset: subtract threshold from x when spike occurs
        x_new = x_new - spikes * 2.0 * self.threshold

        # Update refractory counter
        refractory_counter = torch.where(
            spikes > 0,
            torch.full_like(refractory_counter, self.refractory_period),
            torch.clamp(refractory_counter - 1, min=0),
        )

        spike_count = spike_count + spikes

        new_state = {
            "z": z_new,
            "x": x_new,
            "spike_count": spike_count,
            "refractory_counter": refractory_counter,
        }

        self.record_step(spikes, x_new, new_state)
        return spikes, x_new, new_state

    def extra_repr(self) -> str:
        return (
            f"size={self.size}, w={self.w:.4f}, lambda={self.lambda_damp}, "
            f"natural_freq={self.natural_freq:.4f}Hz, Q={self.quality_factor:.2f}, "
            f"threshold={self.threshold}"
        )
