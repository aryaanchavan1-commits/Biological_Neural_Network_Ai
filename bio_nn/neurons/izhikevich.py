"""Izhikevich-style neuron model."""

from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from .base import BaseNeuron


# Standard Izhikevich neuron type presets
IZHIKEVICH_PRESETS: Dict[str, Dict[str, float]] = {
    "regular_spiking": {"a": 0.02, "b": 0.2, "c": -65.0, "d": 8.0},
    "fast_spiking": {"a": 0.10, "b": 0.2, "c": -65.0, "d": 2.0},
    "low_threshold_spiking": {"a": 0.02, "b": 0.25, "c": -65.0, "d": 2.0},
    "chattering": {"a": 0.02, "b": 0.2, "c": -50.0, "d": 2.0},
    "intrinsically_bursting": {"a": 0.02, "b": 0.2, "c": -55.0, "d": 4.0},
}


class IzhikevichNeuron(BaseNeuron):
    """Izhikevich two-variable spiking neuron model.

    Implements the Izhikevich model with dimensionless parameters:
        dv/dt = 0.04*v^2 + 5*v + 140 - u + I
        du/dt = a*(b*v - u)
        if v >= 30: v = c, u = u + d

    This model reproduces a wide range of cortical neuron spiking patterns
    with just 4 parameters (a, b, c, d), making it computationally efficient
    while remaining biologically plausible.

    Use the `preset` config key to load standard neuron types, or specify
    a, b, c, d directly.

    Args:
        size: Number of neurons.
        config: Dictionary with keys:
            - preset (str): Name of Izhikevich preset. Overrides a/b/c/d.
              Options: "regular_spiking", "fast_spiking", "low_threshold_spiking",
              "chattering", "intrinsically_bursting".
            - a (float): Recovery time constant. Default 0.02.
            - b (float): Recovery sensitivity to subthreshold fluctuations. Default 0.2.
            - c (float): Reset value of v. Default -65.0.
            - d (float): Reset value of u. Default 8.0.
            - v_init (float): Initial membrane potential. Default -70.0.
            - u_init (float): Initial recovery variable. Default -14.0.
            - v_peak (float): Peak voltage for spike detection. Default 30.0.
            - dt (float): Time step for integration. Default 1.0.
    """

    def __init__(self, size: int, config: Optional[Dict[str, Any]] = None):
        super().__init__(size, config)

        preset_name = self.config.get("preset")
        if preset_name and preset_name in IZHIKEVICH_PRESETS:
            preset = IZHIKEVICH_PRESETS[preset_name]
        elif preset_name and preset_name not in IZHIKEVICH_PRESETS:
            available = ", ".join(IZHIKEVICH_PRESETS.keys())
            raise ValueError(f"Unknown preset '{preset_name}'. Available: {available}")
        else:
            preset = {}

        self.a: float = self.config.get("a", preset.get("a", 0.02))
        self.b: float = self.config.get("b", preset.get("b", 0.2))
        self.c: float = self.config.get("c", preset.get("c", -65.0))
        self.d: float = self.config.get("d", preset.get("d", 8.0))

        self.v_init: float = self.config.get("v_init", -70.0)
        self.u_init: float = self.config.get("u_init", -14.0)
        self.v_peak: float = self.config.get("v_peak", 30.0)
        self.dt: float = self.config.get("dt", 1.0)

    def _init_state(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> Dict[str, torch.Tensor]:
        return {
            "v": torch.full((batch_size, self.size), self.v_init, device=device, dtype=dtype),
            "u": torch.full((batch_size, self.size), self.u_init, device=device, dtype=dtype),
            "spike_count": torch.zeros(batch_size, self.size, device=device, dtype=dtype),
        }

    def _update(self, v: torch.Tensor, u: torch.Tensor, input_current: torch.Tensor) -> tuple:
        """Single Euler integration step for the Izhikevich model.

        Uses a two-half-step approach for numerical stability:
        v += 0.5 * dt * (0.04*v^2 + 5*v + 140 - u + I)
        (repeated twice)
        u += dt * a * (b*v - u)

        Args:
            v: Current membrane potential (batch, size).
            u: Current recovery variable (batch, size).
            input_current: Input current (batch, size).

        Returns:
            Tuple of (v_new, u_new, spikes).
        """
        dt = self.dt

        # Two-step Euler method for better numerical stability
        v_new = v + 0.5 * dt * (0.04 * v * v + 5.0 * v + 140.0 - u + input_current)
        v_new = v_new + 0.5 * dt * (0.04 * v_new * v_new + 5.0 * v_new + 140.0 - u + input_current)

        # Update recovery variable
        u_new = u + dt * self.a * (self.b * v - u)

        # Detect spikes
        spikes = (v_new >= self.v_peak).float()

        # Apply reset: neurons that spiked get v = c, u = u + d
        v_new = torch.where(spikes > 0, torch.full_like(v_new, self.c), v_new)
        u_new = u_new + spikes * self.d

        return v_new, u_new, spikes

    def forward(self, input_current: torch.Tensor, state: Dict[str, torch.Tensor]):
        """Single-step Izhikevich neuron forward pass.

        Args:
            input_current: (batch, size) input current.
            state: Must contain "v" and "u" keys.

        Returns:
            Tuple of (spikes, membrane_potential, new_state).
        """
        v = state["v"]
        u = state["u"]
        spike_count = state.get("spike_count", torch.zeros_like(v))

        v_new, u_new, spikes = self._update(v, u, input_current)
        spike_count = spike_count + spikes

        new_state = {
            "v": v_new,
            "u": u_new,
            "spike_count": spike_count,
        }

        self.record_step(spikes, v_new, new_state)
        return spikes, v_new, new_state

    def extra_repr(self) -> str:
        preset_name = self.config.get("preset", "custom")
        return (
            f"size={self.size}, preset={preset_name}, "
            f"a={self.a}, b={self.b}, c={self.c}, d={self.d}"
        )
