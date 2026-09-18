"""Adaptive Exponential Integrate-and-Fire (AdEx) neuron model."""

import math
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from .base import BaseNeuron


class AdExNeuron(BaseNeuron):
    """Adaptive Exponential Integrate-and-Fire neuron.

    The AdEx model combines exponential spike initiation with an adaptation
    current, capturing both the sharp onset of spikes and spike-frequency
    adaptation seen in biological neurons.

    Dynamics:
        tau_mem * dV/dt = -(V - V_rest) + delta_T * exp((V - V_thresh) / delta_T) - R * w + R * I
        tau_w * dw/dt = a * (V - V_rest) - w
        if V >= V_peak: V = V_reset, w = w + b

    The exponential term produces sharp spikes when V approaches V_thresh,
    while the adaptation variable w provides negative feedback that controls
    firing rate and enables burst/singleton mode transitions.

    Burst mode: large b values cause strong post-spike adaptation that
    creates burst-pause patterns. Small b produces regular spiking.

    Args:
        size: Number of neurons.
        config: Dictionary with keys:
            - tau_mem (float): Membrane time constant. Default 20.0.
            - tau_w (float): Adaptation time constant. Default 30.0.
            - V_rest (float): Resting potential. Default -70.0 (mV scale).
            - V_thresh (float): Threshold for exponential term. Default -50.0.
            - V_reset (float): Reset potential after spike. Default -70.0.
            - V_peak (float): Peak voltage for spike detection. Default 20.0.
            - delta_T (float): Slope factor for exponential term. Default 2.0.
            - a (float): Adaptation coupling parameter. Default 4.0 (nS).
            - b (float): Spike-triggered adaptation increment. Default 0.0805 (nA).
            - R (float): Membrane resistance. Default 100.0 (MOhm).
            - I_ext (float): External bias current. Default 0.0.
            - dt (float): Time step for integration. Default 1.0 (ms).
            - spike_grad (str): Surrogate gradient. Default "fast_sigmoid".
    """

    def __init__(self, size: int, config: Optional[Dict[str, Any]] = None):
        super().__init__(size, config)

        self.tau_mem: float = self.config.get("tau_mem", 20.0)
        self.tau_w: float = self.config.get("tau_w", 30.0)
        self.V_rest: float = self.config.get("V_rest", -70.0)
        self.V_thresh: float = self.config.get("V_thresh", -50.0)
        self.V_reset: float = self.config.get("V_reset", -70.0)
        self.V_peak: float = self.config.get("V_peak", 20.0)
        self.delta_T: float = self.config.get("delta_T", 2.0)
        self.a: float = self.config.get("a", 4.0)
        self.b: float = self.config.get("b", 0.0805)
        self.R: float = self.config.get("R", 100.0)
        self.I_ext: float = self.config.get("I_ext", 0.0)
        self.dt: float = self.config.get("dt", 1.0)

        self.beta = math.exp(-self.dt / self.tau_mem)
        self.beta_w = math.exp(-self.dt / self.tau_w)

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
            "v": torch.full((batch_size, self.size), self.V_rest, device=device, dtype=dtype),
            "w": torch.zeros(batch_size, self.size, device=device, dtype=dtype),
            "spike_count": torch.zeros(batch_size, self.size, device=device, dtype=dtype),
        }

    def forward(self, input_current: torch.Tensor, state: Dict[str, torch.Tensor]):
        """Single-step AdEx forward pass.

        Args:
            input_current: (batch, size) input current.
            state: Must contain "v" and "w".

        Returns:
            Tuple of (spikes, membrane_potential, new_state).
        """
        v = state["v"]
        w = state["w"]
        spike_count = state.get("spike_count", torch.zeros_like(v))

        dt = self.dt
        R = self.R

        # Exponential spike initiation term
        exp_term = self.delta_T * torch.exp((v - self.V_thresh) / self.delta_T)

        # Membrane potential update
        # tau_mem * dv/dt = -(V - V_rest) + delta_T * exp((V - V_thresh)/delta_T) - R*w + R*I
        dv = (-(v - self.V_rest) + exp_term - R * w + R * (input_current + self.I_ext)) / self.tau_mem
        v_new = v + dt * dv

        # Adaptation variable update
        # tau_w * dw/dt = a * (V - V_rest) - w
        dw = (self.a * (v - self.V_rest) - w) / self.tau_w
        w_new = w + dt * dw

        # Spike detection via surrogate gradient
        spike_input = self.surrogate_grad(v_new - self.V_peak)
        spikes = (spike_input > 0.5).float()

        # Hard reset: neurons that spike get v = V_reset, w = w + b
        v_new = torch.where(spikes > 0, torch.full_like(v_new, self.V_reset), v_new)
        w_new = w_new + spikes * self.b

        spike_count = spike_count + spikes

        new_state = {
            "v": v_new,
            "w": w_new,
            "spike_count": spike_count,
        }

        self.record_step(spikes, v_new, new_state)
        return spikes, v_new, new_state

    def extra_repr(self) -> str:
        return (
            f"size={self.size}, tau_mem={self.tau_mem}, tau_w={self.tau_w}, "
            f"V_rest={self.V_rest}, V_thresh={self.V_thresh}, V_reset={self.V_reset}, "
            f"delta_T={self.delta_T}, a={self.a}, b={self.b}, R={self.R}"
        )
