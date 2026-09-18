"""Dual-pathway Leaky Integrate-and-Fire neuron model (CATFormer-style)."""

import math
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from .base import BaseNeuron


class DualLIFNeuron(BaseNeuron):
    """Dual-pathway LIF neuron with fast and slow membrane dynamics.

    Two parallel pathways integrate input independently:
        - Fast pathway: short tau, responds to rapid transients
        - Slow pathway: long tau, tracks sustained currents

    Cross-pathway coupling allows each pathway to influence the other,
    enabling richer temporal dynamics than a single LIF. The effective
    membrane potential is a gated combination of both pathways.

    Spike generation uses a shared adaptive threshold.

    Dynamics:
        V_fast[t+1] = beta_fast * V_fast + I * (1 - beta_fast)
                       + w_cross * sigmoid(V_slow - V_fast)
        V_slow[t+1] = beta_slow * V_slow + I * (1 - beta_slow)
                       + w_cross * sigmoid(V_fast - V_slow)
        V_eff = alpha * V_fast + (1 - alpha) * V_slow
        if V_eff >= V_thresh + theta_adapt: spike
        theta_adapt[t+1] = beta_adapt * theta_adapt + b * spike

    Args:
        size: Number of neurons.
        config: Dictionary with keys:
            - tau_fast (float): Fast pathway time constant. Default 8.0.
            - tau_slow (float): Slow pathway time constant. Default 50.0.
            - threshold (float): Base spike threshold. Default 1.0.
            - V_rest (float): Resting potential. Default 0.0.
            - alpha (float): Mixing weight for effective potential [0,1]. Default 0.6.
            - w_cross (float): Cross-pathway coupling strength. Default 0.15.
            - tau_adapt (float): Adaptation time constant. Default 100.0.
            - adapt_increment (float): Adaptation increment per spike. Default 0.01.
            - reset_mechanism (str): "subtract" or "zero". Default "subtract".
            - refractory_period (int): Refractory steps. Default 0.
            - spike_grad (str): Surrogate gradient. Default "fast_sigmoid".
    """

    def __init__(self, size: int, config: Optional[Dict[str, Any]] = None):
        super().__init__(size, config)

        self.tau_fast: float = self.config.get("tau_fast", 8.0)
        self.tau_slow: float = self.config.get("tau_slow", 50.0)
        self.threshold: float = self.config.get("threshold", 1.0)
        self.V_rest: float = self.config.get("V_rest", 0.0)
        self.alpha: float = self.config.get("alpha", 0.6)
        self.w_cross: float = self.config.get("w_cross", 0.15)
        self.tau_adapt: float = self.config.get("tau_adapt", 100.0)
        self.adapt_increment: float = self.config.get("adapt_increment", 0.01)
        self.reset_mechanism: str = self.config.get("reset_mechanism", "subtract")
        self.refractory_period: int = self.config.get("refractory_period", 0)

        self.beta_fast = math.exp(-1.0 / self.tau_fast)
        self.beta_slow = math.exp(-1.0 / self.tau_slow)
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
            "v_fast": torch.full((batch_size, self.size), self.V_rest, device=device, dtype=dtype),
            "v_slow": torch.full((batch_size, self.size), self.V_rest, device=device, dtype=dtype),
            "adaptation": torch.zeros(batch_size, self.size, device=device, dtype=dtype),
            "spike_count": torch.zeros(batch_size, self.size, device=device, dtype=dtype),
            "refractory_counter": torch.zeros(batch_size, self.size, device=device, dtype=torch.long),
        }

    def forward(self, input_current: torch.Tensor, state: Dict[str, torch.Tensor]):
        """Single-step dual-pathway LIF forward pass.

        Args:
            input_current: (batch, size) input current.
            state: Must contain "v_fast", "v_slow", "adaptation".

        Returns:
            Tuple of (spikes, membrane_potential, new_state).
        """
        v_fast = state["v_fast"]
        v_slow = state["v_slow"]
        theta_adapt = state["adaptation"]
        spike_count = state.get("spike_count", torch.zeros_like(v_fast))
        refractory_counter = state.get(
            "refractory_counter",
            torch.zeros(v_fast.shape[0], self.size, device=v_fast.device, dtype=torch.long),
        )

        refractory_mask = refractory_counter > 0

        # Cross-pathway coupling terms
        cross_fast = self.w_cross * torch.sigmoid(v_slow - v_fast)
        cross_slow = self.w_cross * torch.sigmoid(v_fast - v_slow)

        # Update fast pathway
        v_fast_new = (
            self.beta_fast * (v_fast - self.V_rest)
            + input_current * (1.0 - self.beta_fast)
            + cross_fast
            + self.V_rest
        )

        # Update slow pathway
        v_slow_new = (
            self.beta_slow * (v_slow - self.V_rest)
            + input_current * (1.0 - self.beta_slow)
            + cross_slow
            + self.V_rest
        )

        # Mask refractory neurons
        v_fast_new = torch.where(refractory_mask, v_fast, v_fast_new)
        v_slow_new = torch.where(refractory_mask, v_slow, v_slow_new)

        # Effective membrane potential
        v_eff = self.alpha * v_fast_new + (1.0 - self.alpha) * v_slow_new

        # Adaptive threshold
        effective_threshold = self.threshold + theta_adapt

        # Spike generation
        spike_input = self.surrogate_grad(v_eff - effective_threshold)
        spikes = (spike_input > 0.5).float()

        # Reset mechanism
        if self.reset_mechanism == "subtract":
            v_fast_new = v_fast_new - spikes * self.threshold
            v_slow_new = v_slow_new - spikes * self.threshold
        elif self.reset_mechanism == "zero":
            v_fast_new = torch.where(spikes > 0, torch.zeros_like(v_fast_new), v_fast_new)
            v_slow_new = torch.where(spikes > 0, torch.zeros_like(v_slow_new), v_slow_new)

        # Update adaptation
        theta_adapt_new = self.beta_adapt * theta_adapt + (1.0 - self.beta_adapt) * self.adapt_increment * spikes

        # Update refractory counter
        refractory_counter = torch.where(
            spikes > 0,
            torch.full_like(refractory_counter, self.refractory_period),
            torch.clamp(refractory_counter - 1, min=0),
        )

        spike_count = spike_count + spikes

        new_state = {
            "v_fast": v_fast_new,
            "v_slow": v_slow_new,
            "adaptation": theta_adapt_new,
            "spike_count": spike_count,
            "refractory_counter": refractory_counter,
            "effective_threshold": effective_threshold,
            "v_eff": v_eff,
        }

        self.record_step(spikes, v_eff, new_state)
        return spikes, v_eff, new_state

    def extra_repr(self) -> str:
        return (
            f"size={self.size}, tau_fast={self.tau_fast}, tau_slow={self.tau_slow}, "
            f"alpha={self.alpha}, w_cross={self.w_cross}, threshold={self.threshold}, "
            f"tau_adapt={self.tau_adapt}, reset={self.reset_mechanism}"
        )
