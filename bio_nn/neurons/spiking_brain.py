"""SpikingBrain2.0-style neuron model with dendritic integration."""

import math
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from .base import BaseNeuron


class SpikingBrainNeuron(BaseNeuron):
    """SpikingBrain2.0-inspired neuron with multi-timescale adaptation and burst firing.

    Combines three biological features into a single neuron model:
        1. Dual-timescale adaptation (fast Na+ inactivation + slow K+ afterhyperpolarization)
        2. Burst firing via a plateau potential mechanism
        3. Simplified dendritic integration (two compartments: soma + dendrite)

    Dynamics:
        Soma compartment:
            tau_soma * dV_soma/dt = -(V_soma - V_rest) + g_dend * (V_dend - V_soma) + I_ext
            V_dend = tau_dend * dV_dend/dt = -(V_dend - V_rest) + I_input
            if V_soma >= V_thresh + theta_fast + theta_slow: spike
            theta_fast[t+1] = beta_fast * theta_fast + b_fast * spike
            theta_slow[t+1] = beta_slow * theta_slow + b_slow * spike

        Burst detection:
            burst_counter increments when consecutive spikes are too close together
            if burst_counter >= burst_threshold: activate plateau -> raise effective threshold
            (plateau suppresses further spiking, ending the burst)

    Args:
        size: Number of neurons.
        config: Dictionary with keys:
            - tau_soma (float): Soma membrane time constant. Default 20.0.
            - tau_dend (float): Dendritic time constant. Default 30.0.
            - g_dend (float): Dendro-somatic coupling conductance. Default 0.3.
            - threshold (float): Base spike threshold. Default 1.0.
            - V_rest (float): Resting potential. Default 0.0.
            - tau_fast_adapt (float): Fast adaptation time constant. Default 10.0.
            - tau_slow_adapt (float): Slow adaptation time constant. Default 200.0.
            - b_fast (float): Fast adaptation increment. Default 0.02.
            - b_slow (float): Slow adaptation increment. Default 0.005.
            - burst_threshold (int): Spike count within burst_window to trigger burst. Default 3.
            - burst_window (int): Steps to count for burst detection. Default 5.
            - plateau_duration (int): Steps of plateau elevation after burst. Default 3.
            - plateau_strength (float): Extra threshold during plateau. Default 0.5.
            - refractory_period (int): Refractory steps. Default 0.
            - spike_grad (str): Surrogate gradient. Default "fast_sigmoid".
    """

    def __init__(self, size: int, config: Optional[Dict[str, Any]] = None):
        super().__init__(size, config)

        self.tau_soma: float = self.config.get("tau_soma", 20.0)
        self.tau_dend: float = self.config.get("tau_dend", 30.0)
        self.g_dend: float = self.config.get("g_dend", 0.3)
        self.threshold: float = self.config.get("threshold", 1.0)
        self.V_rest: float = self.config.get("V_rest", 0.0)

        self.tau_fast_adapt: float = self.config.get("tau_fast_adapt", 10.0)
        self.tau_slow_adapt: float = self.config.get("tau_slow_adapt", 200.0)
        self.b_fast: float = self.config.get("b_fast", 0.02)
        self.b_slow: float = self.config.get("b_slow", 0.005)

        self.burst_threshold: int = self.config.get("burst_threshold", 3)
        self.burst_window: int = self.config.get("burst_window", 5)
        self.plateau_duration: int = self.config.get("plateau_duration", 3)
        self.plateau_strength: float = self.config.get("plateau_strength", 0.5)
        self.refractory_period: int = self.config.get("refractory_period", 0)

        self.beta_soma = math.exp(-1.0 / self.tau_soma)
        self.beta_dend = math.exp(-1.0 / self.tau_dend)
        self.beta_fast = math.exp(-1.0 / self.tau_fast_adapt)
        self.beta_slow = math.exp(-1.0 / self.tau_slow_adapt)

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
            "v_soma": torch.full((batch_size, self.size), self.V_rest, device=device, dtype=dtype),
            "v_dend": torch.full((batch_size, self.size), self.V_rest, device=device, dtype=dtype),
            "adapt_fast": torch.zeros(batch_size, self.size, device=device, dtype=dtype),
            "adapt_slow": torch.zeros(batch_size, self.size, device=device, dtype=dtype),
            "spike_count": torch.zeros(batch_size, self.size, device=device, dtype=dtype),
            "recent_spikes": torch.zeros(batch_size, self.size, device=device, dtype=dtype),
            "plateau_counter": torch.zeros(batch_size, self.size, device=device, dtype=dtype),
            "refractory_counter": torch.zeros(batch_size, self.size, device=device, dtype=torch.long),
        }

    def forward(self, input_current: torch.Tensor, state: Dict[str, torch.Tensor]):
        """Single-step SpikingBrain forward pass.

        Args:
            input_current: (batch, size) input current to the soma.
            state: Must contain "v_soma", "v_dend", "adapt_fast", "adapt_slow",
                   "recent_spikes", "plateau_counter".

        Returns:
            Tuple of (spikes, membrane_potential, new_state).
        """
        v_soma = state["v_soma"]
        v_dend = state["v_dend"]
        adapt_fast = state["adapt_fast"]
        adapt_slow = state["adapt_slow"]
        recent_spikes = state["recent_spikes"]
        plateau_counter = state["plateau_counter"]
        spike_count = state.get("spike_count", torch.zeros_like(v_soma))
        refractory_counter = state.get(
            "refractory_counter",
            torch.zeros(v_soma.shape[0], self.size, device=v_soma.device, dtype=torch.long),
        )

        refractory_mask = refractory_counter > 0

        # Dendritic compartment update
        dend_current = input_current
        v_dend_new = self.beta_dend * (v_dend - self.V_rest) + dend_current * (1.0 - self.beta_dend) + self.V_rest

        # Somatic compartment update with dendro-somatic coupling
        coupling = self.g_dend * (v_dend_new - v_soma)
        v_soma_new = (
            self.beta_soma * (v_soma - self.V_rest)
            + input_current * (1.0 - self.beta_soma)
            + coupling * (1.0 - self.beta_soma)
            + self.V_rest
        )

        # Mask refractory neurons
        v_soma_new = torch.where(refractory_mask, v_soma, v_soma_new)
        v_dend_new = torch.where(refractory_mask, v_dend, v_dend_new)

        # Adaptive threshold (fast + slow)
        effective_threshold = self.threshold + adapt_fast + adapt_slow

        # Plateau suppression: if plateau active, raise threshold further
        plateau_active = plateau_counter > 0
        effective_threshold = torch.where(plateau_active, effective_threshold + self.plateau_strength, effective_threshold)

        # Spike generation
        spike_input = self.surrogate_grad(v_soma_new - effective_threshold)
        spikes = (spike_input > 0.5).float()

        # Burst detection: count recent spikes within burst_window
        recent_spikes_new = recent_spikes + spikes
        recent_spikes_new = torch.where(
            spikes > 0,
            recent_spikes_new,
            torch.clamp(recent_spikes_new - 1.0 / self.burst_window, min=0),
        )

        burst_detected = recent_spikes_new >= self.burst_threshold
        # Activate plateau when burst detected and not already active
        plateau_counter_new = torch.where(
            burst_detected & ~plateau_active,
            torch.full_like(plateau_counter, self.plateau_duration),
            torch.clamp(plateau_counter - 1, min=0),
        )

        # Reset mechanism
        if self.config.get("reset_mechanism", "subtract") == "subtract":
            v_soma_new = v_soma_new - spikes * self.threshold
            v_dend_new = v_dend_new - spikes * self.threshold
        else:
            v_soma_new = torch.where(spikes > 0, torch.zeros_like(v_soma_new), v_soma_new)
            v_dend_new = torch.where(spikes > 0, torch.zeros_like(v_dend_new), v_dend_new)

        # Update adaptation variables
        adapt_fast_new = self.beta_fast * adapt_fast + (1.0 - self.beta_fast) * self.b_fast * spikes
        adapt_slow_new = self.beta_slow * adapt_slow + (1.0 - self.beta_slow) * self.b_slow * spikes

        # Update refractory counter
        refractory_counter = torch.where(
            spikes > 0,
            torch.full_like(refractory_counter, self.refractory_period),
            torch.clamp(refractory_counter - 1, min=0),
        )

        spike_count = spike_count + spikes

        new_state = {
            "v_soma": v_soma_new,
            "v_dend": v_dend_new,
            "adapt_fast": adapt_fast_new,
            "adapt_slow": adapt_slow_new,
            "spike_count": spike_count,
            "recent_spikes": recent_spikes_new,
            "plateau_counter": plateau_counter_new,
            "refractory_counter": refractory_counter,
            "effective_threshold": effective_threshold,
        }

        self.record_step(spikes, v_soma_new, new_state)
        return spikes, v_soma_new, new_state

    def extra_repr(self) -> str:
        return (
            f"size={self.size}, tau_soma={self.tau_soma}, tau_dend={self.tau_dend}, "
            f"g_dend={self.g_dend}, threshold={self.threshold}, "
            f"tau_fast={self.tau_fast_adapt}, tau_slow={self.tau_slow_adapt}, "
            f"burst_threshold={self.burst_threshold}, plateau_dur={self.plateau_duration}"
        )
