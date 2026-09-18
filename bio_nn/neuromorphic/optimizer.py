"""Hardware-aware optimization for neuromorphic deployment.

Provides energy, latency, and throughput penalty functions that can be
added to the training loss to optimize SNNs for specific neuromorphic
hardware constraints.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class HardwareConstraints:
    """Target hardware constraints for optimization.

    Attributes:
        max_energy_per_inference: Maximum energy per inference in picojoules.
        max_latency_ms: Maximum latency in milliseconds.
        target_throughput: Target throughput in frames per second.
        max_spike_rate: Maximum allowed average firing rate (0-1).
        max_synaptic_ops: Maximum synaptic operations per time step.
        memory_budget_bytes: Maximum on-chip memory in bytes.
    """

    max_energy_per_inference: Optional[float] = None
    max_latency_ms: Optional[float] = None
    target_throughput: Optional[float] = None
    max_spike_rate: Optional[float] = None
    max_synaptic_ops: Optional[int] = None
    memory_budget_bytes: Optional[int] = None


@dataclass
class OptimizationConfig:
    """Configuration for hardware-aware optimization.

    Attributes:
        constraints: Hardware constraints to enforce.
        energy_weight: Weight of the energy penalty in the total loss.
        latency_weight: Weight of the latency penalty.
        throughput_weight: Weight of the throughput penalty.
        spike_reg_weight: Weight of the spike rate regularization.
        constraint_penalty: Penalty multiplier for hard constraint violations.
        warmup_epochs: Number of epochs before applying full penalties.
    """

    constraints: HardwareConstraints = field(default_factory=HardwareConstraints)
    energy_weight: float = 0.1
    latency_weight: float = 0.1
    throughput_weight: float = 0.05
    spike_reg_weight: float = 0.01
    constraint_penalty: float = 10.0
    warmup_epochs: int = 5


@dataclass
class OptimizationResult:
    """Result of a hardware-aware optimization step.

    Attributes:
        total_loss: Combined task + hardware loss.
        task_loss: Original task loss.
        energy_loss: Energy penalty component.
        latency_loss: Latency penalty component.
        throughput_loss: Throughput penalty component.
        spike_reg_loss: Spike regularization component.
        constraint_violations: Dict of violated constraints and their magnitudes.
    """

    total_loss: torch.Tensor
    task_loss: torch.Tensor
    energy_loss: torch.Tensor
    latency_loss: torch.Tensor
    throughput_loss: torch.Tensor
    spike_reg_loss: torch.Tensor
    constraint_violations: Dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Penalty base class
# ---------------------------------------------------------------------------


class HardwarePenalty(ABC):
    """Base class for hardware-aware penalty terms.

    Subclasses implement ``compute()`` which returns a scalar loss tensor.
    """

    @abstractmethod
    def compute(
        self,
        model: nn.Module,
        spike_history: Optional[List[torch.Tensor]] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Compute the penalty term.

        Args:
            model: The SNN model.
            spike_history: List of spike tensors from the forward pass.
            **kwargs: Additional context (e.g., batch size, time steps).

        Returns:
            Scalar loss tensor (must be differentiable).
        """
        ...


# ---------------------------------------------------------------------------
# Energy penalty
# ---------------------------------------------------------------------------


class EnergyAwarePenalty(HardwarePenalty):
    """Penalizes high-energy spike patterns.

    Energy in neuromorphic hardware is proportional to the number of
    synaptic events (spikes × active synapses). This penalty discourages
    excessive spiking and encourages sparse, efficient representations.

    Args:
        energy_per_spike: Estimated energy per synaptic event (pJ).
        synapse_count: Number of synapses per neuron (estimated).
    """

    def __init__(
        self,
        energy_per_spike: float = 1.0,
        synapse_count: Optional[int] = None,
    ):
        self.energy_per_spike = energy_per_spike
        self.synapse_count = synapse_count

    def compute(
        self,
        model: nn.Module,
        spike_history: Optional[List[torch.Tensor]] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Compute energy penalty from spike counts."""
        if not spike_history:
            return torch.tensor(0.0, requires_grad=True)

        # Total synaptic events
        total_spikes = sum(s.float().sum() for s in spike_history)

        # Estimate synapse count from model parameters
        synapses = self.synapse_count
        if synapses is None:
            total_params = sum(p.numel() for p in model.parameters())
            n_neurons = max(
                (p.shape[0] for p in model.parameters() if p.dim() >= 2),
                default=1,
            )
            synapses = max(total_params // max(n_neurons, 1), 1)

        # Energy = spikes × synapses × energy_per_spike
        energy = total_spikes * synapses * self.energy_per_spike

        # Normalize by batch and time steps
        batch_size = kwargs.get("batch_size", 1)
        timesteps = kwargs.get("timesteps", 1)
        energy = energy / max(batch_size * timesteps, 1)

        return energy


# ---------------------------------------------------------------------------
# Latency penalty
# ---------------------------------------------------------------------------


class LatencyPenalty(HardwarePenalty):
    """Penalizes late spikes in the temporal window.

    In event-driven hardware, processing latency is determined by when
    the last relevant spike occurs. This penalty encourages the network
    to make decisions early, reducing inference latency.

    Args:
        latency_per_step: Estimated time per simulation step (ms).
        target_latency: Target maximum latency (ms).
    """

    def __init__(
        self,
        latency_per_step: float = 1.0,
        target_latency: Optional[float] = None,
    ):
        self.latency_per_step = latency_per_step
        self.target_latency = target_latency

    def compute(
        self,
        model: nn.Module,
        spike_history: Optional[List[torch.Tensor]] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Compute latency penalty from spike timing."""
        if not spike_history or len(spike_history) < 2:
            return torch.tensor(0.0, requires_grad=True)

        T = len(spike_history)

        # Compute cumulative spike energy per time step
        step_energy = torch.stack(
            [s.float().sum() for s in spike_history]
        )  # (T,)

        # Find the "center of mass" of spike activity
        time_indices = torch.arange(T, dtype=torch.float32, device=step_energy.device)
        total_energy = step_energy.sum()

        if total_energy < 1e-8:
            return torch.tensor(0.0, requires_grad=True)

        # Weighted average time of spike activity
        center_of_mass = (time_indices * step_energy).sum() / total_energy
        latency = center_of_mass * self.latency_per_step

        # Penalize latency above target
        if self.target_latency is not None:
            excess = torch.relu(latency - self.target_latency)
            return excess ** 2

        # Otherwise, encourage early spiking (penalize center of mass)
        return center_of_mass / T


# ---------------------------------------------------------------------------
# Throughput penalty
# ---------------------------------------------------------------------------


class ThroughputPenalty(HardwarePenalty):
    """Penalizes patterns that reduce batch processing throughput.

    High synaptic activity or irregular spike patterns can bottleneck
    batch processing on neuromorphic hardware. This penalty encourages
    consistent, predictable activity patterns.

    Args:
        target_sparsity: Target fraction of silent neurons per step.
        consistency_weight: Weight for spike pattern consistency.
    """

    def __init__(
        self,
        target_sparsity: float = 0.8,
        consistency_weight: float = 1.0,
    ):
        self.target_sparsity = target_sparsity
        self.consistency_weight = consistency_weight

    def compute(
        self,
        model: nn.Module,
        spike_history: Optional[List[torch.Tensor]] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Compute throughput penalty from spike patterns."""
        if not spike_history or len(spike_history) < 2:
            return torch.tensor(0.0, requires_grad=True)

        # Sparsity regularization: encourage target sparsity level
        spike_rates = torch.stack(
            [s.float().mean() for s in spike_history]
        )  # (T,)
        actual_sparsity = 1.0 - spike_rates
        sparsity_loss = (actual_sparsity - self.target_sparsity).pow(2).mean()

        # Pattern consistency: penalize high variance in spike rates across time
        rate_variance = spike_rates.var()
        consistency_loss = rate_variance * self.consistency_weight

        return sparsity_loss + consistency_loss


# ---------------------------------------------------------------------------
# Hardware-aware optimizer
# ---------------------------------------------------------------------------


class HardwareAwareOptimizer:
    """Optimizes an SNN model for neuromorphic hardware constraints.

    Wraps a standard PyTorch optimizer and adds hardware-aware penalty
    terms to the loss during training.

    Args:
        model: The SNN model to optimize.
        config: Optimization configuration.
        base_optimizer: Existing optimizer to wrap. If None, creates Adam.
        lr: Learning rate (only used if creating a new optimizer).
    """

    def __init__(
        self,
        model: nn.Module,
        config: Optional[OptimizationConfig] = None,
        base_optimizer: Optional[torch.optim.Optimizer] = None,
        lr: float = 1e-3,
    ):
        self.model = model
        self.config = config or OptimizationConfig()
        self._epoch = 0

        if base_optimizer is not None:
            self.optimizer = base_optimizer
        else:
            self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        # Initialize penalty terms
        self._penalties: List[Tuple[HardwarePenalty, float]] = []

        c = self.config.constraints
        if c.max_energy_per_inference is not None:
            self._penalties.append((
                EnergyAwarePenalty(),
                self.config.energy_weight,
            ))
        if c.max_latency_ms is not None:
            self._penalties.append((
                LatencyPenalty(target_latency=c.max_latency_ms),
                self.config.latency_weight,
            ))
        if c.target_throughput is not None:
            self._penalties.append((
                ThroughputPenalty(),
                self.config.throughput_weight,
            ))

        # Always add spike regularization
        self._spike_penalty = EnergyAwarePenalty(energy_per_spike=1.0)

        self._history: List[Dict[str, Any]] = []

    def compute_hardware_loss(
        self,
        task_loss: torch.Tensor,
        spike_history: Optional[List[torch.Tensor]] = None,
        **kwargs: Any,
    ) -> OptimizationResult:
        """Compute the combined task + hardware loss.

        Args:
            task_loss: The original task loss (e.g., cross-entropy).
            spike_history: Spike tensors from the forward pass.
            **kwargs: Additional context passed to penalty functions.

        Returns:
            OptimizationResult with all loss components.
        """
        # Warmup: linearly ramp up penalties
        warmup = min(self._epoch / max(self.config.warmup_epochs, 1), 1.0)

        # Penalty terms
        energy_loss = torch.tensor(0.0, device=task_loss.device)
        latency_loss = torch.tensor(0.0, device=task_loss.device)
        throughput_loss = torch.tensor(0.0, device=task_loss.device)

        for penalty, weight in self._penalties:
            penalty_val = penalty.compute(self.model, spike_history, **kwargs)
            scaled = penalty_val * weight * warmup

            if isinstance(penalty, EnergyAwarePenalty):
                energy_loss = scaled
            elif isinstance(penalty, LatencyPenalty):
                latency_loss = scaled
            elif isinstance(penalty, ThroughputPenalty):
                throughput_loss = scaled

        # Spike rate regularization
        spike_reg = torch.tensor(0.0, device=task_loss.device)
        if spike_history:
            avg_rate = sum(s.float().sum() for s in spike_history) / max(
                sum(s.numel() for s in spike_history), 1
            )
            max_rate = self.config.constraints.max_spike_rate
            if max_rate is not None:
                spike_reg = torch.relu(avg_rate - max_rate) * self.config.spike_reg_weight * warmup
            else:
                spike_reg = avg_rate * self.config.spike_reg_weight * warmup

        # Total loss
        total_loss = task_loss + energy_loss + latency_loss + throughput_loss + spike_reg

        # Check hard constraints
        violations: Dict[str, float] = {}
        c = self.config.constraints
        if c.max_energy_per_inference is not None:
            est_energy = energy_loss.item() / max(self.config.energy_weight, 1e-8)
            if est_energy > c.max_energy_per_inference:
                violations["energy"] = est_energy - c.max_energy_per_inference
                total_loss = total_loss + violations["energy"] * self.config.constraint_penalty

        if c.max_spike_rate is not None and spike_history:
            actual_rate = sum(s.float().sum().item() for s in spike_history) / max(
                sum(s.numel() for s in spike_history), 1
            )
            if actual_rate > c.max_spike_rate:
                violations["spike_rate"] = actual_rate - c.max_spike_rate

        result = OptimizationResult(
            total_loss=total_loss,
            task_loss=task_loss,
            energy_loss=energy_loss,
            latency_loss=latency_loss,
            throughput_loss=throughput_loss,
            spike_reg_loss=spike_reg,
            constraint_violations=violations,
        )

        return result

    def step(
        self,
        task_loss: torch.Tensor,
        spike_history: Optional[List[torch.Tensor]] = None,
        grad_clip: float = 5.0,
        **kwargs: Any,
    ) -> OptimizationResult:
        """Compute hardware loss, backpropagate, and update weights.

        Args:
            task_loss: Task-specific loss tensor.
            spike_history: Spike tensors from forward pass.
            grad_clip: Maximum gradient norm for clipping.
            **kwargs: Additional context for penalty functions.

        Returns:
            OptimizationResult with all loss components.
        """
        self.optimizer.zero_grad()

        result = self.compute_hardware_loss(task_loss, spike_history, **kwargs)
        result.total_loss.backward()

        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)

        self.optimizer.step()
        self._epoch += 1

        # Record history
        record = {
            "epoch": self._epoch,
            "task_loss": result.task_loss.item(),
            "total_loss": result.total_loss.item(),
            "energy_loss": result.energy_loss.item(),
            "latency_loss": result.latency_loss.item(),
            "throughput_loss": result.throughput_loss.item(),
            "spike_reg_loss": result.spike_reg_loss.item(),
            "violations": dict(result.constraint_violations),
        }
        self._history.append(record)

        if self._epoch % 10 == 0:
            logger.info(
                "Epoch %d | Task: %.4f | Total: %.4f | Energy: %.4f | Lat: %.4f",
                self._epoch,
                record["task_loss"],
                record["total_loss"],
                record["energy_loss"],
                record["latency_loss"],
            )

        return result

    def enforce_constraints(self) -> Dict[str, bool]:
        """Check current model against hardware constraints.

        Returns:
            Dict mapping constraint names to whether they are satisfied.
        """
        status: Dict[str, bool] = {}
        c = self.config.constraints

        if c.memory_budget_bytes is not None:
            param_bytes = sum(
                p.numel() * p.element_size() for p in self.model.parameters()
            )
            status["memory"] = param_bytes <= c.memory_budget_bytes

        if c.max_synaptic_ops is not None:
            total_params = sum(p.numel() for p in self.model.parameters() if p.dim() >= 2)
            status["synaptic_ops"] = total_params <= c.max_synaptic_ops

        return status

    @property
    def epoch(self) -> int:
        """Current training epoch."""
        return self._epoch

    @property
    def history(self) -> List[Dict[str, Any]]:
        """Training history with hardware metrics."""
        return list(self._history)

    def state_dict(self) -> Dict[str, Any]:
        """Serialize optimizer state."""
        return {
            "optimizer": self.optimizer.state_dict(),
            "epoch": self._epoch,
            "config": {
                "energy_weight": self.config.energy_weight,
                "latency_weight": self.config.latency_weight,
                "throughput_weight": self.config.throughput_weight,
                "spike_reg_weight": self.config.spike_reg_weight,
                "constraint_penalty": self.config.constraint_penalty,
                "warmup_epochs": self.config.warmup_epochs,
            },
            "history": self._history,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Restore optimizer state."""
        self.optimizer.load_state_dict(state["optimizer"])
        self._epoch = state.get("epoch", 0)
        self._history = state.get("history", [])
