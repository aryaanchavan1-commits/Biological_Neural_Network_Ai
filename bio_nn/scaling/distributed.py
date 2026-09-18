"""Distributed training utilities for BIO-NN.

Supports DataParallel, DistributedDataParallel, and gradient accumulation.
Falls back gracefully on single-GPU / CPU setups.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, DistributedSampler


def is_distributed() -> bool:
    """Check if distributed training is initialized."""
    return torch.distributed.is_available() and torch.distributed.is_initialized()


def get_rank() -> int:
    return torch.distributed.get_rank() if is_distributed() else 0


def get_world_size() -> int:
    return torch.distributed.get_world_size() if is_distributed() else 1


def is_main_process() -> bool:
    return get_rank() == 0


def setup_distributed(
    backend: str = "nccl",
    rank: int = 0,
    world_size: int = 1,
    master_addr: str = "127.0.0.1",
    master_port: str = "29500",
) -> None:
    """Initialize distributed process group.

    For single-machine multi-GPU, auto-configures env variables.
    """
    os.environ.setdefault("MASTER_ADDR", master_addr)
    os.environ.setdefault("MASTER_PORT", master_port)
    os.environ.setdefault("RANK", str(rank))
    os.environ.setdefault("WORLD_SIZE", str(world_size))

    if not torch.distributed.is_initialized():
        device_count = torch.cuda.device_count()
        if device_count > 1 and backend == "nccl":
            torch.distributed.init_process_group(backend=backend, rank=rank, world_size=world_size)
        elif backend == "gloo":
            torch.distributed.init_process_group(backend="gloo", rank=rank, world_size=world_size)


def cleanup_distributed() -> None:
    """Destroy the process group."""
    if is_distributed():
        torch.distributed.destroy_process_group()


def wrap_model(
    model: nn.Module,
    mode: str = "auto",
    device_ids: Optional[List[int]] = None,
) -> nn.Module:
    """Wrap model for distributed / parallel training.

    Args:
        model:     The nn.Module to wrap.
        mode:      One of "ddp", "dp", "auto", or "none".
        device_ids: GPU device IDs. None = all available GPUs.

    Returns:
        Wrapped model (or original if single device / mode="none").
    """
    if mode == "none":
        return model

    if mode in ("dp", "ddp") and torch.cuda.device_count() < 2:
        return model  # nothing to parallelize

    if mode == "ddp":
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        device_id = [local_rank] if device_ids is None else device_ids[:1]
        return nn.parallel.DistributedDataParallel(
            model,
            device_ids=device_id,
            find_unused_parameters=False,
        )

    if mode == "dp":
        return nn.parallel.DataParallel(model, device_ids=device_ids)

    # auto: prefer DDP if distributed, else DP if multi-GPU
    if is_distributed():
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        device_id = [local_rank] if device_ids is None else device_ids[:1]
        return nn.parallel.DistributedDataParallel(model, device_ids=device_id)
    if torch.cuda.device_count() > 1:
        return nn.parallel.DataParallel(model)

    return model


def create_distributed_sampler(
    dataset,
    num_replicas: Optional[int] = None,
    rank: Optional[int] = None,
    shuffle: bool = True,
    seed: int = 0,
) -> DistributedSampler:
    """Create a DistributedSampler for the current process group."""
    return DistributedSampler(
        dataset,
        num_replicas=num_replicas or get_world_size(),
        rank=rank or get_rank(),
        shuffle=shuffle,
        seed=seed,
    )


class GradientAccumulator:
    """Accumulate gradients over N micro-batches before stepping.

    Useful for simulating larger batch sizes on limited VRAM.

    Args:
        optimizer:   The optimizer.
        accumulation_steps: Number of micro-batches per step.
        scaler:      Optional GradScaler for mixed precision.
        max_grad_norm: Gradient clipping norm (0 = no clip).
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        accumulation_steps: int = 1,
        scaler: Optional[torch.amp.GradScaler] = None,
        max_grad_norm: float = 5.0,
    ) -> None:
        self.optimizer = optimizer
        self.accumulation_steps = max(accumulation_steps, 1)
        self.scaler = scaler
        self.max_grad_norm = max_grad_norm
        self._micro_step = 0

    @property
    def should_step(self) -> bool:
        return (self._micro_step + 1) % self.accumulation_steps == 0

    def backward(self, loss: torch.Tensor) -> None:
        """Backpropagate scaled loss (divided by accumulation steps)."""
        scaled_loss = loss / self.accumulation_steps
        if self.scaler is not None:
            self.scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()
        self._micro_step += 1

    def step(self) -> bool:
        """Step optimizer if accumulation is complete. Returns True if stepped."""
        if not self.should_step:
            return False

        if self.scaler is not None:
            self.scaler.unscale_(self.optimizer)
            if self.max_grad_norm > 0:
                nn.utils.clip_grad_norm_(
                    self._get_params(), self.max_grad_norm
                )
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            if self.max_grad_norm > 0:
                nn.utils.clip_grad_norm_(
                    self._get_params(), self.max_grad_norm
                )
            self.optimizer.step()

        self.optimizer.zero_grad()
        self._micro_step = 0
        return True

    def zero_grad(self) -> None:
        """Manually zero gradients."""
        self.optimizer.zero_grad()
        self._micro_step = 0

    def _get_params(self) -> List[torch.Tensor]:
        params = []
        for group in self.optimizer.param_groups:
            params.extend(group["params"])
        return params

    @property
    def info(self) -> Dict[str, Any]:
        return {
            "accumulation_steps": self.accumulation_steps,
            "current_micro_step": self._micro_step,
            "effective_batch_multiplier": self.accumulation_steps,
        }
