"""Mixed precision training for BIO-NN.

Supports FP16/BF16 with dynamic loss scaling via PyTorch's native AMP.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Dict, Optional

import torch
import torch.nn as nn


class MixedPrecisionTrainer:
    """Wraps a model + optimizer for automatic mixed precision training.

    Args:
        model:       The nn.Module to train.
        optimizer:   A PyTorch optimizer.
        dtype:       Target dtype - "fp16", "bf16", or "fp32" (no-op).
        init_scale:  Initial loss scale for GradScaler (fp16 only).
        growth_interval: Steps between scale growth checks (fp16 only).
        enabled:     Force disable AMP (e.g. on CPU without bf16).
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        dtype: str = "fp16",
        init_scale: float = 2.**16,
        growth_interval: int = 2000,
        enabled: Optional[bool] = None,
    ) -> None:
        self.model = model
        self.optimizer = optimizer

        self.dtype = self._resolve_dtype(dtype)
        self.use_scaler = (self.dtype == torch.float16)
        self.enabled = enabled if enabled is not None else self.dtype != torch.float32

        if self.use_scaler and self.enabled:
            self.scaler = torch.amp.GradScaler(
                "cuda",
                init_scale=init_scale,
                growth_interval=growth_interval,
            )
        else:
            self.scaler = None

    @staticmethod
    def _resolve_dtype(dtype: str) -> torch.dtype:
        mapping = {
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
            "fp32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        if dtype.lower() not in mapping:
            raise ValueError(f"Unsupported dtype: {dtype!r}. Use fp16, bf16, or fp32.")
        return mapping[dtype.lower()]

    @contextmanager
    def autocast(self):
        """Context manager for autocast region."""
        if not self.enabled:
            yield
            return
        with torch.amp.autocast(
            device_type="cuda",
            dtype=self.dtype,
        ):
            yield

    def train_step(
        self,
        data: torch.Tensor,
        targets: torch.Tensor,
        loss_fn: nn.Module,
    ) -> Dict[str, float]:
        """Run one mixed-precision training step.

        Returns dict with 'loss' and 'scale' (if grad scaler active).
        """
        self.model.train()
        self.optimizer.zero_grad()

        with self.autocast():
            output = self.model(data)
            if isinstance(output, dict):
                output = output.get("output", output.get("spikes", output))
            loss = loss_fn(output, targets)

        if self.scaler is not None:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            return {"loss": loss.item(), "scale": self.scaler.get_scale()}
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            self.optimizer.step()
            return {"loss": loss.item()}

    @torch.no_grad()
    def eval_step(
        self,
        data: torch.Tensor,
        targets: torch.Tensor,
        loss_fn: nn.Module,
    ) -> Dict[str, float]:
        """Run one mixed-precision eval step (no grads, no scaler)."""
        self.model.eval()
        with self.autocast():
            output = self.model(data)
            if isinstance(output, dict):
                output = output.get("output", output.get("spikes", output))
            loss = loss_fn(output, targets)
        return {"loss": loss.item()}

    def state_dict(self) -> Dict[str, Any]:
        """Return serializable state (scaler + optimizer)."""
        state: Dict[str, Any] = {"optimizer": self.optimizer.state_dict()}
        if self.scaler is not None:
            state["scaler"] = self.scaler.state_dict()
        return state

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Restore from a state dict produced by :meth:`state_dict`."""
        self.optimizer.load_state_dict(state["optimizer"])
        if self.scaler is not None and "scaler" in state:
            self.scaler.load_state_dict(state["scaler"])

    @property
    def info(self) -> Dict[str, Any]:
        """Return a summary of the current mixed precision config."""
        return {
            "dtype": str(self.dtype),
            "enabled": self.enabled,
            "use_scaler": self.use_scaler,
            "current_scale": self.scaler.get_scale() if self.scaler else None,
        }
