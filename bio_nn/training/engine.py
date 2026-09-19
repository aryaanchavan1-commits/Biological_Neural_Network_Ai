"""Training engine for spiking neural networks.

Handles the full forward pass through time, loss accumulation,
surrogate-gradient backprop, and epoch-level training loops.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..evaluation.metrics import accuracy


class TrainingEngine:
    """Core SNN training engine.

    Drives the temporal forward pass, accumulates loss across time steps,
    and calls backward with surrogate gradients.

    Args:
        model:      nn.Module whose forward(data) returns a dict with at least
                    ``"output"`` (T, batch, classes) or ``"spikes"``.
        config:     Training hyperparameters dict.
        device:     torch device.
    """

    def __init__(
        self,
        model: nn.Module,
        config: Dict[str, Any],
        device: torch.device,
    ) -> None:
        self.model = model
        self.config = config
        self.device = device

        self.epochs: int = config.get("epochs", 100)
        self.lr: float = config.get("lr", 1e-3)
        self.weight_decay: float = config.get("weight_decay", 0.0)
        self.timestep: int = config.get("timestep", 15)
        self.patience: int = config.get("patience", 15)
        self.grad_clip: float = config.get("grad_clip", 5.0)
        self.loss_fn: str = config.get("loss", "cross_entropy")
        self.log_interval: int = config.get("log_interval", 10)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="max",
            factor=config.get("lr_factor", 0.5),
            patience=config.get("lr_patience", 5),
        )

        self._init_loss_fn()

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    def _init_loss_fn(self) -> None:
        if self.loss_fn == "cross_entropy":
            self.criterion = nn.CrossEntropyLoss()
        elif self.loss_fn == "nll":
            self.criterion = nn.NLLLoss()
        elif self.loss_fn == "mse":
            self.criterion = nn.MSELoss()
        else:
            self.criterion = nn.CrossEntropyLoss()

    # ------------------------------------------------------------------
    # Forward through time
    # ------------------------------------------------------------------

    def _reset_snn_states(self) -> None:
        """Reset snntorch hidden states so computation graphs are fresh."""
        for module in self.model.modules():
            if hasattr(module, "reset_hidden"):
                module.reset_hidden()
        # Also reset BioNNModel state
        if hasattr(self.model, "reset_state"):
            self.model.reset_state()

    def _forward_snn(self, data: torch.Tensor) -> Dict[str, Any]:
        """Run the SNN for ``self.timestep`` time steps.

        Expects the model to support either:
        1. A ``forward(data, timestep)`` that returns a dict with
           ``"spike_rec"`` (list of T spike tensors) and/or ``"mem_rec"``
           (list of T membrane tensors), or
        2. A ``forward(data)`` that returns a single output, which we
           replicate T times (useful for rate-coded ANNs).

        Returns:
            dict with keys:
                spike_rec: list[Tensor] – per-step spikes (if available)
                mem_rec:   list[Tensor] – per-step membrane (if available)
                output:    Tensor (T, batch, classes) – per-step outputs
                total_spikes: int – total spike count across all steps
        """
        self._reset_snn_states()
        spike_rec: List[torch.Tensor] = []
        mem_rec: List[torch.Tensor] = []
        outputs: List[torch.Tensor] = []

        for t in range(self.timestep):
            out = self.model(data)

            if isinstance(out, dict):
                if "spikes" in out:
                    spike_rec.append(out["spikes"])
                if "membrane" in out:
                    mem_rec.append(out["membrane"])
                if "output" in out:
                    outputs.append(out["output"])
                elif "spikes" in out:
                    outputs.append(out["spikes"])
            elif isinstance(out, tuple):
                # BioNNModel.forward() returns (output,) or (output, details)
                outputs.append(out[0])
            else:
                outputs.append(out)

        # Stack outputs: (T, batch, classes)
        output_stack = torch.stack(outputs, dim=0) if outputs else None

        # Compute total spike count
        total = 0
        if spike_rec:
            total = sum(s.float().sum().item() for s in spike_rec)

        return {
            "spike_rec": spike_rec,
            "mem_rec": mem_rec,
            "output": output_stack,
            "total_spikes": total,
        }

    # ------------------------------------------------------------------
    # Single-batch step
    # ------------------------------------------------------------------

    def _train_batch(
        self, data: torch.Tensor, targets: torch.Tensor
    ) -> Dict[str, float]:
        """Run one training step: forward → loss → backward → update.

        Returns dict with loss, accuracy, spike count.
        """
        self.model.train()
        self.optimizer.zero_grad()

        # Forward through time
        fwd = self._forward_snn(data)

        output = fwd["output"]  # (T, batch, classes)
        if output is None:
            # Fallback: model returned something non-dict
            return {"loss": 0.0, "accuracy": 0.0, "total_spikes": 0}

        # Accumulate loss across time steps
        T = output.shape[0]
        total_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        for t in range(T):
            total_loss = total_loss + self.criterion(output[t], targets)

        # Average over time
        loss = total_loss / T

        loss.backward()

        # Gradient clipping
        if self.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.grad_clip
            )

        self.optimizer.step()

        # Metrics: use last time step for accuracy
        with torch.no_grad():
            preds = output[-1].argmax(dim=-1)
            acc = accuracy(preds.cpu(), targets.cpu())

        return {
            "loss": loss.item(),
            "accuracy": acc,
            "total_spikes": fwd["total_spikes"],
        }

    # ------------------------------------------------------------------
    # Epoch
    # ------------------------------------------------------------------

    def train_epoch(
        self, dataloader: DataLoader, epoch: int
    ) -> Dict[str, float]:
        """Train for one epoch. Returns aggregated metrics."""
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        total_spikes = 0
        num_batches = 0

        for batch_idx, (data, targets) in enumerate(dataloader):
            data = data.to(self.device)
            targets = targets.to(self.device)

            result = self._train_batch(data, targets)

            total_loss += result["loss"]
            total_correct += result["accuracy"] * data.size(0)
            total_samples += data.size(0)
            total_spikes += result["total_spikes"]
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        avg_acc = total_correct / max(total_samples, 1)

        return {
            "loss": avg_loss,
            "accuracy": avg_acc,
            "total_spikes": total_spikes,
            "avg_spikes_per_sample": total_spikes / max(total_samples, 1),
        }

    # ------------------------------------------------------------------
    # Evaluate
    # ------------------------------------------------------------------

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Evaluate on a dataset. Returns metrics dict."""
        self.model.eval()
        total_loss = 0.0
        all_preds: List[torch.Tensor] = []
        all_targets: List[torch.Tensor] = []
        total_spikes = 0
        num_batches = 0

        for data, targets in dataloader:
            data = data.to(self.device)
            targets = targets.to(self.device)

            fwd = self._forward_snn(data)
            output = fwd["output"]
            if output is None:
                continue

            # Use last time step
            loss = self.criterion(output[-1], targets)
            total_loss += loss.item()
            preds = output[-1].argmax(dim=-1)
            all_preds.append(preds.cpu())
            all_targets.append(targets.cpu())
            total_spikes += fwd["total_spikes"]
            num_batches += 1

        if not all_preds:
            return {"loss": 0.0, "accuracy": 0.0, "total_spikes": 0}

        preds_cat = torch.cat(all_preds)
        targets_cat = torch.cat(all_targets)
        avg_loss = total_loss / max(num_batches, 1)

        return {
            "loss": avg_loss,
            "accuracy": accuracy(preds_cat, targets_cat),
            "total_spikes": total_spikes,
        }

    # ------------------------------------------------------------------
    # Full training loop
    # ------------------------------------------------------------------

    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        test_loader: Optional[DataLoader] = None,
    ) -> Dict[str, Any]:
        """Full training loop with validation and checkpointing.

        Returns:
            dict with ``history`` (list of per-epoch dicts), ``best_val_accuracy``,
            and optional ``test_results``.
        """
        history: List[Dict[str, Any]] = []
        best_val_acc = 0.0
        best_state = None
        no_improve = 0

        for epoch in range(1, self.epochs + 1):
            t0 = time.time()

            train_metrics = self.train_epoch(train_loader, epoch)
            val_metrics = (
                self.evaluate(val_loader) if val_loader else {"accuracy": 0.0}
            )

            # Learning rate scheduling
            self.scheduler.step(val_metrics.get("accuracy", 0.0))

            elapsed = time.time() - t0

            epoch_record = {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_accuracy": train_metrics["accuracy"],
                "val_loss": val_metrics.get("loss", 0.0),
                "val_accuracy": val_metrics.get("accuracy", 0.0),
                "spikes": train_metrics["total_spikes"],
                "time": elapsed,
                "lr": self.optimizer.param_groups[0]["lr"],
            }
            history.append(epoch_record)

            if epoch % self.log_interval == 0 or epoch == 1:
                print(
                    f"Epoch {epoch:4d}/{self.epochs} | "
                    f"Train Acc: {train_metrics['accuracy']:.4f} | "
                    f"Val Acc: {val_metrics.get('accuracy', 0):.4f} | "
                    f"Loss: {train_metrics['loss']:.4f} | "
                    f"Spikes: {train_metrics['total_spikes']:.0f} | "
                    f"Time: {elapsed:.2f}s"
                )

            # Early stopping / best model tracking
            val_acc = val_metrics.get("accuracy", 0.0)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = {
                    k: v.cpu().clone() for k, v in self.model.state_dict().items()
                }
                no_improve = 0
            else:
                no_improve += 1
                if self.patience > 0 and no_improve >= self.patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

        # Restore best weights
        if best_state is not None:
            self.model.load_state_dict(best_state)
            self.model.to(self.device)

        results: Dict[str, Any] = {
            "history": history,
            "best_val_accuracy": best_val_acc,
        }

        # Optional test evaluation
        if test_loader:
            results["test_results"] = self.evaluate(test_loader)
            print(f"Test Accuracy: {results['test_results']['accuracy']:.4f}")

        return results
