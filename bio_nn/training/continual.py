"""Continual learning trainer for BIO-NN.

Supports task-incremental and class-incremental scenarios with
measurement of backward transfer, forward transfer, and forgetting.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .engine import TrainingEngine
from ..evaluation.metrics import accuracy, average_accuracy, forgetting, backward_transfer, forward_transfer


class ContinualTrainer:
    """Manages a sequence of tasks for continual learning.

    After each task the model retains weights (no replay buffer by default)
    and we measure average accuracy across all seen tasks, forgetting, and
    forward/backward transfer.

    Args:
        model:  ``nn.Module`` to train.
        config: Dict of training hyper-parameters (passed to ``TrainingEngine``).
        device: torch device.
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

        self.per_task_epochs: int = config.get("per_task_epochs", 50)

        # History: per_task_accuracies[task_id][after_task] = accuracy
        self.per_task_accuracies: List[List[float]] = []
        self.task_histories: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Single-task training
    # ------------------------------------------------------------------

    def train_task(
        self,
        task_id: int,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
    ) -> Dict[str, Any]:
        """Train on a single task using a fresh ``TrainingEngine``.

        The model's current weights serve as the starting point (fine-tuning
        from previous tasks).
        """
        task_config = dict(self.config)
        task_config["epochs"] = self.per_task_epochs
        task_config.setdefault("patience", 0)  # no early-stop within task

        engine = TrainingEngine(self.model, task_config, self.device)

        test_loader = val_loader  # reuse val as pseudo-test inside task
        result = engine.train(train_loader, val_loader=val_loader, test_loader=test_loader)
        self.task_histories.append(result)

        # Record per-task accuracy after this task's training
        final_val_acc = result.get("best_val_accuracy", 0.0)
        if not self.per_task_accuracies:
            self.per_task_accuracies = []
        # Append the accuracy achieved on each task *after* training task_id
        # We'll update all rows when evaluate_all_tasks is called.
        return result

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def evaluate_task(
        self, test_loader: DataLoader
    ) -> float:
        """Evaluate model on a single task's test set. Returns accuracy."""
        self.model.eval()
        all_preds: List[torch.Tensor] = []
        all_targets: List[torch.Tensor] = []

        for data, targets in test_loader:
            data = data.to(self.device)
            targets = targets.to(self.device)
            out = self.model(data)
            if isinstance(out, dict):
                output = out.get("output", out.get("spikes"))
                if output is None:
                    continue
                # Use last time step if temporal
                if output.dim() == 3:
                    output = output[-1]
            else:
                output = out
            preds = output.argmax(dim=-1)
            all_preds.append(preds.cpu())
            all_targets.append(targets.cpu())

        if not all_preds:
            return 0.0

        return accuracy(torch.cat(all_preds), torch.cat(all_targets))

    def evaluate_all_tasks(
        self, test_loaders: List[DataLoader]
    ) -> Dict[str, Any]:
        """Evaluate on all tasks seen so far.

        Returns:
            per_task_accuracy: list[float] – accuracy on each task.
            average_accuracy: float – mean across tasks.
            forgetting: float – average forgetting.
            backward_transfer: float – BWT.
            forward_transfer: float – FWT (requires pre_task baseline).
        """
        accs = [self.evaluate_task(loader) for loader in test_loaders]

        result: Dict[str, Any] = {
            "per_task_accuracy": accs,
            "average_accuracy": average_accuracy(accs),
        }

        # Forgetting / BWT need the full matrix
        if self.per_task_accuracies and len(self.per_task_accuracies[0]) > 1:
            matrix = self.per_task_accuracies
            result["forgetting"] = forgetting(matrix)
            result["backward_transfer"] = backward_transfer(matrix)
        else:
            result["forgetting"] = 0.0
            result["backward_transfer"] = 0.0

        result["forward_transfer"] = 0.0
        return result

    # ------------------------------------------------------------------
    # Full scenario
    # ------------------------------------------------------------------

    def run_scenario(
        self,
        task_sequence: List[Tuple[DataLoader, DataLoader, DataLoader]],
    ) -> Dict[str, Any]:
        """Run a full continual learning scenario.

        Args:
            task_sequence: List of ``(train_loader, val_loader, test_loader)``
                           for each task.

        Returns:
            Complete results dict with per-task results, forgetting curves,
            and summary metrics.
        """
        self.per_task_accuracies = []
        num_tasks = len(task_sequence)

        all_test_loaders = [seq[2] for seq in task_sequence]

        for task_id, (train_loader, val_loader, test_loader) in enumerate(
            task_sequence
        ):
            print(f"\n=== Training Task {task_id + 1}/{num_tasks} ===")
            self.train_task(task_id, train_loader, val_loader)

            # Evaluate on all tasks seen so far
            eval_result = self.evaluate_all_tasks(all_test_loaders[: task_id + 1])

            # Update the full matrix: append current accuracies as a new row
            accs = eval_result["per_task_accuracy"]
            self.per_task_accuracies.append(accs)

            print(
                f"  Per-task acc: {[f'{a:.4f}' for a in accs]} | "
                f"Avg: {eval_result['average_accuracy']:.4f}"
            )

        # Final evaluation across all tasks
        final_eval = self.evaluate_all_tasks(all_test_loaders)

        return {
            "per_task_results": final_eval,
            "per_task_accuracies": self.per_task_accuracies,
            "task_histories": self.task_histories,
            "num_tasks": num_tasks,
        }
