"""Standard evaluation metrics for classification.

All functions accept plain ``torch.Tensor`` or ``numpy.ndarray`` inputs
and return Python floats.
"""

from __future__ import annotations

import numpy as np
import torch
from typing import Sequence, Union


def _to_numpy(x: Union[torch.Tensor, np.ndarray, Sequence]) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


# ------------------------------------------------------------------
# Basic
# ------------------------------------------------------------------


def accuracy(predictions, targets) -> float:
    """Top-1 accuracy."""
    pred = _to_numpy(predictions)
    tgt = _to_numpy(targets)
    if pred.ndim > 1:
        pred = pred.argmax(axis=-1)
    return float(np.mean(pred == tgt))


def balanced_accuracy(predictions, targets) -> float:
    """Balanced (macro-averaged per-class recall)."""
    from sklearn.metrics import balanced_accuracy_score
    pred = _to_numpy(predictions)
    tgt = _to_numpy(targets)
    if pred.ndim > 1:
        pred = pred.argmax(axis=-1)
    return float(balanced_accuracy_score(tgt, pred))


def precision(predictions, targets, average: str = "macro") -> float:
    from sklearn.metrics import precision_score
    pred = _to_numpy(predictions)
    tgt = _to_numpy(targets)
    if pred.ndim > 1:
        pred = pred.argmax(axis=-1)
    return float(precision_score(tgt, pred, average=average, zero_division=0))


def recall(predictions, targets, average: str = "macro") -> float:
    from sklearn.metrics import recall_score
    pred = _to_numpy(predictions)
    tgt = _to_numpy(targets)
    if pred.ndim > 1:
        pred = pred.argmax(axis=-1)
    return float(recall_score(tgt, pred, average=average, zero_division=0))


def f1_score(predictions, targets, average: str = "macro") -> float:
    from sklearn.metrics import f1_score as sk_f1
    pred = _to_numpy(predictions)
    tgt = _to_numpy(targets)
    if pred.ndim > 1:
        pred = pred.argmax(axis=-1)
    return float(sk_f1(tgt, pred, average=average, zero_division=0))


def confusion_matrix(predictions, targets) -> np.ndarray:
    from sklearn.metrics import confusion_matrix as sk_cm
    pred = _to_numpy(predictions)
    tgt = _to_numpy(targets)
    if pred.ndim > 1:
        pred = pred.argmax(axis=-1)
    return sk_cm(tgt, pred)


# ------------------------------------------------------------------
# Continual learning metrics
# ------------------------------------------------------------------


def average_accuracy(per_task_accuracies) -> float:
    """Mean accuracy across all tasks.

    Args:
        per_task_accuracies: list[float] or 2-D matrix (tasks × evaluation points).
    """
    accs = _to_numpy(per_task_accuracies)
    if accs.ndim == 2:
        # Use the last column (accuracy after all tasks trained)
        accs = accs[:, -1]
    return float(accs.mean())


def forgetting(per_task_accuracies) -> float:
    """Average forgetting: mean over tasks of (best_acc − final_acc).

    Expects a 2-D matrix of shape (num_tasks, num_eval_points) where
    ``matrix[t][e]`` is the accuracy on task *t* after training task *e*.
    """
    m = _to_numpy(per_task_accuracies)
    if m.ndim < 2 or m.shape[1] < 2:
        return 0.0

    num_tasks = m.shape[0]
    forget_vals = []
    for t in range(num_tasks):
        best = m[t, : t + 1].max()
        final = m[t, -1]
        forget_vals.append(best - final)
    return float(np.mean(forget_vals))


def backward_transfer(per_task_accuracies) -> float:
    """Backward transfer: mean over tasks of (final_acc − acc_after_task).

    Negative BWT indicates catastrophic forgetting.
    """
    m = _to_numpy(per_task_accuracies)
    if m.ndim < 2 or m.shape[1] < 2:
        return 0.0

    num_tasks = m.shape[0]
    bwt_vals = []
    for t in range(num_tasks):
        acc_after_task = m[t, t]
        final_acc = m[t, -1]
        bwt_vals.append(final_acc - acc_after_task)
    return float(np.mean(bwt_vals))


def forward_transfer(per_task_accuracies) -> float:
    """Forward transfer: average improvement on task *t* from training earlier tasks.

    ``matrix[t][t-1]`` (accuracy on task t before training it) vs baseline 0.
    """
    m = _to_numpy(per_task_accuracies)
    if m.ndim < 2 or m.shape[1] < 2:
        return 0.0

    num_tasks = m.shape[0]
    fwt_vals = []
    for t in range(1, num_tasks):
        # Accuracy on task t after training task t-1
        acc_before = m[t, t - 1]
        fwt_vals.append(acc_before)
    return float(np.mean(fwt_vals)) if fwt_vals else 0.0
