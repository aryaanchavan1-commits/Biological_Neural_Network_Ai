"""Continual learning visualization for BIO-NN."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


def plot_continual_accuracy(
    per_task_accuracies,
    title: str = "Continual Learning Accuracy",
    save_path: str | None = None,
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Plot per-task accuracy after each task.

    Args:
        per_task_accuracies: 2D array (n_tasks_seen, n_tasks) where entry [i][j]
            is accuracy of task j after learning task i.
        title: Plot title.
        save_path: If given, save figure.
        ax: Optional matplotlib axes.

    Returns:
        matplotlib Figure.
    """
    acc = np.asarray(per_task_accuracies, dtype=float)
    n_tasks_seen, n_tasks = acc.shape

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(10, 5))
    else:
        fig = ax.get_figure()

    cmap = plt.cm.plasma
    for t in range(n_tasks):
        color = cmap(t / max(n_tasks - 1, 1))
        task_acc = acc[:, t]
        valid = task_acc > 0
        if valid.any():
            ax.plot(np.where(valid)[0], task_acc[valid], linewidth=1.2, color=color, marker="o", markersize=3, label=f"Task {t}")

    avg_acc = np.array([acc[i, :i + 1].mean() for i in range(n_tasks_seen)])
    ax.plot(range(n_tasks_seen), avg_acc, linewidth=2, color="white", linestyle="--", label="Average", zorder=10)

    ax.set_facecolor("#0e0e12")
    ax.set_xlabel("Tasks Seen")
    ax.set_ylabel("Accuracy")
    ax.set_title(title)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=7, framealpha=0.3, labelcolor="white", ncol=2)

    fig.patch.set_facecolor("#0e0e12")
    ax.tick_params(colors="#888888")
    for spine in ax.spines.values():
        spine.set_color("#333333")

    if own_fig:
        fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    return fig


def plot_forgetting_curve(
    per_task_accuracies,
    title: str = "Forgetting Curve",
    save_path: str | None = None,
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Plot forgetting over time.

    For each task, forgetting = max accuracy - current accuracy.

    Args:
        per_task_accuracies: 2D array (n_tasks_seen, n_tasks).
        title: Plot title.
        save_path: If given, save figure.
        ax: Optional matplotlib axes.

    Returns:
        matplotlib Figure.
    """
    acc = np.asarray(per_task_accuracies, dtype=float)
    n_tasks_seen, n_tasks = acc.shape

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(10, 5))
    else:
        fig = ax.get_figure()

    cmap = plt.cm.plasma
    for t in range(n_tasks):
        color = cmap(t / max(n_tasks - 1, 1))
        task_acc = acc[:, t]
        valid = task_acc > 0
        if valid.any():
            max_acc = task_acc[valid].max()
            forgetting = max_acc - task_acc[valid]
            ax.plot(np.where(valid)[0], forgetting, linewidth=1.2, color=color, label=f"Task {t}")

    ax.set_facecolor("#0e0e12")
    ax.set_xlabel("Tasks Seen")
    ax.set_ylabel("Forgetting (1 - Acc)")
    ax.set_title(title)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=7, framealpha=0.3, labelcolor="white", ncol=2)

    fig.patch.set_facecolor("#0e0e12")
    ax.tick_params(colors="#888888")
    for spine in ax.spines.values():
        spine.set_color("#333333")

    if own_fig:
        fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    return fig


def plot_bwt(
    fwt_values,
    title: str = "Backward/Forward Transfer",
    save_path: str | None = None,
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Plot backward and forward transfer over tasks.

    Args:
        fwt_values: dict with keys 'bwt' and 'fwt', each a list of per-task values.
        title: Plot title.
        save_path: If given, save figure.
        ax: Optional matplotlib axes.

    Returns:
        matplotlib Figure.
    """
    bwt = np.asarray(fwt_values.get("bwt", []), dtype=float)
    fwt = np.asarray(fwt_values.get("fwt", []), dtype=float)

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(10, 4))
    else:
        fig = ax.get_figure()

    if len(bwt) > 0:
        ax.plot(range(len(bwt)), bwt, linewidth=1.2, color="#4a9eff", marker="o", markersize=3, label="BWT")
    if len(fwt) > 0:
        ax.plot(range(len(fwt)), fwt, linewidth=1.2, color="#ff9f43", marker="s", markersize=3, label="FWT")

    ax.axhline(0, color="#888888", linewidth=0.8, linestyle="--")
    ax.set_facecolor("#0e0e12")
    ax.set_xlabel("Task")
    ax.set_ylabel("Transfer Score")
    ax.set_title(title)
    ax.legend(fontsize=8, framealpha=0.3, labelcolor="white")

    fig.patch.set_facecolor("#0e0e12")
    ax.tick_params(colors="#888888")
    for spine in ax.spines.values():
        spine.set_color("#333333")

    if own_fig:
        fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    return fig
