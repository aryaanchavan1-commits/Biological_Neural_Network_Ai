"""Plasticity visualization for BIO-NN."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm


def plot_plasticity_over_time(
    weight_history,
    title: str = "Weight Changes Over Time",
    save_path: str | None = None,
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Plot mean/std of weights per epoch.

    Args:
        weight_history: List or array of weight matrices, shape (n_epochs, n_in, n_out) or list thereof.
        title: Plot title.
        save_path: If given, save figure.
        ax: Optional matplotlib axes.

    Returns:
        matplotlib Figure.
    """
    if not isinstance(weight_history, np.ndarray):
        weight_history = np.array([np.asarray(w) for w in weight_history])

    means = np.array([w.mean() for w in weight_history])
    stds = np.array([w.std() for w in weight_history])
    epochs = np.arange(len(means))

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(10, 4))
    else:
        fig = ax.get_figure()

    ax.plot(epochs, means, linewidth=1.2, color="#4a9eff", label="Mean")
    ax.fill_between(epochs, means - stds, means + stds, alpha=0.2, color="#4a9eff", label="Std")
    ax.plot(epochs, stds, linewidth=0.8, color="#ff9f43", linestyle="--", label="Std Dev")

    ax.set_facecolor("#0e0e12")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Weight Value")
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


def plot_plasticity_heatmap(
    weight_changes,
    title: str = "Plasticity Heatmap",
    save_path: str | None = None,
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Plot which connections changed most.

    Args:
        weight_changes: 2D array (n_in, n_out) of absolute weight changes.
        title: Plot title.
        save_path: If given, save figure.
        ax: Optional matplotlib axes.

    Returns:
        matplotlib Figure.
    """
    wc = np.asarray(weight_changes)

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(8, 8))
    else:
        fig = ax.get_figure()

    vmax = np.percentile(np.abs(wc), 99) or 1.0
    im = ax.imshow(wc, cmap="hot", aspect="auto", vmin=0, vmax=vmax)

    ax.set_facecolor("#0e0e12")
    ax.set_xlabel("Target Neuron")
    ax.set_ylabel("Source Neuron")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.patch.set_facecolor("#0e0e12")
    ax.tick_params(colors="#888888")
    for spine in ax.spines.values():
        spine.set_color("#333333")

    if own_fig:
        fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    return fig
