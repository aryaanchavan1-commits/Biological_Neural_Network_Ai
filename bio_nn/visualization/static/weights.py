"""Weight visualization for BIO-NN."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm


def plot_weight_heatmap(
    weights,
    title: str = "Synaptic Weights",
    save_path: str | None = None,
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Plot weight matrix as heatmap.

    Args:
        weights: 2D array (n_neurons, n_neurons) or a list/stacked array (takes last).
        title: Plot title.
        save_path: If given, save figure.
        ax: Optional matplotlib axes.

    Returns:
        matplotlib Figure.
    """
    w = np.asarray(weights)
    if w.ndim == 3:
        w = w[-1]

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(8, 8))
    else:
        fig = ax.get_figure()

    vmax = np.percentile(np.abs(w), 99) or 1.0
    norm = SymLogNorm(linthresh=0.01, vmin=-vmax, vmax=vmax)
    im = ax.imshow(w, cmap="RdBu_r", norm=norm, aspect="auto")

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


def plot_weight_distribution(
    weights,
    title: str = "Weight Distribution",
    save_path: str | None = None,
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Plot histogram of weight values.

    Args:
        weights: Weight array (any shape, flattened for histogram).
        title: Plot title.
        save_path: If given, save figure.
        ax: Optional matplotlib axes.

    Returns:
        matplotlib Figure.
    """
    w = np.asarray(weights).flatten()

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(8, 4))
    else:
        fig = ax.get_figure()

    ax.hist(w, bins=100, color="#4a9eff", edgecolor="#0e0e12", linewidth=0.3, alpha=0.9)
    ax.axvline(0, color="#ff4a4a", linewidth=0.8, linestyle="--", alpha=0.6)

    ax.set_facecolor("#0e0e12")
    ax.set_xlabel("Weight Value")
    ax.set_ylabel("Count")
    ax.set_title(title)

    fig.patch.set_facecolor("#0e0e12")
    ax.tick_params(colors="#888888")
    for spine in ax.spines.values():
        spine.set_color("#333333")

    if own_fig:
        fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    return fig
