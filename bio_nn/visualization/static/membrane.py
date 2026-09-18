"""Membrane potential visualization for BIO-NN."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


def plot_membrane_potential(
    membrane_history,
    neuron_ids: list[int] | None = None,
    title: str = "Membrane Potential",
    save_path: str | None = None,
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Plot membrane potential over time for selected neurons.

    Args:
        membrane_history: Array of shape (time_steps, n_neurons) or (time_steps, batch, n_neurons).
        neuron_ids: Which neurons to plot. None plots first 10.
        title: Plot title.
        save_path: If given, save figure to this path.
        ax: Optional matplotlib axes.

    Returns:
        matplotlib Figure.
    """
    membrane = np.asarray(membrane_history)
    if membrane.ndim == 3:
        membrane = membrane.mean(axis=1)

    n_total = membrane.shape[1]
    if neuron_ids is None:
        neuron_ids = list(range(min(10, n_total)))
    neuron_ids = [i for i in neuron_ids if i < n_total]

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(12, 5))
    else:
        fig = ax.get_figure()

    cmap = plt.cm.viridis
    for idx, nid in enumerate(neuron_ids):
        color = cmap(idx / max(len(neuron_ids) - 1, 1))
        ax.plot(membrane[:, nid], linewidth=0.8, label=f"Neuron {nid}", color=color)

    ax.set_facecolor("#0e0e12")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Membrane Potential")
    ax.set_title(title)
    if len(neuron_ids) <= 20:
        ax.legend(fontsize=7, loc="upper right", framealpha=0.3, labelcolor="white")

    fig.patch.set_facecolor("#0e0e12")
    ax.tick_params(colors="#888888")
    for spine in ax.spines.values():
        spine.set_color("#333333")

    if own_fig:
        fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    return fig
