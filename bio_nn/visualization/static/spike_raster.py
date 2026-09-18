"""Spike raster visualization for BIO-NN."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def plot_spike_raster(
    spike_history,
    neuron_range: tuple[int, int] | None = None,
    time_range: tuple[int, int] | None = None,
    title: str = "Spike Raster",
    save_path: str | None = None,
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Plot spike raster (neuron x time).

    Args:
        spike_history: Array of shape (time_steps, n_neurons) or (time_steps, batch, n_neurons).
        neuron_range: (start, end) neuron indices to display.
        time_range: (start, end) time step indices to display.
        title: Plot title.
        save_path: If given, save figure to this path.
        ax: Optional matplotlib axes to plot on.

    Returns:
        matplotlib Figure.
    """
    spikes = np.asarray(spike_history)
    if spikes.ndim == 3:
        spikes = spikes.mean(axis=1)

    t_start, t_end = time_range or (0, spikes.shape[0])
    n_start, n_end = neuron_range or (0, spikes.shape[1])
    spikes = spikes[t_start:t_end, n_start:n_end]

    times, neurons = np.where(spikes > 0.5)

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(12, 5))
    else:
        fig = ax.get_figure()

    ax.scatter(times, neurons, s=0.5, c="white", marker="|", linewidths=0.3)
    ax.set_facecolor("#0e0e12")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Neuron Index")
    ax.set_title(title)
    ax.set_xlim(0, spikes.shape[0])
    ax.set_ylim(0, spikes.shape[1])

    fig.patch.set_facecolor("#0e0e12")
    ax.tick_params(colors="#888888")
    for spine in ax.spines.values():
        spine.set_color("#333333")

    if own_fig:
        fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    return fig
