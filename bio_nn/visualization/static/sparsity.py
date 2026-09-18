"""Sparsity visualization for BIO-NN."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


def plot_sparsity_over_time(
    spike_history,
    title: str = "Sparsity Over Time",
    save_path: str | None = None,
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Plot fraction of silent neurons per time step.

    Args:
        spike_history: (time_steps, n_neurons) or (time_steps, batch, n_neurons).
        title: Plot title.
        save_path: If given, save figure.
        ax: Optional matplotlib axes.

    Returns:
        matplotlib Figure.
    """
    spikes = np.asarray(spike_history)
    if spikes.ndim == 3:
        spikes = spikes.mean(axis=1)

    fraction_silent = (spikes < 0.5).mean(axis=1)

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(12, 4))
    else:
        fig = ax.get_figure()

    ax.plot(fraction_silent, linewidth=0.8, color="#4a9eff")
    ax.fill_between(range(len(fraction_silent)), fraction_silent, alpha=0.2, color="#4a9eff")

    ax.set_facecolor("#0e0e12")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Fraction Silent")
    ax.set_title(title)
    ax.set_ylim(0, 1.05)

    fig.patch.set_facecolor("#0e0e12")
    ax.tick_params(colors="#888888")
    for spine in ax.spines.values():
        spine.set_color("#333333")

    if own_fig:
        fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    return fig
