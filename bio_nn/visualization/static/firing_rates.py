"""Firing rate visualization for BIO-NN."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


def plot_firing_rates(
    spike_history,
    title: str = "Firing Rate Distribution",
    save_path: str | None = None,
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Plot per-neuron firing rate histogram.

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

    rates = spikes.mean(axis=0)

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(8, 4))
    else:
        fig = ax.get_figure()

    ax.hist(rates, bins=60, color="#4a9eff", edgecolor="#0e0e12", linewidth=0.3, alpha=0.9)
    ax.axvline(rates.mean(), color="#ff4a4a", linewidth=1, linestyle="--", label=f"Mean={rates.mean():.4f}")

    ax.set_facecolor("#0e0e12")
    ax.set_xlabel("Firing Rate (spikes/step)")
    ax.set_ylabel("Neuron Count")
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


def plot_firing_rate_over_time(
    spike_history,
    title: str = "Firing Rate Over Time",
    save_path: str | None = None,
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Plot average firing rate per time step.

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

    rate_over_time = spikes.mean(axis=1)

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(12, 4))
    else:
        fig = ax.get_figure()

    ax.plot(rate_over_time, linewidth=0.8, color="#4a9eff")
    ax.fill_between(range(len(rate_over_time)), rate_over_time, alpha=0.2, color="#4a9eff")

    ax.set_facecolor("#0e0e12")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Average Firing Rate")
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
