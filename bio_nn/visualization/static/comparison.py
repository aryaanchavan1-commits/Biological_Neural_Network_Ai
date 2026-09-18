"""Comparison visualization for BIO-NN."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


def plot_baseline_comparison(
    results_dict: dict[str, float],
    metric: str = "accuracy",
    title: str = "Baseline Comparison",
    save_path: str | None = None,
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Bar chart comparing different models.

    Args:
        results_dict: Mapping of model_name -> metric_value.
        metric: Metric name (used as y-axis label).
        title: Plot title.
        save_path: If given, save figure.
        ax: Optional matplotlib axes.

    Returns:
        matplotlib Figure.
    """
    names = list(results_dict.keys())
    values = list(results_dict.values())

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(10, 5))
    else:
        fig = ax.get_figure()

    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(names)))
    bars = ax.bar(names, values, color=colors, edgecolor="#0e0e12", linewidth=0.5)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", va="bottom", fontsize=8, color="white")

    ax.set_facecolor("#0e0e12")
    ax.set_ylabel(metric.capitalize())
    ax.set_title(title)
    ax.set_ylim(0, max(values) * 1.15 if values else 1)

    fig.patch.set_facecolor("#0e0e12")
    ax.tick_params(colors="#888888")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=8)
    for spine in ax.spines.values():
        spine.set_color("#333333")

    if own_fig:
        fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    return fig


def plot_ablation_comparison(
    results_dict: dict[str, float],
    metric: str = "accuracy",
    title: str = "Ablation Study",
    save_path: str | None = None,
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Bar chart for ablation study results.

    Args:
        results_dict: Mapping of variant_name -> metric_value.
        metric: Metric name.
        title: Plot title.
        save_path: If given, save figure.
        ax: Optional matplotlib axes.

    Returns:
        matplotlib Figure.
    """
    return plot_baseline_comparison(results_dict, metric=metric, title=title, save_path=save_path, ax=ax)


def plot_training_curves(
    histories_dict: dict[str, dict[str, list[float]]],
    title: str = "Training Curves",
    save_path: str | None = None,
) -> plt.Figure:
    """Plot loss/accuracy curves for multiple experiments.

    Args:
        histories_dict: Mapping of experiment_name -> dict with keys like
            'train_loss', 'val_loss', 'train_acc', 'val_acc'.
        title: Plot title.
        save_path: If given, save figure.

    Returns:
        matplotlib Figure.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("#0e0e12")

    for ax in axes:
        ax.set_facecolor("#0e0e12")
        ax.tick_params(colors="#888888")
        for spine in ax.spines.values():
            spine.set_color("#333333")

    cmap = plt.cm.tab10
    for idx, (name, hist) in enumerate(histories_dict.items()):
        color = cmap(idx % 10)

        if "train_loss" in hist:
            axes[0].plot(hist["train_loss"], linewidth=1, color=color, linestyle="-", label=f"{name} train")
        if "val_loss" in hist:
            axes[0].plot(hist["val_loss"], linewidth=1, color=color, linestyle="--", label=f"{name} val")
        if "train_acc" in hist:
            axes[1].plot(hist["train_acc"], linewidth=1, color=color, linestyle="-", label=f"{name} train")
        if "val_acc" in hist:
            axes[1].plot(hist["val_acc"], linewidth=1, color=color, linestyle="--", label=f"{name} val")

    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss")
    axes[0].legend(fontsize=7, framealpha=0.3, labelcolor="white")

    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Accuracy")
    axes[1].legend(fontsize=7, framealpha=0.3, labelcolor="white")

    fig.suptitle(title, color="white")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    return fig
