"""Network structure visualization for BIO-NN."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


def plot_connectivity_over_time(
    connectivity_history,
    title: str = "Connectivity Over Time",
    save_path: str | None = None,
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Plot number of active connections per epoch.

    Args:
        connectivity_history: List of weight matrices or integer counts of active connections.
        title: Plot title.
        save_path: If given, save figure.
        ax: Optional matplotlib axes.

    Returns:
        matplotlib Figure.
    """
    if len(connectivity_history) > 0 and np.isscalar(connectivity_history[0]):
        active_counts = np.array(connectivity_history, dtype=float)
    else:
        active_counts = np.array([
            (np.asarray(w) > 0).sum() if np.issubdtype(np.asarray(w).dtype, np.number) else float(w)
            for w in connectivity_history
        ], dtype=float)

    epochs = np.arange(len(active_counts))

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(10, 4))
    else:
        fig = ax.get_figure()

    ax.plot(epochs, active_counts, linewidth=1.2, color="#4a9eff")
    ax.fill_between(epochs, active_counts, alpha=0.15, color="#4a9eff")

    ax.set_facecolor("#0e0e12")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Active Connections")
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


def plot_network_topology(
    weights,
    threshold: float = 0.01,
    title: str = "Network Topology",
    save_path: str | None = None,
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Plot network graph (neurons as nodes, connections as edges).

    Uses networkx if available, otherwise falls back to a scatter-based layout.

    Args:
        weights: 2D weight matrix (n_neurons, n_neurons).
        threshold: Minimum absolute weight to draw an edge.
        title: Plot title.
        save_path: If given, save figure.
        ax: Optional matplotlib axes.

    Returns:
        matplotlib Figure.
    """
    w = np.asarray(weights)
    n = w.shape[0]

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(8, 8))
    else:
        fig = ax.get_figure()

    try:
        import networkx as nx
        G = nx.DiGraph()
        G.add_nodes_from(range(n))
        for i in range(n):
            for j in range(n):
                if i != j and abs(w[i, j]) > threshold:
                    G.add_edge(i, j, weight=float(w[i, j]))

        pos = nx.spring_layout(G, seed=42, k=1.5 / np.sqrt(n))

        edge_weights = [abs(d["weight"]) for _, _, d in G.edges(data=True)]
        if edge_weights:
            ew_max = max(edge_weights) or 1.0
        else:
            ew_max = 1.0
        edge_widths = [1.5 * (ew / ew_max) for ew in edge_weights]
        edge_colors = ["#4a9eff" if d["weight"] > 0 else "#ff4a4a" for _, _, d in G.edges(data=True)]

        node_degrees = [G.degree(i) for i in range(n)]
        max_deg = max(node_degrees) if node_degrees else 1
        node_sizes = [100 + 400 * (d / max(max_deg, 1)) for d in node_degrees]

        nx.draw_networkx_nodes(G, pos, ax=ax, node_size=node_sizes, node_color="#4a9eff", alpha=0.8)
        if G.edges():
            nx.draw_networkx_edges(G, pos, ax=ax, width=edge_widths, edge_color=edge_colors, alpha=0.5, arrows=True, arrowsize=6)
    except ImportError:
        angle = np.linspace(0, 2 * np.pi, n, endpoint=False)
        x = np.cos(angle)
        y = np.sin(angle)

        for i in range(n):
            for j in range(n):
                if i != j and abs(w[i, j]) > threshold:
                    alpha = min(abs(w[i, j]) * 2, 0.6)
                    color = "#4a9eff" if w[i, j] > 0 else "#ff4a4a"
                    ax.annotate("", xy=(x[j], y[j]), xytext=(x[i], y[i]),
                                arrowprops=dict(arrowstyle="-", color=color, alpha=alpha, lw=0.5))

        ax.scatter(x, y, s=80, c="#4a9eff", zorder=5)
        for i in range(n):
            ax.annotate(str(i), (x[i], y[i]), fontsize=6, ha="center", va="center", color="white")

    ax.set_facecolor("#0e0e12")
    ax.set_title(title)
    ax.set_aspect("equal")

    fig.patch.set_facecolor("#0e0e12")
    ax.tick_params(colors="#888888", labelbottom=False, labelleft=False)
    for spine in ax.spines.values():
        spine.set_color("#333333")

    if own_fig:
        fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    return fig
