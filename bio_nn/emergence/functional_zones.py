"""Functional specialization detection for spiking neural networks.

Detects the emergence of specialized functional regions via clustering,
information-theoretic measures, and functional connectivity analysis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from scipy import stats as sp_stats
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform


def _to_numpy(x: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _safe_log(p: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return np.log(np.clip(p, eps, 1.0))


@dataclass
class FunctionalZoneReport:
    """Results of functional specialization analysis."""

    n_clusters: int = 0
    cluster_labels: Optional[np.ndarray] = None
    cluster_sizes: Optional[np.ndarray] = None
    specialization_index: float = 0.0
    mutual_information_matrix: Optional[np.ndarray] = None
    average_mi: float = 0.0
    transfer_entropy_matrix: Optional[np.ndarray] = None
    average_te: float = 0.0
    functional_connectivity: Optional[np.ndarray] = None
    modularity: float = 0.0
    participation_ratio: float = 0.0
    zone_descriptions: Optional[Dict[int, Dict]] = None


def compute_pairwise_correlation(
    spike_train: torch.Tensor | np.ndarray,
    method: str = "pearson",
) -> np.ndarray:
    """Compute pairwise correlation matrix between neurons.

    Args:
        spike_train: Matrix (time, neurons).
        method: 'pearson', 'spearman', or 'covariance'.

    Returns:
        Correlation matrix of shape (neurons, neurons).
    """
    sp = _to_numpy(spike_train)
    if sp.ndim == 1:
        sp = sp[:, np.newaxis]

    n = sp.shape[1]
    if n < 2:
        return np.eye(1)

    if method == "pearson":
        corr = np.corrcoef(sp.T)
    elif method == "spearman":
        corr, _ = sp_stats.spearmanr(sp)
        if np.isscalar(corr):
            corr = np.array([[1.0, corr], [corr, 1.0]])
    elif method == "covariance":
        cov = np.cov(sp.T)
        d = np.sqrt(np.diag(cov))
        d[d == 0] = 1.0
        corr = cov / np.outer(d, d)
    else:
        corr = np.corrcoef(sp.T)

    return np.nan_to_num(corr, nan=0.0)


def compute_mutual_information(
    spike_train: torch.Tensor | np.ndarray,
    n_bins: int = 10,
) -> np.ndarray:
    """Compute pairwise mutual information matrix.

    Uses histogram-based estimation of MI between neuron rate codes.

    Args:
        spike_train: Matrix (time, neurons).
        n_bins: Number of bins for histogram estimation.

    Returns:
        MI matrix of shape (neurons, neurons) in bits.
    """
    sp = _to_numpy(spike_train)
    if sp.ndim == 1:
        sp = sp[:, np.newaxis]

    n_neurons = sp.shape[1]
    mi_matrix = np.zeros((n_neurons, n_neurons))

    # Discretize each neuron's activity
    disc = np.zeros_like(sp, dtype=int)
    for j in range(n_neurons):
        col = sp[:, j]
        if col.max() == col.min():
            disc[:, j] = 0
        else:
            disc[:, j] = np.digitize(col, bins=np.linspace(col.min(), col.max(), n_bins)) - 1
            disc[:, j] = np.clip(disc[:, j], 0, n_bins - 1)

    for i in range(n_neurons):
        for j in range(i + 1, n_neurons):
            # Joint histogram
            joint = np.zeros((n_bins, n_bins))
            for t in range(len(sp)):
                joint[int(disc[t, i]), int(disc[t, j])] += 1
            joint /= joint.sum() + 1e-12

            p_i = joint.sum(axis=1)
            p_j = joint.sum(axis=0)

            # MI = sum P(i,j) * log(P(i,j) / (P(i)*P(j)))
            mi = 0.0
            for ii in range(n_bins):
                for jj in range(n_bins):
                    if joint[ii, jj] > 1e-12:
                        mi += joint[ii, jj] * _safe_log(
                            np.array([joint[ii, jj] / (p_i[ii] * p_j[jj] + 1e-12)])
                        ).item()

            mi_matrix[i, j] = mi
            mi_matrix[j, i] = mi

    return mi_matrix


def compute_transfer_entropy(
    spike_train: torch.Tensor | np.ndarray,
    n_bins: int = 5,
    k: int = 1,
) -> np.ndarray:
    """Compute pairwise transfer entropy matrix.

    TE_{X->Y} measures the directed information flow from X to Y,
    capturing causal influence beyond correlation.

    Uses histogram estimation with history length k.

    Args:
        spike_train: Matrix (time, neurons).
        n_bins: Bins for discretization.
        k: History length (number of past time steps).

    Returns:
        TE matrix of shape (neurons, neurons). Entry [i,j] = TE from i to j.
    """
    sp = _to_numpy(spike_train)
    if sp.ndim == 1:
        sp = sp[:, np.newaxis]

    n_neurons = sp.shape[1]
    T = sp.shape[0]
    te_matrix = np.zeros((n_neurons, n_neurons))

    if T <= k + 1:
        return te_matrix

    # Discretize
    disc = np.zeros_like(sp, dtype=int)
    for j in range(n_neurons):
        col = sp[:, j]
        if col.max() == col.min():
            disc[:, j] = 0
        else:
            disc[:, j] = np.digitize(col, bins=np.linspace(col.min(), col.max(), n_bins)) - 1
            disc[:, j] = np.clip(disc[:, j], 0, n_bins - 1)

    for src in range(n_neurons):
        for tgt in range(n_neurons):
            if src == tgt:
                continue

            # Build state vectors: (target_history, source_current, target_next)
            valid = T - k
            target_past = disc[k:T, tgt]  # Current target state
            target_future = disc[k + 1 : T + 1, tgt] if T > k + 1 else disc[k:T, tgt]
            source_past = np.zeros((valid, k), dtype=int)
            for lag in range(k):
                source_past[:, lag] = disc[k - lag - 1 : T - lag - 1, src]

            if len(target_future) != valid:
                continue

            # TE = H(Y'|Y) - H(Y'|Y, X)
            # where Y'=target_future, Y=target_history, X=source_history
            n_states = n_bins ** (k + 1)

            # Entropy without source
            joint_no_src = np.zeros((n_bins, n_states))
            state_idx_no_src = np.zeros(valid, dtype=int)
            for t in range(valid):
                state_idx_no_src[t] = target_past[t]
                for lag in range(k):
                    state_idx_no_src[t] = state_idx_no_src[t] * n_bins + source_past[t, lag] // n_bins

            for t in range(valid):
                joint_no_src[target_future[t], state_idx_no_src[t]] += 1

            h_no_src = 0.0
            for s in range(n_states):
                p_y_given_s = joint_no_src[:, s] / (joint_no_src[:, s].sum() + 1e-12)
                p_s = joint_no_src[:, s].sum() / valid
                if p_s > 1e-12:
                    entropy = -np.sum(p_y_given_s * _safe_log(p_y_given_s))
                    h_no_src -= p_s * _safe_log(np.array([p_s])).item() * 0
                    h_no_src += p_s * entropy

            # Entropy with source
            joint_with_src = np.zeros((n_bins, n_bins, n_bins ** k))
            state_idx_src = np.zeros(valid, dtype=int)
            for t in range(valid):
                state_idx_src[t] = 0
                for lag in range(k):
                    state_idx_src[t] = state_idx_src[t] * n_bins + source_past[t, lag]

            for t in range(valid):
                joint_with_src[target_future[t], target_past[t], state_idx_src[t]] += 1

            h_with_src = 0.0
            for sp_idx in range(n_bins ** k):
                for y_past in range(n_bins):
                    p = joint_with_src[:, y_past, sp_idx]
                    p_total = p.sum()
                    if p_total > 1e-12:
                        p_norm = p / p_total
                        p_joint = p_total / valid
                        entropy = -np.sum(p_norm * _safe_log(p_norm))
                        h_with_src -= p_joint * _safe_log(np.array([p_joint])).item() * 0
                        h_with_src += p_joint * entropy

            te = h_no_src - h_with_src
            te_matrix[src, tgt] = max(te, 0.0)

    return te_matrix


def cluster_neurons(
    spike_train: torch.Tensor | np.ndarray,
    n_clusters: Optional[int] = None,
    max_clusters: int = 10,
    method: str = "correlation",
) -> Tuple[np.ndarray, int]:
    """Cluster neurons into functional groups based on activity patterns.

    Uses hierarchical clustering on correlation distance.

    Args:
        spike_train: Matrix (time, neurons).
        n_clusters: Target number of clusters. If None, auto-determine.
        max_clusters: Maximum clusters to consider.
        method: 'correlation' or 'mi' (mutual information).

    Returns:
        Tuple of (cluster_labels, n_clusters_chosen).
    """
    sp = _to_numpy(spike_train)
    if sp.ndim == 1:
        sp = sp[:, np.newaxis]

    n_neurons = sp.shape[1]
    if n_neurons < 2:
        return np.array([0]), 1

    if method == "mi":
        dist_matrix = compute_mutual_information(sp)
        # Convert MI to distance (higher MI = closer)
        dist_matrix = 1.0 - dist_matrix / (dist_matrix.max() + 1e-12)
        np.fill_diagonal(dist_matrix, 0.0)
        dist_matrix = np.maximum(dist_matrix, 0.0)
        # Ensure symmetry
        dist_matrix = (dist_matrix + dist_matrix.T) / 2.0
    else:
        corr = compute_pairwise_correlation(sp)
        dist_matrix = 1.0 - np.abs(corr)
        np.fill_diagonal(dist_matrix, 0.0)

    # Condensed distance for linkage
    try:
        condensed = squareform(dist_matrix, checks=False)
    except ValueError:
        condensed = dist_matrix[np.triu_indices(n_neurons, k=1)]

    if n_clusters is None:
        # Auto-determine via inconsistency
        Z = linkage(condensed, method="average")
        max_k = min(max_clusters, n_neurons)
        best_k = 2
        best_score = -1.0

        for k in range(2, max_k + 1):
            labels = fcluster(Z, t=k, criterion="maxclust")
            # Silhouette-like score
            unique_labels = np.unique(labels)
            if len(unique_labels) < 2:
                continue

            silhouette_scores = []
            for idx in range(n_neurons):
                same_cluster = labels == labels[idx]
                other_clusters = labels != labels[idx]

                if same_cluster.sum() <= 1 or other_clusters.sum() == 0:
                    silhouette_scores.append(0.0)
                    continue

                a = np.mean(dist_matrix[idx, same_cluster])
                b_vals = []
                for c in unique_labels:
                    if c != labels[idx]:
                        mask = labels == c
                        b_vals.append(np.mean(dist_matrix[idx, mask]))
                b = min(b_vals) if b_vals else 0.0

                s = (b - a) / max(a, b, 1e-12)
                silhouette_scores.append(s)

            score = np.mean(silhouette_scores)
            if score > best_score:
                best_score = score
                best_k = k

        n_clusters = best_k

    Z = linkage(condensed, method="average")
    labels = fcluster(Z, t=n_clusters, criterion="maxclust")

    return labels - 1, n_clusters  # 0-indexed


def compute_modularity(
    correlation_matrix: np.ndarray,
    labels: np.ndarray,
) -> float:
    """Compute Newman-Girvan modularity Q for a given partition.

    Q = (1/2m) * sum_{ij} [A_{ij} - k_i*k_j/(2m)] * delta(c_i, c_j)

    Args:
        correlation_matrix: Weighted adjacency matrix (e.g., abs correlation).
        labels: Cluster assignment for each neuron.

    Returns:
        Modularity Q in [-0.5, 1]. Q > 0.3 indicates significant structure.
    """
    A = np.abs(correlation_matrix)
    np.fill_diagonal(A, 0.0)

    k = A.sum(axis=1)
    m = k.sum() / 2.0

    if m < 1e-12:
        return 0.0

    Q = 0.0
    unique_labels = np.unique(labels)
    for c in unique_labels:
        mask = labels == c
        subgraph = A[np.ix_(mask, mask)]
        ls = subgraph.sum()
        ks = k[mask].sum()
        Q += ls - ks ** 2 / (2.0 * m)

    return float(Q / (2.0 * m))


def compute_participation_ratio(
    spike_train: torch.Tensor | np.ndarray,
) -> float:
    """Compute the participation ratio from PCA eigenvalues.

    The participation ratio measures how many neurons effectively
    participate in the network's activity. Low PR = specialized,
    high PR = distributed.

    PR = (sum lambda_i)^2 / sum lambda_i^2

    Args:
        spike_train: Matrix (time, neurons).

    Returns:
        Participation ratio (dimensionless, in [1, n_neurons]).
    """
    sp = _to_numpy(spike_train)
    if sp.ndim == 1:
        return 1.0

    # Center and compute covariance
    sp_centered = sp - sp.mean(axis=0)
    cov = np.cov(sp_centered.T)

    eigenvalues = np.linalg.eigvalsh(cov)
    eigenvalues = eigenvalues[eigenvalues > 0]

    if len(eigenvalues) == 0:
        return 1.0

    sum_lam = eigenvalues.sum()
    sum_lam_sq = (eigenvalues ** 2).sum()

    if sum_lam_sq < 1e-12:
        return 1.0

    return float(sum_lam ** 2 / sum_lam_sq)


def analyze_functional_zones(
    spike_train: torch.Tensor | np.ndarray,
    n_clusters: Optional[int] = None,
    n_bins_mi: int = 10,
    n_bins_te: int = 5,
    te_k: int = 1,
) -> FunctionalZoneReport:
    """Full functional specialization analysis.

    Detects specialized regions, computes information-theoretic measures,
    functional connectivity, and modularity.

    Args:
        spike_train: Matrix (time, neurons) of spike values.
        n_clusters: Target cluster count (auto if None).
        n_bins_mi: Bins for mutual information.
        n_bins_te: Bins for transfer entropy.
        te_k: History length for transfer entropy.

    Returns:
        FunctionalZoneReport with all computed metrics.
    """
    sp = _to_numpy(spike_train)
    report = FunctionalZoneReport()

    n_neurons = sp.shape[1] if sp.ndim > 1 else 1

    # Clustering
    labels, n_clusters_chosen = cluster_neurons(sp, n_clusters=n_clusters)
    report.n_clusters = n_clusters_chosen
    report.cluster_labels = labels

    unique, counts = np.unique(labels, return_counts=True)
    report.cluster_sizes = counts

    # Specialization index: ratio of within-cluster to between-cluster variance
    within_var = 0.0
    between_var = 0.0
    global_mean = sp.mean()

    for c in unique:
        mask = labels == c
        cluster_data = sp[:, mask]
        within_var += np.var(cluster_data)
        cluster_mean = cluster_data.mean()
        between_var += mask.sum() * (cluster_mean - global_mean) ** 2

    total_var = np.var(sp) * n_neurons
    report.specialization_index = 1.0 - (within_var / (total_var + 1e-12))

    # Mutual information
    report.mutual_information_matrix = compute_mutual_information(sp, n_bins=n_bins_mi)
    report.average_mi = float(
        report.mutual_information_matrix[
            np.triu_indices(n_neurons, k=1)
        ].mean()
    ) if n_neurons > 1 else 0.0

    # Transfer entropy
    if sp.shape[0] > te_k + 2:
        report.transfer_entropy_matrix = compute_transfer_entropy(
            sp, n_bins=n_bins_te, k=te_k
        )
        report.average_te = float(
            report.transfer_entropy_matrix[
                np.triu_indices(n_neurons, k=1)
            ].mean()
        ) if n_neurons > 1 else 0.0

    # Functional connectivity
    report.functional_connectivity = compute_pairwise_correlation(sp)

    # Modularity
    report.modularity = compute_modularity(report.functional_connectivity, labels)

    # Participation ratio
    report.participation_ratio = compute_participation_ratio(sp)

    # Zone descriptions
    report.zone_descriptions = {}
    for c in unique:
        mask = labels == c
        zone_data = sp[:, mask]
        report.zone_descriptions[int(c)] = {
            "size": int(mask.sum()),
            "mean_firing_rate": float(zone_data.mean()),
            "firing_rate_std": float(zone_data.std()),
            "intra_cluster_corr": float(
                np.mean(compute_pairwise_correlation(zone_data)[np.triu_indices(mask.sum(), k=1)])
            ) if mask.sum() > 1 else 0.0,
        }

    return report
