"""Memory routing analysis for spiking neural networks.

Tracks information flow, detects adaptive routing vs fixed paths,
analyzes memory consolidation, identifies bottlenecks, and tracks
memory capacity growth.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


def _to_numpy(x: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


@dataclass
class RoutingReport:
    """Results of memory routing analysis."""

    flow_matrix: Optional[np.ndarray] = None
    adaptive_index: float = 0.0
    path_entropy: float = 0.0
    consolidation_ratio: float = 0.0
    bottleneck_scores: Optional[np.ndarray] = None
    bottleneck_indices: Optional[np.ndarray] = None
    capacity_history: Optional[np.ndarray] = None
    capacity_growth_rate: float = 0.0
    rerouting_events: int = 0
    path_stability: float = 0.0
    congestion_index: float = 0.0


def compute_information_flow(
    spike_train: torch.Tensor | np.ndarray,
    time_window: int = 10,
    lag: int = 1,
) -> np.ndarray:
    """Compute directed information flow matrix between neurons.

    Uses time-lagged cross-correlation to estimate directed causal
    influence. Flow[i,j] = influence of neuron i on neuron j.

    Args:
        spike_train: Matrix (time, neurons).
        time_window: Window for computing local flow.
        lag: Time lag for directed influence.

    Returns:
        Flow matrix (neurons, neurons), non-negative.
    """
    sp = _to_numpy(spike_train)
    if sp.ndim == 1:
        sp = sp[:, np.newaxis]

    T, N = sp.shape
    flow = np.zeros((N, N))

    if T <= lag:
        return flow

    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            # Time-lagged correlation: cov(x_t, y_{t+lag})
            x = sp[:T - lag, i]
            y = sp[lag:, j]

            # Compute directed transfer function magnitude
            # Use partial coherence approximation via lagged correlation
            cross_corr = np.mean(x * y)
            auto_i = np.mean(x * x)
            auto_j = np.mean(y * y)

            denom = np.sqrt(auto_i * auto_j)
            if denom > 1e-12:
                flow[i, j] = max(cross_corr / denom, 0.0)

    return flow


def compute_adaptive_index(
    flow_matrix_sequence: List[np.ndarray],
) -> float:
    """Measure how much routing patterns change over time (adaptivity).

    Adaptive routing = paths change based on input/context.
    Fixed routing = same paths regardless of input.

    The adaptive index is the normalized variance of flow patterns:
    - 0 = perfectly fixed routing
    - 1 = maximally adaptive routing

    Args:
        flow_matrix_sequence: List of flow matrices at different time windows.

    Returns:
        Adaptive index in [0, 1].
    """
    if len(flow_matrix_sequence) < 2:
        return 0.0

    matrices = np.stack(flow_matrix_sequence, axis=0)
    total_var = np.var(matrices)
    mean_val = np.mean(matrices)

    if mean_val < 1e-12:
        return 0.0

    # Coefficient of variation across time, normalized
    cv = np.sqrt(total_var) / (mean_val + 1e-12)
    # Map to [0, 1] using sigmoid-like saturation
    return float(cv / (1.0 + cv))


def compute_path_entropy(
    flow_matrix: torch.Tensor | np.ndarray,
    top_k: int = 3,
) -> float:
    """Compute entropy over dominant routing paths.

    High path entropy = many possible routes (robust, distributed).
    Low path entropy = few dominant paths (specialized, fragile).

    Args:
        flow_matrix: Directed flow matrix (neurons, neurons).
        top_k: Number of top paths to consider per source.

    Returns:
        Path entropy in bits.
    """
    flow = _to_numpy(flow_matrix)
    N = flow.shape[0]
    all_probs = []

    for i in range(N):
        row = flow[i].copy()
        row[i] = 0.0  # No self-connections
        if row.sum() < 1e-12:
            continue

        # Take top-k destinations
        top_indices = np.argsort(row)[-top_k:]
        probs = row[top_indices]
        probs = probs / probs.sum()
        all_probs.extend(probs)

    if not all_probs:
        return 0.0

    all_probs = np.array(all_probs)
    all_probs = all_probs / all_probs.sum()

    # Shannon entropy
    entropy = -np.sum(all_probs * np.log2(all_probs + 1e-12))
    max_entropy = np.log2(N) if N > 1 else 1.0

    return float(entropy / max_entropy) if max_entropy > 0 else 0.0


def compute_consolidation_ratio(
    short_term_flow: np.ndarray,
    long_term_flow: np.ndarray,
) -> float:
    """Compute memory consolidation ratio.

    Compares short-term (recent) flow patterns to long-term (accumulated)
    patterns. High ratio = active consolidation, memories being transferred
    to stable storage.

    Args:
        short_term_flow: Recent flow matrix.
        long_term_flow: Accumulated flow matrix.

    Returns:
        Consolidation ratio in [0, 1]. 1 = perfect consolidation.
    """
    st = _to_numpy(short_term_flow)
    lt = _to_numpy(long_term_flow)

    if st.sum() < 1e-12 or lt.sum() < 1e-12:
        return 0.0

    # Normalize
    st_norm = st / (st.sum() + 1e-12)
    lt_norm = lt / (lt.sum() + 1e-12)

    # Cosine similarity of flow distributions
    dot = np.sum(st_norm * lt_norm)
    norm_st = np.sqrt(np.sum(st_norm ** 2))
    norm_lt = np.sqrt(np.sum(lt_norm ** 2))

    if norm_st < 1e-12 or norm_lt < 1e-12:
        return 0.0

    return float(dot / (norm_st * norm_lt))


def compute_bottleneck_scores(
    flow_matrix: torch.Tensor | np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Identify memory bottleneck points via betweenness centrality.

    Bottleneck neurons are those through which most information flows.
    They are critical for memory routing but vulnerable to disruption.

    Args:
        flow_matrix: Directed flow matrix.

    Returns:
        Tuple of (bottleneck_scores, bottleneck_indices_sorted_desc).
    """
    flow = _to_numpy(flow_matrix)
    N = flow.shape[0]

    if N < 2:
        return np.array([1.0]), np.array([0])

    # Approximate betweenness centrality via flow-mediated paths
    centrality = np.zeros(N)

    # Convert to probability transition matrix
    trans = flow.copy()
    row_sums = trans.sum(axis=1, keepdims=True)
    row_sums[row_sums < 1e-12] = 1.0
    trans = trans / row_sums

    # Power iteration to approximate centrality
    # Stationary distribution of the flow network
    pi = np.ones(N) / N
    for _ in range(50):
        pi_new = pi @ trans
        if np.abs(pi_new - pi).sum() < 1e-8:
            break
        pi = pi_new

    # Betweenness approximation: nodes with high in-flow AND out-flow
    in_flow = flow.sum(axis=0)   # How much flows INTO this node
    out_flow = flow.sum(axis=1)  # How much flows OUT of this node

    # A bottleneck has both high in and out flow
    centrality = np.sqrt(in_flow * out_flow)
    centrality_sum = centrality.sum()

    if centrality_sum > 1e-12:
        centrality = centrality / centrality_sum

    sorted_indices = np.argsort(centrality)[::-1]

    return centrality, sorted_indices


def compute_capacity_growth(
    spike_history: List[np.ndarray],
    window_size: int = 50,
    measure: str = "rank",
) -> np.ndarray:
    """Track memory capacity growth over time.

    Estimates the effective dimensionality (rank) of neural representations
    as a proxy for memory capacity.

    Args:
        spike_history: List of spike matrices (time, neurons) at different epochs.
        window_size: Number of time steps for rank estimation.
        measure: 'rank' for effective rank, 'variance' for total variance.

    Returns:
        Array of capacity values, one per entry in spike_history.
    """
    capacities = []

    for sp in spike_history:
        sp_np = _to_numpy(sp)
        if sp_np.ndim == 1:
            sp_np = sp_np[:, np.newaxis]

        T, N = sp_np.shape
        sp_window = sp_np[:min(window_size, T)]

        if measure == "rank":
            # Effective rank via singular values
            centered = sp_window - sp_window.mean(axis=0)
            sv = np.linalg.svd(centered, compute_uv=False)
            sv = sv[sv > 1e-10]

            if len(sv) == 0:
                capacities.append(0.0)
                continue

            # Effective rank = exp(entropy of normalized singular values)
            sv_norm = sv / sv.sum()
            entropy = -np.sum(sv_norm * np.log(sv_norm + 1e-12))
            capacities.append(float(np.exp(entropy)))
        else:
            # Total variance as capacity proxy
            capacities.append(float(np.var(sp_window)))

    return np.array(capacities)


def detect_rerouting_events(
    flow_sequence: List[np.ndarray],
    threshold: float = 0.3,
) -> int:
    """Count significant rerouting events in the flow history.

    A rerouting event is detected when the flow pattern changes
    more than the threshold relative to the previous stable state.

    Args:
        flow_sequence: Ordered list of flow matrices.
        threshold: Relative change threshold for rerouting detection.

    Returns:
        Number of detected rerouting events.
    """
    if len(flow_sequence) < 2:
        return 0

    events = 0
    prev_flow = flow_sequence[0]

    for i in range(1, len(flow_sequence)):
        curr = flow_sequence[i]
        diff = np.abs(curr - prev_flow).sum()
        norm = np.abs(prev_flow).sum() + 1e-12

        if diff / norm > threshold:
            events += 1
            prev_flow = curr
        else:
            # Gradual update
            prev_flow = 0.9 * prev_flow + 0.1 * curr

    return events


def compute_path_stability(
    flow_sequence: List[np.ndarray],
    top_k: int = 3,
) -> float:
    """Measure stability of dominant routing paths over time.

    High stability = same dominant paths throughout.
    Low stability = paths change frequently.

    Args:
        flow_sequence: Ordered list of flow matrices.
        top_k: Number of top paths to track per source.

    Returns:
        Stability in [0, 1].
    """
    if len(flow_sequence) < 2:
        return 1.0

    # Extract top-k paths for each source at each time step
    path_sets = []
    for flow in flow_sequence:
        paths = set()
        N = flow.shape[0]
        for i in range(N):
            row = flow[i].copy()
            row[i] = 0.0
            top_indices = np.argsort(row)[-top_k:]
            for j in top_indices:
                if row[j] > 0:
                    paths.add((i, j))
        path_sets.append(paths)

    # Compute Jaccard similarity between consecutive time steps
    similarities = []
    for t in range(1, len(path_sets)):
        if path_sets[t] or path_sets[t - 1]:
            intersection = len(path_sets[t] & path_sets[t - 1])
            union = len(path_sets[t] | path_sets[t - 1])
            similarities.append(intersection / max(union, 1))
        else:
            similarities.append(1.0)

    return float(np.mean(similarities)) if similarities else 1.0


def analyze_memory_routing(
    spike_train: torch.Tensor | np.ndarray,
    spike_history: Optional[List[np.ndarray]] = None,
    time_window: int = 50,
    lag: int = 1,
) -> RoutingReport:
    """Full memory routing analysis.

    Tracks information flow, adaptivity, consolidation, bottlenecks,
    and capacity growth.

    Args:
        spike_train: Current spike train matrix (time, neurons).
        spike_history: Historical spike trains for capacity tracking.
        time_window: Window size for flow computation.
        lag: Time lag for directed flow.

    Returns:
        RoutingReport with all computed metrics.
    """
    sp = _to_numpy(spike_train)
    report = RoutingReport()

    # Flow matrix
    report.flow_matrix = compute_information_flow(sp, time_window=time_window, lag=lag)

    # Adaptive index (requires multiple windows)
    T = sp.shape[0]
    n_windows = max(T // time_window, 1)
    flow_sequence = []
    for w in range(n_windows):
        start = w * time_window
        end = min(start + time_window, T)
        if end - start > lag:
            flow_w = compute_information_flow(sp[start:end], lag=lag)
            flow_sequence.append(flow_w)

    if len(flow_sequence) >= 2:
        report.adaptive_index = compute_adaptive_index(flow_sequence)
        report.rerouting_events = detect_rerouting_events(flow_sequence)
        report.path_stability = compute_path_stability(flow_sequence)

    # Path entropy
    report.path_entropy = compute_path_entropy(report.flow_matrix)

    # Congestion index: fraction of neurons receiving disproportionate flow
    if report.flow_matrix is not None:
        in_flow = report.flow_matrix.sum(axis=0)
        if in_flow.sum() > 0:
            in_flow_norm = in_flow / in_flow.sum()
            # Gini coefficient as congestion measure
            sorted_flow = np.sort(in_flow_norm)
            n = len(sorted_flow)
            index = np.arange(1, n + 1)
            report.congestion_index = float(
                (2 * np.sum(index * sorted_flow) / (n * np.sum(sorted_flow) + 1e-12)) - (n + 1) / n
            )

    # Bottleneck analysis
    scores, indices = compute_bottleneck_scores(report.flow_matrix)
    report.bottleneck_scores = scores
    report.bottleneck_indices = indices

    # Consolidation ratio (compare first and second half)
    half = T // 2
    if half > lag:
        st_flow = compute_information_flow(sp[:half], lag=lag)
        lt_flow = compute_information_flow(sp[half:], lag=lag)
        report.consolidation_ratio = compute_consolidation_ratio(st_flow, lt_flow)

    # Capacity growth
    if spike_history and len(spike_history) > 1:
        report.capacity_history = compute_capacity_growth(spike_history)
        if len(report.capacity_history) >= 2:
            x = np.arange(len(report.capacity_history))
            slope, _, _, _, _ = __import__("scipy").stats.linregress(x, report.capacity_history)
            report.capacity_growth_rate = float(slope)

    return report
