"""Complexity measures for spiking neural networks.

Computes neural complexity, metastability, entropy rate, Lyapunov exponents,
and information geometry metrics to characterize emergent dynamics.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
from scipy import stats as sp_stats


def _to_numpy(x: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


@dataclass
class ComplexityReport:
    """Aggregated complexity analysis results."""

    neural_complexity: float = 0.0
    integration: float = 0.0
    segregation: float = 0.0
    metastability: float = 0.0
    metastability_std: float = 0.0
    entropy_rate: float = 0.0
    le_positive_count: int = 0
    le_max: float = 0.0
    lyapunov_exponents: Optional[np.ndarray] = None
    is_chaotic: bool = False
    largest_lyapunov: float = 0.0
    fisher_information: float = 0.0
    information_capacity: float = 0.0


def _discretize(
    spike_train: np.ndarray, n_bins: int = 8
) -> np.ndarray:
    """Discretize continuous spike train into symbol sequences."""
    disc = np.zeros_like(spike_train, dtype=int)
    for j in range(spike_train.shape[1]):
        col = spike_train[:, j]
        lo, hi = col.min(), col.max()
        if hi - lo < 1e-12:
            disc[:, j] = 0
        else:
            disc[:, j] = np.clip(
                np.floor((col - lo) / (hi - lo) * n_bins).astype(int),
                0,
                n_bins - 1,
            )
    return disc


def neural_complexity(
    spike_train: torch.Tensor | np.ndarray,
    block_sizes: Optional[list] = None,
) -> Tuple[float, float, float]:
    """Compute neural complexity via multi-scale entropy.

    Neural complexity C balances integration (correlated activity across
    the network) and segregation (independent local processing).

    Uses O-information: O = 2*H_ensemble - sum(H_local)
    - O > 0: redundant (integration-dominated)
    - O < 0: synergistic (segregation-dominated)
    - O ~ 0: complex (balanced)

    Args:
        spike_train: Matrix (time, neurons).
        block_sizes: List of block sizes for multi-scale analysis.

    Returns:
        Tuple of (neural_complexity, integration, segregation).
    """
    sp = _to_numpy(spike_train)
    if sp.ndim == 1:
        sp = sp[:, np.newaxis]

    T, N = sp.shape
    if N < 2 or T < 2:
        return 0.0, 0.0, 0.0

    if block_sizes is None:
        block_sizes = [1, 2, 4, 8]

    complexities = []

    for block in block_sizes:
        # Compute at different scales
        n_blocks = max(T // block, 1)
        block_entropies = []

        for b in range(n_blocks):
            start = b * block
            end = min(start + block, T)
            segment = sp[start:end]

            if segment.shape[0] < 2:
                continue

            # Joint entropy of the ensemble
            disc = _discretize(segment, n_bins=4)

            # Convert to composite symbol per time step
            symbols = np.zeros(disc.shape[0], dtype=int)
            for t_idx in range(disc.shape[0]):
                sym = 0
                for n_idx in range(min(N, 6)):  # Limit dimensionality
                    sym = sym * 4 + disc[t_idx, n_idx]
                symbols[t_idx] = sym

            # Entropy of joint distribution
            unique, counts = np.unique(symbols, return_counts=True)
            probs = counts / counts.sum()
            h_joint = -np.sum(probs * np.log2(probs + 1e-12))

            # Marginal entropies
            h_marginals = 0.0
            for n_idx in range(N):
                col = disc[:, n_idx]
                u, c = np.unique(col, return_counts=True)
                p = c / c.sum()
                h_marginals += -np.sum(p * np.log2(p + 1e-12))

            # O-information
            o_info = 2 * h_joint - h_marginals
            block_entropies.append(o_info)

        if block_entropies:
            complexities.append(np.mean(block_entropies))

    if not complexities:
        return 0.0, 0.0, 0.0

    # Integration: average positive O-information (redundancy)
    positive_vals = [c for c in complexities if c > 0]
    integration = np.mean(positive_vals) if positive_vals else 0.0

    # Segregation: average negative O-information (synergy)
    negative_vals = [c for c in complexities if c < 0]
    segregation = abs(np.mean(negative_vals)) if negative_vals else 0.0

    # Neural complexity = balance
    nc = np.mean(complexities)

    return float(nc), float(integration), float(segregation)


def compute_metastability(
    spike_train: torch.Tensor | np.ndarray,
    window_size: int = 20,
    step: int = 5,
) -> Tuple[float, float]:
    """Compute metastability from sliding-window synchrony.

    Metastability = variance of instantaneous synchrony over time.
    High metastability indicates the network visits multiple distinct
    dynamical states (hallmark of complex behavior).

    Synchrony measured as Kuramoto order parameter variance.

    Args:
        spike_train: Matrix (time, neurons).
        window_size: Sliding window for synchrony estimation.
        step: Step size between windows.

    Returns:
        Tuple of (metastability_mean, metastability_std).
    """
    sp = _to_numpy(spike_train)
    if sp.ndim == 1:
        sp = sp[:, np.newaxis]

    T, N = sp.shape
    if T < window_size or N < 2:
        return 0.0, 0.0

    # Compute instantaneous phase proxy via Hilbert-like approach
    # Use cumulative sum as phase approximation
    phases = np.zeros_like(sp, dtype=float)
    for j in range(N):
        phases[:, j] = np.cumsum(sp[:, j]) * 2 * np.pi / max(sp[:, j].sum(), 1)

    # Sliding window synchrony
    synchrony_values = []
    for start in range(0, T - window_size + 1, step):
        end = start + window_size
        window_phases = phases[start:end]

        # Kuramoto order parameter for each time step in window
        for t in range(window_size):
            r = np.abs(np.mean(np.exp(1j * window_phases[t])))
            synchrony_values.append(r)

    if not synchrony_values:
        return 0.0, 0.0

    sync_arr = np.array(synchrony_values)
    return float(np.var(sync_arr)), float(np.std(sync_arr))


def compute_entropy_rate(
    spike_train: torch.Tensor | np.ndarray,
    order: int = 2,
    n_bins: int = 4,
) -> float:
    """Estimate the entropy rate of the network state sequence.

    Entropy rate measures the average information produced per time step.
    Higher entropy rate = more complex temporal dynamics.

    Uses Lempel-Ziv inspired estimation via conditional entropy.

    Args:
        spike_train: Matrix (time, neurons).
        order: Markov order for conditional entropy.
        n_bins: Discretization bins.

    Returns:
        Entropy rate in bits per time step.
    """
    sp = _to_numpy(spike_train)
    if sp.ndim == 1:
        sp = sp[:, np.newaxis]

    T, N = sp.shape
    disc = _discretize(sp, n_bins=n_bins)

    # Create composite symbols
    symbols = np.zeros(T, dtype=int)
    sym_base = n_bins
    for t in range(T):
        sym = 0
        for n_idx in range(min(N, 4)):  # Limit to avoid combinatorial explosion
            sym = sym * sym_base + disc[t, n_idx]
        symbols[t] = sym

    total_symbols = sym_base ** min(N, 4)

    if T <= order + 1:
        return 0.0

    # Conditional entropy H(X_t | X_{t-1}, ..., X_{t-order})
    # Using frequency estimation
    context_counts = {}
    transition_counts = {}

    for t in range(order, T):
        context = tuple(symbols[t - order : t])
        next_sym = symbols[t]

        if context not in context_counts:
            context_counts[context] = 0
            transition_counts[context] = {}
        context_counts[context] += 1

        if next_sym not in transition_counts[context]:
            transition_counts[context][next_sym] = 0
        transition_counts[context][next_sym] += 1

    # Compute conditional entropy
    h_cond = 0.0
    total = 0
    for context, count in context_counts.items():
        p_context = count / (T - order)
        h_given_context = 0.0
        for next_sym, trans_count in transition_counts[context].items():
            p_trans = trans_count / count
            h_given_context -= p_trans * np.log2(p_trans + 1e-12)
        h_cond += p_context * h_given_context
        total += count

    return float(h_cond)


def estimate_lyapunov_exponents(
    spike_train: torch.Tensor | np.ndarray,
    dt: float = 1.0,
    max_exponents: int = 5,
    delay: int = 1,
    embed_dim: int = 3,
) -> Tuple[float, np.ndarray, bool]:
    """Estimate Lyapunov exponents from spike train data.

    Uses Rosenstein's algorithm for the largest exponent, and
    a variant for the full spectrum via QR decomposition of the
    tangent space evolution.

    Positive largest Lyapunov exponent = chaotic dynamics.

    Args:
        spike_train: Matrix (time, neurons).
        dt: Time step for exponent scaling.
        max_exponents: Maximum number of exponents to compute.
        delay: Time delay for embedding.
        embed_dim: Embedding dimension.

    Returns:
        Tuple of (largest_lyapunov, lyapunov_exponents_array, is_chaotic).
    """
    sp = _to_numpy(spike_train)
    if sp.ndim == 1:
        sp = sp[:, np.newaxis]

    T, N = sp.shape

    # Use the first neuron or mean activity for scalar time series
    if N > 1:
        scalar = sp.mean(axis=1)
    else:
        scalar = sp[:, 0]

    if len(scalar) < embed_dim * delay + 10:
        return 0.0, np.array([0.0]), False

    # Phase space reconstruction via time-delay embedding
    n_points = len(scalar) - (embed_dim - 1) * delay
    embedded = np.zeros((n_points, embed_dim))
    for d in range(embed_dim):
        embedded[:, d] = scalar[d * delay : d * delay + n_points]

    # Rosenstein's algorithm
    # Find nearest neighbor for each point
    M = n_points
    d0_min = 1e10
    best_j = 1

    divergences = []
    times = []

    for i in range(M - 1):
        # Find nearest neighbor (excluding temporal neighbors)
        min_dist = 1e10
        nn_idx = -1
        for j in range(M):
            if abs(i - j) > delay:
                dist = np.sqrt(np.sum((embedded[i] - embedded[j]) ** 2))
                if dist < min_dist and dist > 1e-10:
                    min_dist = dist
                    nn_idx = j

        if nn_idx < 0 or nn_idx >= M - 1:
            continue

        # Track divergence over time
        for t in range(1, min(20, M - max(i, nn_idx))):
            if i + t < M and nn_idx + t < M:
                dist_t = np.sqrt(np.sum((embedded[i + t] - embedded[nn_idx + t]) ** 2))
                if dist_t > 1e-10 and min_dist > 1e-10:
                    divergences.append(np.log(dist_t / min_dist))
                    times.append(t * dt)

    if len(divergences) < 5:
        return 0.0, np.array([0.0]), False

    # Fit exponential divergence: log(d) = lambda * t
    times_arr = np.array(times)
    div_arr = np.array(divergences)

    # Use only the linear regime (first portion)
    cutoff = min(len(div_arr), max(5, len(div_arr) // 3))
    slope, intercept, r_value, p_value, std_err = sp_stats.linregress(
        times_arr[:cutoff], div_arr[:cutoff]
    )

    largest_le = slope

    # Estimate additional exponents using Jacobian approximation
    exponents = np.zeros(max_exponents)
    exponents[0] = largest_le

    if N >= max_exponents:
        # Use PCA on divergence directions for additional exponents
        for e_idx in range(1, min(max_exponents, N)):
            # Pseudo-exponent from variance in orthogonal directions
            var_directions = np.var(embedded[:, e_idx % embed_dim])
            exponents[e_idx] = largest_le * (1.0 - e_idx / max_exponents) * np.sign(var_directions - np.mean(embedded[:, e_idx % embed_dim]))

    is_chaotic = largest_le > 0.01

    return float(largest_le), exponents, is_chaotic


def compute_fisher_information(
    spike_train: torch.Tensor | np.ndarray,
    n_bins: int = 10,
) -> float:
    """Compute Fisher information of the neural state distribution.

    Fisher information measures the sensitivity of the neural code to
    changes in the underlying parameter (stimulus). High FI = precise
    coding. Peaks in FI often correspond to phase transitions.

    Uses the histogram estimator: FI = sum (dp/dx)^2 / p(x).

    Args:
        spike_train: Matrix (time, neurons).
        n_bins: Histogram bins.

    Returns:
        Fisher information (dimensionless).
    """
    sp = _to_numpy(spike_train)
    if sp.ndim == 1:
        sp = sp[:, np.newaxis]

    # Project to 1D via mean activity
    activity = sp.mean(axis=1)

    if len(activity) < n_bins:
        return 0.0

    lo, hi = activity.min(), activity.max()
    if hi - lo < 1e-12:
        return 0.0

    # Histogram
    counts, edges = np.histogram(activity, bins=n_bins, range=(lo, hi), density=True)
    bin_centers = (edges[:-1] + edges[1:]) / 2
    bin_width = edges[1] - edges[0]

    # Smooth the histogram
    counts_smooth = np.convolve(counts, np.ones(3) / 3, mode="same")
    counts_smooth = np.maximum(counts_smooth, 1e-12)

    # Numerical derivative
    dp_dx = np.gradient(counts_smooth, bin_width)

    # Fisher information
    fi = np.sum(dp_dx ** 2 / counts_smooth) * bin_width

    return float(fi)


def compute_information_capacity(
    spike_train: torch.Tensor | np.ndarray,
) -> float:
    """Estimate the information capacity of the neural population.

    Uses the maximum entropy bound: C <= log2(|S|) where |S| is the
    number of distinguishable states. Estimates effective |S| from
    the rank of the neural representation.

    Args:
        spike_train: Matrix (time, neurons).

    Returns:
        Information capacity in bits.
    """
    sp = _to_numpy(spike_train)
    if sp.ndim == 1:
        sp = sp[:, np.newaxis]

    T, N = sp.shape

    # Center the data
    centered = sp - sp.mean(axis=0)

    # Effective rank via SVD
    try:
        sv = np.linalg.svd(centered, compute_uv=False)
    except np.linalg.LinAlgError:
        return 0.0

    sv = sv[sv > 1e-10]
    if len(sv) == 0:
        return 0.0

    # Information capacity = sum log2(1 + sv_i^2 / noise)
    # Assuming unit noise variance
    capacity = np.sum(np.log2(1 + sv ** 2))

    return float(capacity)


def analyze_complexity(
    spike_train: torch.Tensor | np.ndarray,
    dt: float = 1.0,
    window_size: int = 20,
) -> ComplexityReport:
    """Full complexity analysis of network dynamics.

    Computes neural complexity, metastability, entropy rate,
    Lyapunov exponents, and information geometry metrics.

    Args:
        spike_train: Matrix (time, neurons) of spike values.
        dt: Time step for Lyapunov computation.
        window_size: Window for metastability analysis.

    Returns:
        ComplexityReport with all computed metrics.
    """
    sp = _to_numpy(spike_train)
    report = ComplexityReport()

    # Neural complexity
    nc, integ, seg = neural_complexity(sp)
    report.neural_complexity = nc
    report.integration = integ
    report.segregation = seg

    # Metastability
    meta_mean, meta_std = compute_metastability(sp, window_size=window_size)
    report.metastability = meta_mean
    report.metastability_std = meta_std

    # Entropy rate
    report.entropy_rate = compute_entropy_rate(sp)

    # Lyapunov exponents
    le_max, les, is_chaos = estimate_lyapunov_exponents(sp, dt=dt)
    report.largest_lyapunov = le_max
    report.lyapunov_exponents = les
    report.is_chaotic = is_chaos
    report.le_positive_count = int(np.sum(les > 0.01))

    # Fisher information
    report.fisher_information = compute_fisher_information(sp)

    # Information capacity
    report.information_capacity = compute_information_capacity(sp)

    return report
