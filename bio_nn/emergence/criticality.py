"""Criticality detection for spiking neural networks.

Detects proximity to critical phase transitions via avalanche statistics,
branching ratios, susceptibility, and order parameter analysis.

Theory: A network at criticality maximizes dynamic range and information
transmission. Key signatures include power-law avalanche distributions
with exponent tau ~ 1.5, branching ratio sigma ~ 1, and diverging
susceptibility.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import torch
from scipy import stats as sp_stats


@dataclass
class CriticalityReport:
    """Aggregated criticality analysis results."""

    branching_ratio: float = 0.0
    branching_std: float = 0.0
    avalanche_sizes: Optional[np.ndarray] = None
    avalanche_durations: Optional[np.ndarray] = None
    power_law_exponent: float = 0.0
    power_law_p_value: float = 0.0
    power_law_xmin: float = 0.0
    is_power_law: bool = False
    susceptibility: float = 0.0
    order_parameter: float = 0.0
    correlation_length: float = 0.0
    distance_to_criticality: float = 1.0
    is_critical: bool = False


def _to_numpy(x: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _detect_avalanches(spike_train: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Detect neural avalanches from a binary spike train matrix.

    An avalanche is a contiguous period of activity separated by
    silent time bins.

    Args:
        spike_train: Matrix of shape (time, neurons) with binary spike values.

    Returns:
        Tuple of (avalanche_sizes, avalanche_durations) as integer arrays.
        Size = total neuron spikes in avalanche. Duration = number of
        time bins.
    """
    if spike_train.ndim == 1:
        spike_train = spike_train[:, np.newaxis]

    # Sum across neurons per time bin to get population activity
    activity = spike_train.sum(axis=1)
    is_active = activity > 0

    sizes: List[int] = []
    durations: List[int] = []

    in_avalanche = False
    current_size = 0
    current_duration = 0

    for t in range(len(is_active)):
        if is_active[t]:
            if not in_avalanche:
                in_avalanche = True
                current_size = 0
                current_duration = 0
            current_size += int(activity[t])
            current_duration += 1
        else:
            if in_avalanche:
                sizes.append(current_size)
                durations.append(current_duration)
                in_avalanche = False

    # Close any open avalanche
    if in_avalanche:
        sizes.append(current_size)
        durations.append(current_duration)

    return np.array(sizes, dtype=np.int64), np.array(durations, dtype=np.int64)


def _fit_power_law(
    data: np.ndarray,
    xmin: Optional[float] = None,
    xmax: Optional[float] = None,
) -> Tuple[float, float, float, bool]:
    """Fit a power-law distribution using MLE (Clauset et al. 2009).

    Args:
        data: Positive integer data to fit.
        xmin: Lower bound for power-law region. If None, estimated.
        xmax: Upper bound. If None, uses data max.

    Returns:
        Tuple of (alpha, p_value, xmin_fitted, is_power_law).
    """
    data = data[data > 0]
    if len(data) < 10:
        return 0.0, 0.0, 0.0, False

    if xmin is None:
        # Empirical heuristic: use median of positive values as starting point
        xmin_candidates = np.unique(data)
        xmin = float(np.median(data))

    fitted = data[data >= xmin]
    if len(fitted) < 5:
        return 0.0, 0.0, float(xmin), False

    # MLE for power-law exponent
    alpha = 1.0 + len(fitted) / np.sum(np.log(fitted / (xmin - 0.5)))

    # Kolmogorov-Smirnov goodness-of-fit via bootstrap
    n_boot = 100
    ks_stat, _ = sp_stats.kstest(
        fitted, lambda x: 1.0 - (xmin / x) ** (alpha - 1)
    )

    # Estimate p-value by comparing to synthetic power-law samples
    synthetic_ks = []
    for _ in range(n_boot):
        synthetic = xmin * (np.random.random(len(fitted)) ** (-1.0 / (alpha - 1)))
        s_ks, _ = sp_stats.kstest(
            synthetic, lambda x: 1.0 - (xmin / x) ** (alpha - 1)
        )
        synthetic_ks.append(s_ks)

    p_value = np.mean([s >= ks_stat for s in synthetic_ks])
    is_pl = p_value > 0.05 and alpha > 1.0 and alpha < 4.0

    return float(alpha), float(p_value), float(xmin), is_pl


def compute_branching_ratio(
    spike_train: torch.Tensor | np.ndarray,
) -> Tuple[float, float]:
    """Estimate the branching ratio (sigma) from spike train data.

    Sigma = <n_{t+1}> / <n_t> where n_t is the number of active neurons
    at time t. Sigma=1 indicates criticality, <1 is subcritical,
    >1 is supercritical.

    Args:
        spike_train: Matrix (time, neurons) of binary spikes.

    Returns:
        Tuple of (mean_branching_ratio, standard_deviation).
    """
    sp = _to_numpy(spike_train)
    if sp.ndim == 1:
        sp = sp[:, np.newaxis]

    activity = sp.sum(axis=1)
    if len(activity) < 2:
        return 0.0, 0.0

    # Compute ratio of successive activity levels, avoiding division by zero
    active_mask = activity[:-1] > 0
    if active_mask.sum() == 0:
        return 0.0, 0.0

    ratios = activity[1:][active_mask] / activity[:-1][active_mask]
    return float(np.mean(ratios)), float(np.std(ratios))


def compute_susceptibility(
    spike_train: torch.Tensor | np.ndarray,
    window_size: int = 10,
) -> float:
    """Compute susceptibility chi as variance of avalanche sizes.

    In critical phenomena, susceptibility diverges at the critical point.
    For finite systems, a peak in chi signals proximity to criticality.

    chi = (<s^2> - <s>^2) / <s>

    where s are avalanche sizes.

    Args:
        spike_train: Matrix (time, neurons) of binary spikes.
        window_size: Sliding window for local susceptibility.

    Returns:
        Susceptibility value (dimensionless).
    """
    sp = _to_numpy(spike_train)
    sizes, _ = _detect_avalanches(sp)

    if len(sizes) < 2:
        return 0.0

    mean_s = np.mean(sizes)
    var_s = np.var(sizes)

    if mean_s == 0:
        return 0.0

    return float(var_s / mean_s)


def compute_order_parameter(
    spike_train: torch.Tensor | np.ndarray,
) -> float:
    """Compute the order parameter phi = <n> / N.

    This measures the fraction of neurons active on average.
    In the ordered phase phi > 0, in the disordered phase phi -> 0.

    Args:
        spike_train: Matrix (time, neurons) of binary spikes.

    Returns:
        Order parameter in [0, 1].
    """
    sp = _to_numpy(spike_train)
    n_neurons = sp.shape[1] if sp.ndim > 1 else 1
    if n_neurons == 0:
        return 0.0

    activity = sp.mean(axis=0) if sp.ndim > 1 else sp.mean()
    return float(np.mean(activity))


def estimate_correlation_length(
    spike_train: torch.Tensor | np.ndarray,
    max_lag: int = 50,
) -> float:
    """Estimate spatial correlation length from spike train cross-correlations.

    The correlation length xi characterizes the spatial scale over which
    neurons are correlated. Divergence of xi signals criticality.

    Args:
        spike_train: Matrix (time, neurons).
        max_lag: Maximum spatial lag to evaluate.

    Returns:
        Estimated correlation length (in neuron-index units).
    """
    sp = _to_numpy(spike_train)
    if sp.ndim == 1:
        sp = sp[:, np.newaxis]

    n_neurons = sp.shape[1]
    if n_neurons < 2:
        return 0.0

    # Compute pairwise correlations
    corr_matrix = np.corrcoef(sp.T)
    corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)

    # Average correlation as function of neuron index distance
    max_lag = min(max_lag, n_neurons - 1)
    lag_corrs = np.zeros(max_lag + 1)

    for lag in range(max_lag + 1):
        pairs = []
        for i in range(n_neurons - lag):
            pairs.append(corr_matrix[i, i + lag])
        lag_corrs[lag] = np.mean(pairs) if pairs else 0.0

    # Fit exponential decay: C(d) ~ exp(-d/xi)
    d = np.arange(max_lag + 1)
    pos_mask = lag_corrs > 0
    if pos_mask.sum() < 3:
        return 1.0

    log_corr = np.log(lag_corrs[pos_mask] + 1e-10)
    d_pos = d[pos_mask]

    # Linear fit in log space
    slope, intercept, r_value, p_value, std_err = sp_stats.linregress(d_pos, log_corr)

    if slope >= 0:
        return float(max_lag)

    xi = -1.0 / slope
    return float(xi)


def analyze_criticality(
    spike_train: torch.Tensor | np.ndarray,
    window_size: int = 100,
    xmin: Optional[float] = None,
) -> CriticalityReport:
    """Full criticality analysis on a spike train recording.

    Computes branching ratio, avalanche size distribution, power-law fit,
    susceptibility, order parameter, and correlation length. Determines
    proximity to criticality.

    Args:
        spike_train: Matrix (time, neurons) of binary spike values.
        window_size: Window for sliding analyses.
        xmin: Optional power-law lower cutoff.

    Returns:
        CriticalityReport with all computed metrics.
    """
    sp = _to_numpy(spike_train)
    report = CriticalityReport()

    # Branching ratio
    sigma, sigma_std = compute_branching_ratio(sp)
    report.branching_ratio = sigma
    report.branching_std = sigma_std

    # Avalanches
    sizes, durations = _detect_avalanches(sp)
    report.avalanche_sizes = sizes
    report.avalanche_durations = durations

    # Power-law fit on avalanche sizes
    if len(sizes) > 10:
        alpha, p_val, xmin_fit, is_pl = _fit_power_law(sizes, xmin=xmin)
        report.power_law_exponent = alpha
        report.power_law_p_value = p_val
        report.power_law_xmin = xmin_fit
        report.is_power_law = is_pl

    # Susceptibility
    report.susceptibility = compute_susceptibility(sp, window_size)

    # Order parameter
    report.order_parameter = compute_order_parameter(sp)

    # Correlation length
    report.correlation_length = estimate_correlation_length(sp)

    # Distance to criticality (heuristic multi-criterion)
    sigma_dist = abs(sigma - 1.0)
    xi_dist = 1.0 / (1.0 + report.correlation_length)
    chi_signal = min(report.susceptibility / 10.0, 1.0)

    report.distance_to_criticality = float(
        np.sqrt(sigma_dist ** 2 + xi_dist ** 2 + (1.0 - chi_signal) ** 2) / math.sqrt(3)
    )

    # Critical if branching ratio near 1, power-law distributed, high susceptibility
    report.is_critical = (
        sigma_dist < 0.15
        and report.is_power_law
        and report.susceptibility > 1.0
    )

    return report


class CriticalityDetector:
    """Streaming criticality detector for online monitoring.

    Accumulates spike data in a buffer and periodically runs
    criticality analysis.

    Args:
        n_neurons: Expected number of neurons.
        buffer_size: Number of time steps to buffer before analysis.
        analysis_interval: Run full analysis every N time steps.
    """

    def __init__(
        self,
        n_neurons: int,
        buffer_size: int = 500,
        analysis_interval: int = 100,
    ) -> None:
        self.n_neurons = n_neurons
        self.buffer_size = buffer_size
        self.analysis_interval = analysis_interval

        self.buffer = np.zeros((buffer_size, n_neurons), dtype=np.float32)
        self.buffer_idx = 0
        self.step_count = 0
        self.history: List[CriticalityReport] = []

    def update(self, spikes: torch.Tensor | np.ndarray) -> Optional[CriticalityReport]:
        """Add a time step of spike data.

        Args:
            spikes: Vector of shape (n_neurons,) with binary spike values.

        Returns:
            CriticalityReport if analysis was triggered, else None.
        """
        sp = _to_numpy(spikes).flatten()
        if len(sp) != self.n_neurons:
            raise ValueError(
                f"Expected {self.n_neurons} neurons, got {len(sp)}"
            )

        self.buffer[self.buffer_idx % self.buffer_size] = sp
        self.buffer_idx += 1
        self.step_count += 1

        if self.step_count % self.analysis_interval == 0 and self.step_count >= self.buffer_size:
            valid_len = min(self.buffer_idx, self.buffer_size)
            report = analyze_criticality(self.buffer[:valid_len])
            self.history.append(report)
            return report

        return None

    def get_history(self) -> List[CriticalityReport]:
        """Return all historical criticality reports."""
        return list(self.history)

    def is_near_critical(self, threshold: float = 0.3) -> bool:
        """Check if the most recent analysis indicates near-critical dynamics."""
        if not self.history:
            return False
        return self.history[-1].distance_to_criticality < threshold
