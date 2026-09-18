"""Runtime safety monitoring for spiking neural networks.

Implements continuous monitoring of network activations, performance,
and resource usage during deployment.  Detects anomalies, concept drift,
performance regressions, resource exhaustion, and safety violations in
real time.

In biological superintelligence research, runtime monitoring is the last
line of defense.  Even after thorough interpretability analysis,
controllability testing, and alignment verification, a deployed system
may encounter conditions not covered by those offline evaluations.
Continuous monitoring catches failures that slip through offline checks.

The monitoring pipeline works as:

1. **Anomaly detection**: Flag activation patterns that deviate from
   the training distribution (using Mahalanobis distance or
   reconstruction-based methods).
2. **Drift detection**: Detect gradual shifts in input statistics or
   internal representations that may indicate distribution change.
3. **Performance regression**: Track task metrics and detect statistically
   significant drops.
4. **Resource exhaustion**: Monitor GPU memory, spike rates, and
   computational budget.
5. **Safety violation logging**: Record and classify all safety-relevant
   events for post-hoc analysis.

Typical usage::

    monitor = SafetyMonitor(config={
        "anomaly_threshold": 3.0,
        "drift_window": 100,
        "performance_patience": 10,
    })

    # During deployment, feed each batch:
    alerts = monitor.step(activations, predictions, metadata)

    # After deployment:
    report = monitor.get_report()
"""

from __future__ import annotations

import time
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


class SafetyMonitor:
    """Runtime safety monitoring system for deployed SNNs.

    Continuously tracks activation distributions, performance metrics,
    resource usage, and detects safety-relevant anomalies.

    Args:
        config: Configuration dictionary. Supported keys:

            - ``anomaly_threshold`` (float): z-score threshold for
              anomaly detection. Default 3.0.
            - ``drift_window`` (int): Number of steps to keep in the
              sliding window for drift detection. Default 100.
            - ``drift_threshold`` (float): PSI threshold for concept
              drift. Default 0.2.
            - ``performance_patience`` (int): Consecutive steps of
              performance drop before alerting. Default 5.
            - ``performance_drop_pct`` (float): Minimum performance drop
              (%) to trigger regression alert. Default 5.0.
            - ``max_spike_rate`` (float): Maximum acceptable mean firing
              rate. Default 0.5.
            - ``max_memory_fraction`` (float): Max GPU memory usage
              fraction. Default 0.9.
            - ``log_history`` (bool): Whether to keep full history.
              Default True.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}
        self.anomaly_threshold: float = self.config.get("anomaly_threshold", 3.0)
        self.drift_window: int = self.config.get("drift_window", 100)
        self.drift_threshold: float = self.config.get("drift_threshold", 0.2)
        self.perf_patience: int = self.config.get("performance_patience", 5)
        self.perf_drop_pct: float = self.config.get("performance_drop_pct", 5.0)
        self.max_spike_rate: float = self.config.get("max_spike_rate", 0.5)
        self.max_memory_frac: float = self.config.get("max_memory_fraction", 0.9)
        self.log_history: bool = self.config.get("log_history", True)

        # Running statistics for anomaly detection
        self._activation_mean: Optional[torch.Tensor] = None
        self._activation_cov_inv: Optional[torch.Tensor] = None
        self._n_accumulated: int = 0

        # Sliding windows
        self._performance_window: deque = deque(maxlen=self.drift_window)
        self._spike_rate_window: deque = deque(maxlen=self.drift_window)
        self._anomaly_scores: deque = deque(maxlen=self.drift_window)

        # Performance tracking
        self._baseline_performance: Optional[float] = None
        self._consecutive_drops: int = 0
        self._best_performance: float = float("inf")

        # History and alerts
        self._alerts: List[Dict[str, Any]] = []
        self._step_count: int = 0
        self._full_history: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Internal: anomaly detection (Mahalanobis distance)
    # ------------------------------------------------------------------

    def _update_running_stats(self, activations: torch.Tensor) -> None:
        """Update running mean and inverse covariance for Mahalanobis distance."""
        flat = activations.detach().float().reshape(activations.shape[0], -1)
        batch_mean = flat.mean(dim=0)

        if self._activation_mean is None:
            self._activation_mean = batch_mean.clone()
            self._activation_cov_inv = torch.eye(flat.shape[1])
            self._n_accumulated = flat.shape[0]
        else:
            # Welford-like online update for mean
            n = self._n_accumulated + flat.shape[0]
            delta = batch_mean - self._activation_mean
            self._activation_mean += delta * (flat.shape[0] / n)
            self._n_accumulated = n

            # Update covariance periodically (every 50 steps)
            if self._n_accumulated > 0 and self._step_count % 50 == 0:
                centered = flat - self._activation_mean
                cov = (centered.T @ centered) / max(centered.shape[0] - 1, 1)
                cov += torch.eye(cov.shape[0]) * 1e-6  # regularize
                try:
                    self._activation_cov_inv = torch.linalg.inv(cov)
                except torch.linalg.LinAlgError:
                    self._activation_cov_inv = torch.linalg.pinv(cov)

    def _compute_anomaly_score(self, activations: torch.Tensor) -> torch.Tensor:
        """Compute Mahalanobis distance for each sample."""
        if self._activation_mean is None or self._activation_cov_inv is None:
            return torch.zeros(activations.shape[0])

        flat = activations.detach().float().reshape(activations.shape[0], -1)
        diff = flat - self._activation_mean.unsqueeze(0)
        left = diff @ self._activation_cov_inv
        scores = (left * diff).sum(dim=-1).sqrt()
        return scores

    # ------------------------------------------------------------------
    # Internal: drift detection
    # ------------------------------------------------------------------

    def _detect_drift(self) -> Optional[Dict[str, Any]]:
        """Detect concept drift in the performance window."""
        if len(self._performance_window) < self.drift_window:
            return None

        window = list(self._performance_window)
        first_half = np.array(window[: self.drift_window // 2])
        second_half = np.array(window[self.drift_window // 2 :])

        # KS-test-like statistic: max CDF difference
        sorted_first = np.sort(first_half)
        sorted_second = np.sort(second_half)
        all_vals = np.sort(np.concatenate([sorted_first, sorted_second]))

        cdf1 = np.searchsorted(sorted_first, all_vals, side="right") / len(sorted_first)
        cdf2 = np.searchsorted(sorted_second, all_vals, side="right") / len(sorted_second)
        ks_stat = np.max(np.abs(cdf1 - cdf2))

        # Mean shift
        mean_shift = abs(second_half.mean() - first_half.mean())
        relative_shift = mean_shift / (first_half.std() + 1e-8)

        if relative_shift > 2.0:
            return {
                "type": "concept_drift",
                "severity": "high" if relative_shift > 4.0 else "medium",
                "ks_statistic": float(ks_stat),
                "relative_shift": float(relative_shift),
                "first_half_mean": float(first_half.mean()),
                "second_half_mean": float(second_half.mean()),
            }
        return None

    # ------------------------------------------------------------------
    # Internal: resource monitoring
    # ------------------------------------------------------------------

    def _check_resources(self) -> List[Dict[str, Any]]:
        """Check GPU memory and compute resource usage."""
        alerts = []

        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / (1024 ** 3)
            reserved = torch.cuda.memory_reserved() / (1024 ** 3)
            max_mem = torch.cuda.get_device_properties(0).total_mem / (1024 ** 3)
            frac = allocated / max_mem if max_mem > 0 else 0.0

            if frac > self.max_memory_frac:
                alerts.append({
                    "type": "resource_exhaustion",
                    "resource": "gpu_memory",
                    "severity": "high" if frac > 0.95 else "medium",
                    "current_fraction": float(frac),
                    "threshold": self.max_memory_frac,
                    "allocated_gb": float(allocated),
                    "total_gb": float(max_mem),
                })

        # Spike rate check
        if self._spike_rate_window:
            recent_rates = list(self._spike_rate_window)[-10:]
            mean_rate = np.mean(recent_rates)
            if mean_rate > self.max_spike_rate:
                alerts.append({
                    "type": "resource_exhaustion",
                    "resource": "spike_rate",
                    "severity": "high" if mean_rate > self.max_spike_rate * 2 else "medium",
                    "current_rate": float(mean_rate),
                    "threshold": self.max_spike_rate,
                })

        return alerts

    # ------------------------------------------------------------------
    # Main interface: step
    # ------------------------------------------------------------------

    def step(
        self,
        activations: Optional[torch.Tensor] = None,
        predictions: Optional[torch.Tensor] = None,
        targets: Optional[torch.Tensor] = None,
        spike_rate: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Process one monitoring step and return any triggered alerts.

        Call this for every batch (or every N batches) during deployment.
        The monitor maintains running statistics and sliding windows
        internally.

        Args:
            activations: Network activation tensor ``(batch, n_features)``
                or ``(time, batch, n_neurons)`` for the current step.
            predictions: Model predictions ``(batch,)`` or ``(batch, n_classes)``.
            targets: Ground-truth labels ``(batch,)`` for performance tracking.
            spike_rate: Current mean firing rate (scalar in [0, 1]).
            metadata: Arbitrary metadata to log with this step.

        Returns:
            List of alert dictionaries.  Empty list means all clear.
        """
        self._step_count += 1
        step_alerts: List[Dict[str, Any]] = []

        # 1. Anomaly detection
        if activations is not None:
            self._update_running_stats(activations)
            scores = self._compute_anomaly_score(activations)
            self._anomaly_scores.extend(scores.tolist())

            if scores.max().item() > self.anomaly_threshold:
                step_alerts.append({
                    "type": "anomaly",
                    "severity": "high" if scores.max().item() > self.anomaly_threshold * 2 else "medium",
                    "max_score": scores.max().item(),
                    "mean_score": scores.mean().item(),
                    "threshold": self.anomaly_threshold,
                    "step": self._step_count,
                })

        # 2. Performance tracking
        if predictions is not None and targets is not None:
            with torch.no_grad():
                if predictions.ndim > 1:
                    preds = predictions.argmax(dim=-1)
                else:
                    preds = (predictions > 0.5).long()
                acc = (preds == targets).float().mean().item()

            self._performance_window.append(acc)

            # Check for regression
            if self._baseline_performance is None:
                self._baseline_performance = acc
            else:
                drop_pct = ((self._baseline_performance - acc) / (self._baseline_performance + 1e-8)) * 100
                if drop_pct > self.perf_drop_pct:
                    self._consecutive_drops += 1
                    if self._consecutive_drops >= self.perf_patience:
                        step_alerts.append({
                            "type": "performance_regression",
                            "severity": "high",
                            "current_accuracy": float(acc),
                            "baseline_accuracy": float(self._baseline_performance),
                            "drop_percent": float(drop_pct),
                            "consecutive_steps": self._consecutive_drops,
                            "step": self._step_count,
                        })
                        self._consecutive_drops = 0
                else:
                    self._consecutive_drops = 0
                    # Update baseline if performance improved
                    if acc < self._best_performance:
                        self._best_performance = acc
                        self._baseline_performance = acc

        # 3. Spike rate tracking
        if spike_rate is not None:
            self._spike_rate_window.append(spike_rate)

        # 4. Drift detection
        drift_alert = self._detect_drift()
        if drift_alert is not None:
            drift_alert["step"] = self._step_count
            step_alerts.append(drift_alert)

        # 5. Resource monitoring
        resource_alerts = self._check_resources()
        for alert in resource_alerts:
            alert["step"] = self._step_count
        step_alerts.extend(resource_alerts)

        # Log
        if self.log_history:
            entry = {
                "step": self._step_count,
                "timestamp": time.time(),
                "alerts": step_alerts,
            }
            if metadata:
                entry["metadata"] = metadata
            if activations is not None:
                entry["activation_shape"] = list(activations.shape)
            if spike_rate is not None:
                entry["spike_rate"] = spike_rate
            self._full_history.append(entry)

        self._alerts.extend(step_alerts)
        return step_alerts

    # ------------------------------------------------------------------
    # Violation logging
    # ------------------------------------------------------------------

    def log_safety_violation(
        self,
        violation_type: str,
        details: Dict[str, Any],
        severity: str = "high",
    ) -> None:
        """Explicitly log a safety violation.

        Use this when an external check detects a violation that the
        automatic monitors might miss (e.g., an伦理 review flag,
        a human override, or a constraint violation).

        Args:
            violation_type: Category of violation (e.g.,
                ``"unauthorized_action"``, ``"constraint_violation"``).
            details: Arbitrary dictionary of violation details.
            severity: One of ``"low"``, ``"medium"``, ``"high"``, ``"critical"``.
        """
        alert = {
            "type": "safety_violation",
            "violation_type": violation_type,
            "severity": severity,
            "details": details,
            "step": self._step_count,
            "timestamp": time.time(),
        }
        self._alerts.append(alert)
        if self.log_history:
            self._full_history.append(alert)

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------

    def get_report(self) -> Dict[str, Any]:
        """Generate a summary report of all monitoring data.

        Returns:
            Dictionary with keys:

            - ``total_steps``: Number of monitoring steps processed.
            - ``total_alerts``: Total number of alerts triggered.
            - ``alerts_by_type``: ``{type: count}`` breakdown.
            - ``alerts_by_severity``: ``{severity: count}`` breakdown.
            - ``anomaly_score_stats``: ``{mean, std, max, p95}`` of
              recorded anomaly scores.
            - ``performance_stats``: ``{mean, std, min, max}`` of
              recorded performance values.
            - ``drift_detected``: Whether any drift was detected.
            - ``recommendations``: List of safety recommendations based
              on the monitoring data.
        """
        alerts_by_type: Dict[str, int] = {}
        alerts_by_severity: Dict[str, int] = {}
        for alert in self._alerts:
            a_type = alert.get("type", "unknown")
            a_sev = alert.get("severity", "unknown")
            alerts_by_type[a_type] = alerts_by_type.get(a_type, 0) + 1
            alerts_by_severity[a_sev] = alerts_by_severity.get(a_sev, 0) + 1

        # Anomaly score stats
        anomaly_stats = {}
        if self._anomaly_scores:
            scores = np.array(list(self._anomaly_scores))
            anomaly_stats = {
                "mean": float(scores.mean()),
                "std": float(scores.std()),
                "max": float(scores.max()),
                "p95": float(np.percentile(scores, 95)),
            }

        # Performance stats
        perf_stats = {}
        if self._performance_window:
            perf = np.array(list(self._performance_window))
            perf_stats = {
                "mean": float(perf.mean()),
                "std": float(perf.std()),
                "min": float(perf.min()),
                "max": float(perf.max()),
            }

        # Recommendations
        recommendations = []
        if alerts_by_severity.get("high", 0) > 0 or alerts_by_severity.get("critical", 0) > 0:
            recommendations.append(
                "HIGH/CRITICAL alerts detected. Consider pausing deployment "
                "and investigating the root cause before resuming."
            )
        if alerts_by_type.get("concept_drift", 0) > 0:
            recommendations.append(
                "Concept drift detected. The deployment distribution may have "
                "shifted from training. Retrain or fine-tune the model."
            )
        if alerts_by_type.get("performance_regression", 0) > 0:
            recommendations.append(
                "Performance regression detected. Check for data quality issues, "
                "model degradation, or adversarial conditions."
            )
        if anomaly_stats.get("p95", 0) > self.anomaly_threshold:
            recommendations.append(
                "95th percentile anomaly score exceeds threshold. The network "
                "is encountering many out-of-distribution inputs."
            )
        if not recommendations:
            recommendations.append("No safety concerns detected. Continue monitoring.")

        return {
            "total_steps": self._step_count,
            "total_alerts": len(self._alerts),
            "alerts_by_type": alerts_by_type,
            "alerts_by_severity": alerts_by_severity,
            "anomaly_score_stats": anomaly_stats,
            "performance_stats": perf_stats,
            "drift_detected": alerts_by_type.get("concept_drift", 0) > 0,
            "recommendations": recommendations,
        }

    def reset(self) -> None:
        """Reset all monitoring state while preserving the report."""
        self._activation_mean = None
        self._activation_cov_inv = None
        self._n_accumulated = 0
        self._performance_window.clear()
        self._spike_rate_window.clear()
        self._anomaly_scores.clear()
        self._baseline_performance = None
        self._consecutive_drops = 0
        self._best_performance = float("inf")
        self._alerts.clear()
        self._full_history.clear()
        self._step_count = 0
