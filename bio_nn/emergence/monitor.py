"""Real-time emergence monitoring during SNN training.

Continuously tracks criticality, functional specialization, memory routing,
and complexity metrics. Provides alerts for phase transitions, logging
integration, and visualization data generation.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch

from .criticality import CriticalityDetector, CriticalityReport, analyze_criticality
from .complexity import ComplexityReport, analyze_complexity
from .functional_zones import FunctionalZoneReport, analyze_functional_zones
from .memory_routing import RoutingReport, analyze_memory_routing

logger = logging.getLogger(__name__)


@dataclass
class AlertConfig:
    """Configuration for emergence alerts."""

    criticality_threshold: float = 0.3
    chaos_alert: bool = True
    complexity_spike_factor: float = 2.0
    modularity_threshold: float = 0.3
    adaptive_routing_threshold: float = 0.5
    capacity_growth_alert: bool = True


@dataclass
class EmergenceSnapshot:
    """Single-point-in-time emergence metrics."""

    step: int = 0
    timestamp: float = 0.0
    criticality: Optional[Dict[str, Any]] = None
    complexity: Optional[Dict[str, Any]] = None
    functional_zones: Optional[Dict[str, Any]] = None
    memory_routing: Optional[Dict[str, Any]] = None
    alerts: Optional[List[str]] = None


@dataclass
class EmergenceState:
    """Persistent monitoring state for save/load."""

    history: List[EmergenceSnapshot] = field(default_factory=list)
    baseline_metrics: Optional[Dict[str, float]] = None
    alert_log: List[Dict[str, Any]] = field(default_factory=list)
    step: int = 0


class EmergenceMonitor:
    """Real-time emergence behavior monitor for BIO-NN training.

    Tracks all emergence metrics during training, generates alerts
    for significant events, and produces visualization-ready data.

    Args:
        n_neurons: Number of neurons in the network.
        alert_config: Alert threshold configuration.
        log_dir: Directory for saving monitoring data. None = no disk I/O.
        alert_callbacks: List of callables invoked on each alert.
        analysis_interval: Run full analysis every N steps.
        criticality_buffer: Buffer size for criticality detector.
    """

    def __init__(
        self,
        n_neurons: int,
        alert_config: Optional[AlertConfig] = None,
        log_dir: Optional[str | Path] = None,
        alert_callbacks: Optional[List[Callable]] = None,
        analysis_interval: int = 50,
        criticality_buffer: int = 200,
    ) -> None:
        self.n_neurons = n_neurons
        self.alert_config = alert_config or AlertConfig()
        self.log_dir = Path(log_dir) if log_dir else None
        self.alert_callbacks = alert_callbacks or []
        self.analysis_interval = analysis_interval

        self.criticality_detector = CriticalityDetector(
            n_neurons=n_neurons,
            buffer_size=criticality_buffer,
            analysis_interval=analysis_interval,
        )

        self.state = EmergenceState()
        self._spike_buffer: List[np.ndarray] = []
        self._flow_buffer: List[np.ndarray] = []
        self._max_buffer = 1000

        if self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)

    def update(
        self,
        spikes: torch.Tensor | np.ndarray,
        step: Optional[int] = None,
        extra_info: Optional[Dict[str, Any]] = None,
    ) -> Optional[EmergenceSnapshot]:
        """Process one time step of spike data.

        Args:
            spikes: Vector (n_neurons,) or matrix (n_neurons, batch).
            step: Current training step. Auto-incremented if None.
            extra_info: Optional metadata to attach to the snapshot.

        Returns:
            EmergenceSnapshot if a full analysis was triggered, else None.
        """
        if isinstance(spikes, torch.Tensor):
            sp = spikes.detach().cpu().numpy()
        else:
            sp = np.asarray(spikes)

        # Average over batch if 2D
        if sp.ndim == 2:
            sp = sp.mean(axis=1)

        sp = sp.flatten()
        if len(sp) != self.n_neurons:
            raise ValueError(f"Expected {self.n_neurons} neurons, got {len(sp)}")

        # Update step
        if step is not None:
            self.state.step = step
        else:
            self.state.step += 1

        current_step = self.state.step
        timestamp = time.time()

        # Buffer spikes
        self._spike_buffer.append(sp.copy())
        if len(self._spike_buffer) > self._max_buffer:
            self._spike_buffer.pop(0)

        # Criticality detector (runs its own schedule)
        crit_report = self.criticality_detector.update(sp)

        # Full analysis at interval
        if current_step % self.analysis_interval == 0 and len(self._spike_buffer) >= 10:
            snapshot = self._run_full_analysis(
                current_step, timestamp, extra_info
            )
            self.state.history.append(snapshot)

            # Check for alerts
            self._check_alerts(snapshot)

            # Save to disk
            if self.log_dir:
                self._save_snapshot(snapshot)

            return snapshot

        return None

    def _run_full_analysis(
        self,
        step: int,
        timestamp: float,
        extra_info: Optional[Dict[str, Any]],
    ) -> EmergenceSnapshot:
        """Run all emergence analyses on buffered data."""
        spike_matrix = np.array(self._spike_buffer)

        # Criticality
        crit_report = analyze_criticality(spike_matrix)

        # Complexity
        comp_report = analyze_complexity(spike_matrix)

        # Functional zones
        func_report = analyze_functional_zones(spike_matrix)

        # Memory routing
        mem_report = analyze_memory_routing(
            spike_matrix,
            spike_history=self._spike_buffer[-10:] if len(self._spike_buffer) >= 10 else None,
        )

        # Track flow for adaptive routing
        if mem_report.flow_matrix is not None:
            self._flow_buffer.append(mem_report.flow_matrix.copy())
            if len(self._flow_buffer) > self._max_buffer:
                self._flow_buffer.pop(0)

        # Build snapshot
        snapshot = EmergenceSnapshot(
            step=step,
            timestamp=timestamp,
            criticality=self._report_to_dict(crit_report),
            complexity=self._report_to_dict(comp_report),
            functional_zones=self._report_to_dict(func_report),
            memory_routing=self._report_to_dict(mem_report),
            alerts=[],
        )

        return snapshot

    def _report_to_dict(self, report: Any) -> Dict[str, Any]:
        """Convert a dataclass report to a JSON-serializable dict."""
        d = asdict(report)
        # Convert numpy arrays to lists
        for key, val in d.items():
            if isinstance(val, np.ndarray):
                d[key] = val.tolist()
            elif isinstance(val, dict):
                for k2, v2 in val.items():
                    if isinstance(v2, np.ndarray):
                        val[k2] = v2.tolist()
        return d

    def _check_alerts(self, snapshot: EmergenceSnapshot) -> List[str]:
        """Check snapshot against alert thresholds. Returns alert messages."""
        alerts = []
        cfg = self.alert_config

        # Criticality alert
        if snapshot.criticality:
            dist = snapshot.criticality.get("distance_to_criticality", 1.0)
            if dist < cfg.criticality_threshold:
                msg = (
                    f"[CRITICALITY] Network near critical point at step {snapshot.step} "
                    f"(distance={dist:.4f})"
                )
                alerts.append(msg)

        # Chaos alert
        if snapshot.complexity and cfg.chaos_alert:
            if snapshot.complexity.get("is_chaotic", False):
                le = snapshot.complexity.get("largest_lyapunov", 0.0)
                msg = (
                    f"[CHAOS] Chaotic dynamics detected at step {snapshot.step} "
                    f"(Lyapunov={le:.4f})"
                )
                alerts.append(msg)

        # Complexity spike
        if snapshot.complexity and len(self.state.history) >= 3:
            current_nc = snapshot.complexity.get("neural_complexity", 0.0)
            recent_nc = [
                h.complexity.get("neural_complexity", 0.0)
                for h in self.state.history[-3:]
                if h.complexity
            ]
            if recent_nc:
                avg_nc = np.mean(recent_nc)
                if avg_nc > 0 and current_nc > avg_nc * cfg.complexity_spike_factor:
                    msg = (
                        f"[COMPLEXITY] Complexity spike at step {snapshot.step} "
                        f"({current_nc:.4f} vs avg {avg_nc:.4f})"
                    )
                    alerts.append(msg)

        # Modularity emergence
        if snapshot.functional_zones:
            mod = snapshot.functional_zones.get("modularity", 0.0)
            if mod > cfg.modularity_threshold:
                msg = (
                    f"[MODULARITY] High modularity detected at step {snapshot.step} "
                    f"(Q={mod:.4f})"
                )
                alerts.append(msg)

        # Adaptive routing
        if snapshot.memory_routing:
            adaptive = snapshot.memory_routing.get("adaptive_index", 0.0)
            if adaptive > cfg.adaptive_routing_threshold:
                msg = (
                    f"[ROUTING] Adaptive routing active at step {snapshot.step} "
                    f"(adaptive_index={adaptive:.4f})"
                )
                alerts.append(msg)

        # Capacity growth
        if snapshot.memory_routing and cfg.capacity_growth_alert:
            growth = snapshot.memory_routing.get("capacity_growth_rate", 0.0)
            if growth > 0.01:
                msg = (
                    f"[CAPACITY] Memory capacity growing at step {snapshot.step} "
                    f"(rate={growth:.4f})"
                )
                alerts.append(msg)

        if alerts:
            snapshot.alerts = alerts
            self.state.alert_log.extend(
                [{"step": snapshot.step, "timestamp": snapshot.timestamp, "message": a}]
                for a in alerts
            )
            for alert_msg in alerts:
                logger.warning(alert_msg)
                for callback in self.alert_callbacks:
                    try:
                        callback(alert_msg, snapshot)
                    except Exception as e:
                        logger.error(f"Alert callback failed: {e}")

        return alerts

    def _save_snapshot(self, snapshot: EmergenceSnapshot) -> None:
        """Save snapshot to disk."""
        if not self.log_dir:
            return

        snapshot_file = self.log_dir / f"emergence_step_{snapshot.step:08d}.json"
        with open(snapshot_file, "w") as f:
            json.dump(asdict(snapshot), f, indent=2, default=str)

    def get_history(self) -> List[EmergenceSnapshot]:
        """Return all historical snapshots."""
        return list(self.state.history)

    def get_metric_timeseries(self, metric_path: str) -> np.ndarray:
        """Extract a specific metric as a time series.

        Args:
            metric_path: Dot-separated path, e.g. 'criticality.branching_ratio'.

        Returns:
            Array of metric values over time.
        """
        values = []
        for snap in self.state.history:
            parts = metric_path.split(".")
            obj = asdict(snap) if hasattr(snap, "__dataclass_fields__") else snap
            for part in parts:
                if isinstance(obj, dict):
                    obj = obj.get(part, None)
                else:
                    obj = None
                    break
            values.append(obj)
        return np.array([v for v in values if v is not None])

    def get_alert_summary(self) -> Dict[str, int]:
        """Count alerts by type."""
        counts: Dict[str, int] = {}
        for entry in self.state.alert_log:
            msg = entry.get("message", "")
            # Extract tag
            if msg.startswith("["):
                tag = msg.split("]")[0] + "]"
                counts[tag] = counts.get(tag, 0) + 1
        return counts

    def generate_viz_data(self) -> Dict[str, Any]:
        """Generate data suitable for visualization/plotting.

        Returns:
            Dictionary with timeseries arrays for each metric category.
        """
        viz: Dict[str, Any] = {
            "steps": [],
            "timestamps": [],
        }

        for snap in self.state.history:
            viz["steps"].append(snap.step)
            viz["timestamps"].append(snap.timestamp)

            # Criticality metrics
            if snap.criticality:
                for key in ["branching_ratio", "susceptibility", "order_parameter",
                             "distance_to_criticality", "power_law_exponent"]:
                    full_key = f"crit_{key}"
                    if full_key not in viz:
                        viz[full_key] = []
                    viz[full_key].append(snap.criticality.get(key, None))

            # Complexity metrics
            if snap.complexity:
                for key in ["neural_complexity", "metastability", "entropy_rate",
                             "largest_lyapunov", "fisher_information"]:
                    full_key = f"comp_{key}"
                    if full_key not in viz:
                        viz[full_key] = []
                    viz[full_key].append(snap.complexity.get(key, None))

            # Functional zones
            if snap.functional_zones:
                for key in ["specialization_index", "modularity", "participation_ratio", "average_mi"]:
                    full_key = f"func_{key}"
                    if full_key not in viz:
                        viz[full_key] = []
                    viz[full_key].append(snap.functional_zones.get(key, None))

            # Memory routing
            if snap.memory_routing:
                for key in ["adaptive_index", "path_entropy", "consolidation_ratio",
                             "path_stability", "congestion_index"]:
                    full_key = f"mem_{key}"
                    if full_key not in viz:
                        viz[full_key] = []
                    viz[full_key].append(snap.memory_routing.get(key, None))

        # Convert to numpy
        for key in viz:
            viz[key] = np.array(viz[key])

        return viz

    def save_state(self, path: str | Path) -> None:
        """Save monitoring state to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        state_dict = {
            "step": self.state.step,
            "alert_log": self.state.alert_log,
            "history": [asdict(s) for s in self.state.history],
        }

        with open(path, "w") as f:
            json.dump(state_dict, f, indent=2, default=str)

        logger.info(f"Emergence monitor state saved to {path}")

    def load_state(self, path: str | Path) -> None:
        """Load monitoring state from disk."""
        path = Path(path)
        if not path.exists():
            logger.warning(f"No state file found at {path}")
            return

        with open(path) as f:
            state_dict = json.load(f)

        self.state.step = state_dict.get("step", 0)
        self.state.alert_log = state_dict.get("alert_log", [])
        self.state.history = [
            EmergenceSnapshot(**h) for h in state_dict.get("history", [])
        ]

        logger.info(f"Emergence monitor state loaded from {path} ({len(self.state.history)} snapshots)")

    def summary(self) -> Dict[str, Any]:
        """Generate a summary of all monitored emergence metrics."""
        if not self.state.history:
            return {"status": "no data"}

        latest = self.state.history[-1]
        summary: Dict[str, Any] = {
            "total_steps": self.state.step,
            "total_snapshots": len(self.state.history),
            "total_alerts": len(self.state.alert_log),
            "alert_summary": self.get_alert_summary(),
        }

        # Latest metrics
        if latest.criticality:
            summary["criticality"] = {
                "branching_ratio": latest.criticality.get("branching_ratio", 0),
                "is_critical": latest.criticality.get("is_critical", False),
                "distance": latest.criticality.get("distance_to_criticality", 1),
            }
        if latest.complexity:
            summary["complexity"] = {
                "neural_complexity": latest.complexity.get("neural_complexity", 0),
                "is_chaotic": latest.complexity.get("is_chaotic", False),
                "entropy_rate": latest.complexity.get("entropy_rate", 0),
            }
        if latest.functional_zones:
            summary["functional_zones"] = {
                "n_clusters": latest.functional_zones.get("n_clusters", 0),
                "modularity": latest.functional_zones.get("modularity", 0),
            }
        if latest.memory_routing:
            summary["memory_routing"] = {
                "adaptive_index": latest.memory_routing.get("adaptive_index", 0),
                "consolidation_ratio": latest.memory_routing.get("consolidation_ratio", 0),
            }

        return summary
