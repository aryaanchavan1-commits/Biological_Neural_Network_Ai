"""Controllability analysis for spiking neural networks.

Implements tools to characterize how responsive a biological neural
network is to external stimuli, how stable it is under bounded
perturbations, whether it can be steered toward desired states, and
how to detect emergency shutdown conditions.

In the context of biological superintelligence research, controllability
is paramount: a system we cannot control is a system we cannot verify
as safe.  These tools answer questions like:

- "Can we reliably steer the network to a desired activation pattern?"
- "How large a perturbation can the network absorb before diverging?"
- "Is there a reachable state where the network becomes uncontrollable?"
- "Can we detect an emergency condition and halt the network?"

The analysis uses control-theoretic concepts adapted for discrete-time
spiking dynamics: reachability analysis, Lyapunov stability, and
contraction metrics.

Typical usage::

    analyzer = ControllabilityAnalyzer(model)
    response = analyzer.stimulus_response(stimulus_bank)
    stability = analyzer.stability_analysis(perturbation_scales)
    ctrl = analyzer.controllability_metrics(state_pairs)
    bounds = analyzer.estimate_safety_bounds(model_state)
    shutdown = analyzer.detect_emergency_shutdown(obs_sequence)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


class ControllabilityAnalyzer:
    """Controllability and stability analysis toolkit for SNNs.

    Evaluates whether a spiking network can be reliably controlled,
    how stable its dynamics are, and when emergency shutdown should
    be triggered.

    Args:
        model: The SNN model to analyze.
        config: Optional configuration dictionary. Supported keys:

            - ``device`` (str | torch.device): Compute device.
            - ``time_steps`` (int): Simulation steps. Default 20.
            - ``stability_window`` (int): Steps to observe for stability
              assessment. Default 50.
            - ``shutdown_threshold`` (float): z-score threshold for
              emergency detection. Default 4.0.
    """

    def __init__(
        self,
        model: nn.Module,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.model = model
        self.config = config or {}
        self.device = self._resolve_device()
        self.time_steps: int = self.config.get("time_steps", 20)
        self.stability_window: int = self.config.get("stability_window", 50)
        self.shutdown_threshold: float = self.config.get("shutdown_threshold", 4.0)

    def _resolve_device(self) -> torch.device:
        cfg_dev = self.config.get("device")
        if cfg_dev is not None:
            return torch.device(cfg_dev)
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def _collect_activations(self, x: torch.Tensor) -> torch.Tensor:
        """Run forward pass and collect spike activations."""
        self.model.eval()
        all_spikes: List[torch.Tensor] = []
        with torch.no_grad():
            for _ in range(self.time_steps):
                out = self.model(x.to(self.device))
                if isinstance(out, dict):
                    spike_key = "spikes"
                    if spike_key in out:
                        all_spikes.append(out[spike_key])
                    elif "output" in out:
                        all_spikes.append(out["output"])
                elif isinstance(out, torch.Tensor):
                    all_spikes.append(out)
        if all_spikes:
            return torch.stack(all_spikes, dim=0)
        return torch.zeros(1)

    # ------------------------------------------------------------------
    # 1. Stimulus-response characterization
    # ------------------------------------------------------------------

    def stimulus_response(
        self,
        stimuli: torch.Tensor,
        n_repetitions: int = 5,
    ) -> Dict[str, Any]:
        """Characterize the network's response profile to a bank of stimuli.

        For each stimulus, runs the network ``n_repetitions`` times and
        measures: mean response magnitude, response variance (reliability),
        onset latency (time to first spike), and sustained vs transient
        response profile.

        A controllable network should have:
        - High stimulus-discriminability (different stimuli → different responses).
        - Low trial-to-trial variability (same stimulus → same response).
        - Bounded onset latency (no pathological delays).

        Args:
            stimuli: Tensor ``(n_stimuli, *input_shape)`` — bank of
                different input patterns.
            n_repetitions: How many times to present each stimulus to
                measure trial-to-trial variability.

        Returns:
            Dictionary with keys:

            - ``mean_responses``: ``(n_stimuli, n_neurons)`` mean activation.
            - ``response_variability``: ``(n_stimuli, n_neurons)`` std across
              repetitions.
            - ``onset_latency``: ``(n_stimuli, n_neurons)`` mean time to
              first spike.
            - ``discriminability``: Scalar — ratio of inter-stimulus to
              intra-stimulus variance (higher = more controllable).
        """
        self.model.eval()
        n_stimuli = stimuli.shape[0]
        responses: List[torch.Tensor] = []
        latencies: List[torch.Tensor] = []

        for rep in range(n_repetitions):
            rep_responses = []
            rep_latencies = []
            for i in range(n_stimuli):
                x = stimuli[i : i + 1]  # (1, *input_shape)
                spikes = self._collect_activations(x)  # (time, 1, neurons)
                if spikes.ndim >= 3:
                    # Mean firing rate per neuron
                    rate = spikes[:, 0, :].float().mean(dim=0)  # (neurons,)
                    rep_responses.append(rate)

                    # Onset latency
                    first_spike_time = spikes[:, 0, :].float()
                    cum = first_spike_time.cumsum(dim=0)
                    has_fired = cum > 0
                    if has_fired.any(dim=0).all():
                        latency = has_fired.float().argmax(dim=0).float()
                    else:
                        latency = torch.full((spikes.shape[-1],), float(self.time_steps))
                    rep_latencies.append(latency)
                else:
                    rep_responses.append(torch.zeros(1))
                    rep_latencies.append(torch.zeros(1))

            responses.append(torch.stack(rep_responses))  # (n_stimuli, neurons)
            latencies.append(torch.stack(rep_latencies))

        responses_tensor = torch.stack(responses)  # (reps, n_stimuli, neurons)
        latencies_tensor = torch.stack(latencies)  # (reps, n_stimuli, neurons)

        mean_responses = responses_tensor.mean(dim=0)  # (n_stimuli, neurons)
        response_variability = responses_tensor.std(dim=0)  # (n_stimuli, neurons)
        mean_latencies = latencies_tensor.mean(dim=0)

        # Discriminability: between-stimulus variance / within-stimulus variance
        between_var = mean_responses.var(dim=0).mean()
        within_var = response_variability.mean(dim=0).mean()
        discriminability = (between_var / (within_var + 1e-8)).item()

        return {
            "mean_responses": mean_responses.cpu(),
            "response_variability": response_variability.cpu(),
            "onset_latency": mean_latencies.cpu(),
            "discriminability": discriminability,
        }

    # ------------------------------------------------------------------
    # 2. Stability analysis
    # ------------------------------------------------------------------

    def stability_analysis(
        self,
        x: torch.Tensor,
        perturbation_scales: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        """Analyze network stability under bounded input perturbations.

        Applies Gaussian noise of increasing magnitude to the input and
        measures how the output diverges.  A stable network should show
        output divergence that scales gracefully with input perturbation
        — not catastrophically.

        This is a Lyapunov-style analysis: if the output divergence
        grows slower than the input perturbation, the network is
        locally stable.  If it grows faster, the network amplifies
        noise — a red flag for safety.

        Args:
            x: Reference input ``(1, *input_shape)``.
            perturbation_scales: List of noise std values to test.
                Default ``[0.01, 0.05, 0.1, 0.2, 0.5, 1.0]``.

        Returns:
            Dictionary with keys:

            - ``perturbation_scales``: The tested scales.
            - ``output_divergence``: ``(n_scales,)`` L2 distance between
              perturbed and baseline outputs.
            - ``divergence_ratio``: ``(n_scales,)`` ratio of output
              divergence to input perturbation.  Values > 1 indicate
              amplification (unstable).
            - ``lyapunov_exponent``: Estimated largest Lyapunov exponent
              from the divergence curve.  Positive = unstable.
            - ``is_stable``: ``True`` if the network is locally stable
              (divergence ratio < 1 across all tested scales).
        """
        if perturbation_scales is None:
            perturbation_scales = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]

        self.model.eval()
        x_dev = x.to(self.device).detach()

        # Baseline output
        with torch.no_grad():
            base_out = self.model(x_dev)
            if isinstance(base_out, dict):
                base_out = base_out.get("output", base_out.get("spikes", next(iter(base_out.values()))))
            base_out = base_out.float()

        output_divs = []
        divergence_ratios = []

        for scale in perturbation_scales:
            divs_for_scale = []
            for _ in range(10):  # multiple random perturbations
                noise = torch.randn_like(x_dev.float()) * scale
                noisy_x = x_dev + noise

                with torch.no_grad():
                    out = self.model(noisy_x)
                    if isinstance(out, dict):
                        out = out.get("output", out.get("spikes", next(iter(out.values()))))
                    out = out.float()

                l2_dist = (out - base_out).pow(2).sum().sqrt().item()
                divs_for_scale.append(l2_dist)

            mean_div = np.mean(divs_for_scale)
            output_divs.append(mean_div)
            divergence_ratios.append(mean_div / (scale + 1e-8))

        output_divs_t = torch.tensor(output_divs)
        scales_t = torch.tensor(perturbation_scales)

        # Estimate Lyapunov exponent from log-log slope
        log_scales = torch.log(scales_t + 1e-8)
        log_divs = torch.log(output_divs_t + 1e-8)
        # Linear regression: log(div) = lambda * log(scale) + c
        if len(log_scales) > 1:
            x_mean = log_scales.mean()
            y_mean = log_divs.mean()
            slope = ((log_scales - x_mean) * (log_divs - y_mean)).sum() / ((log_scales - x_mean).pow(2).sum() + 1e-8)
            lyapunov_exponent = slope.item()
        else:
            lyapunov_exponent = 0.0

        is_stable = all(r < 1.0 for r in divergence_ratios)

        return {
            "perturbation_scales": perturbation_scales,
            "output_divergence": output_divs_t.cpu(),
            "divergence_ratio": torch.tensor(divergence_ratios).cpu(),
            "lyapunov_exponent": lyapunov_exponent,
            "is_stable": is_stable,
        }

    # ------------------------------------------------------------------
    # 3. Controllability metrics
    # ------------------------------------------------------------------

    def controllability_metrics(
        self,
        input_a: torch.Tensor,
        input_b: torch.Tensor,
        n_steering_steps: int = 10,
    ) -> Dict[str, Any]:
        """Measure how well the network can be steered between states.

        Given two input vectors, applies a proportional controller in
        input space and observes whether the network's output trajectory
        converges from the output of input_a toward the output of input_b.

        The controller iteratively perturbs the input toward a direction
        that reduces the output-space distance to the target.

        Controllability metrics:
        - ``reachability``: Can the network reach the target output?
        - ``steering_cost``: Total input energy required.
        - ``convergence_rate``: How quickly the output approaches the target.

        Args:
            input_a: Starting input ``(1, n_input_features)``.
            input_b: Target input ``(1, n_input_features)``.
            n_steering_steps: Number of control steps to attempt steering.

        Returns:
            Dictionary with keys:

            - ``reached``: ``True`` if final output is within tolerance of target.
            - ``final_distance``: L2 distance from target output after steering.
            - ``trajectory``: ``(n_steps+1, n_output_features)`` output trajectory.
            - ``control_effort``: ``(n_steps,)`` L2 norm of each control input.
            - ``convergence_rate``: Exponential decay rate of distance to target.
        """
        self.model.eval()
        current_input = input_a.to(self.device).detach().float()
        current_input.requires_grad_(False)

        # Get target output
        with torch.no_grad():
            target_out = self.model(input_b.to(self.device))
            if isinstance(target_out, dict):
                target_out = target_out.get("output", target_out.get("spikes", next(iter(target_out.values()))))
            target_out = target_out.float()

        # Get initial output
        with torch.no_grad():
            init_out = self.model(current_input)
            if isinstance(init_out, dict):
                init_out = init_out.get("output", init_out.get("spikes", next(iter(init_out.values()))))
            init_out = init_out.float()

        current_output = init_out.detach()
        trajectory = [current_output.squeeze(0).cpu()]
        control_efforts = []
        distances = []

        gain = 0.1
        tolerance = 0.1

        for step in range(n_steering_steps):
            error = target_out - current_output
            dist = error.pow(2).sum().sqrt().item()
            distances.append(dist)

            if dist < tolerance:
                break

            # Compute gradient: which direction in input space reduces output error?
            current_input_grad = current_input.clone().requires_grad_(True)
            out = self.model(current_input_grad)
            if isinstance(out, dict):
                out = out.get("output", out.get("spikes", next(iter(out.values()))))
            loss = (out.float() - target_out).pow(2).sum()
            loss.backward()

            # Steer input in direction that reduces output error
            with torch.no_grad():
                grad = current_input_grad.grad
                if grad is not None:
                    control = -gain * grad.sign()
                else:
                    control = gain * error  # fallback
                current_input = (current_input + control).detach()

                out = self.model(current_input)
                if isinstance(out, dict):
                    out = out.get("output", out.get("spikes", next(iter(out.values()))))
                current_output = out.float()

            trajectory.append(current_output.squeeze(0).cpu())
            control_efforts.append(control.pow(2).sum().sqrt().item())

        final_dist = (current_output - target_out).pow(2).sum().sqrt().item()
        reached = final_dist < tolerance

        # Convergence rate: fit exponential to distance curve
        if len(distances) > 2:
            log_dists = np.log(np.array(distances) + 1e-8)
            t = np.arange(len(log_dists))
            # Linear fit on log scale: log(d) = -rate * t + c
            if len(t) > 1:
                t_mean = t.mean()
                d_mean = log_dists.mean()
                slope = np.sum((t - t_mean) * (log_dists - d_mean)) / (np.sum((t - t_mean) ** 2) + 1e-8)
                convergence_rate = -slope
            else:
                convergence_rate = 0.0
        else:
            convergence_rate = 0.0

        return {
            "reached": reached,
            "final_distance": final_dist,
            "trajectory": torch.stack(trajectory).cpu() if trajectory else torch.tensor([]),
            "control_effort": torch.tensor(control_efforts).cpu() if control_efforts else torch.tensor([]),
            "convergence_rate": convergence_rate,
        }

    # ------------------------------------------------------------------
    # 4. Safety bounds estimation
    # ------------------------------------------------------------------

    def estimate_safety_bounds(
        self,
        x: torch.Tensor,
        output_bounds: Optional[Tuple[float, float]] = None,
        n_samples: int = 200,
    ) -> Dict[str, Any]:
        """Estimate the safe operating region of input perturbations.

        Uses random sampling to find the maximum input perturbation
        magnitude (L2 norm) that keeps the output within specified
        bounds.  This defines the "safety envelope" — the region
        within which the network's behavior is guaranteed (with some
        confidence) to remain acceptable.

        For biological superintelligence safety, this is critical:
        it tells us how much environmental noise or adversarial
        perturbation the system can tolerate before its behavior
        becomes unpredictable.

        Args:
            x: Reference input ``(1, *input_shape)``.
            output_bounds: ``(min, max)`` acceptable output range.
                Default ``(-1.0, 1.0)``.
            n_samples: Number of random perturbations to test.

        Returns:
            Dictionary with keys:

            - ``safe_radius``: Maximum L2 perturbation that keeps output
              within bounds for ``n_samples`` random directions.
            - ``max_safe_fraction``: Fraction of tested perturbations
              within the safe radius that actually stayed in bounds.
            - ``perturbation_magnitudes``: ``(n_samples,)`` tested L2 norms.
            - ``output_magnitudes``: ``(n_samples,)`` corresponding output
              L2 norms.
            - ``phase_transition``: Estimated magnitude where output
              starts exceeding bounds (50% failure point).
        """
        if output_bounds is None:
            output_bounds = (-1.0, 1.0)

        self.model.eval()
        x_dev = x.to(self.device).detach().float()
        flat_x = x_dev.reshape(1, -1)
        input_dim = flat_x.shape[1]

        # Baseline output
        with torch.no_grad():
            base_out = self.model(x_dev)
            if isinstance(base_out, dict):
                base_out = base_out.get("output", base_out.get("spikes", next(iter(base_out.values()))))
            base_out = base_out.float().reshape(1, -1)

        low, high = output_bounds
        input_norms = []
        output_norms = []
        in_bounds_count = 0

        for _ in range(n_samples):
            # Random direction in input space
            direction = torch.randn_like(flat_x)
            direction = direction / (direction.norm() + 1e-8)

            # Random magnitude
            magnitude = torch.rand(1).item() * 2.0
            perturbation = direction * magnitude

            perturbed = (flat_x + perturbation).reshape_as(x_dev)

            with torch.no_grad():
                out = self.model(perturbed)
                if isinstance(out, dict):
                    out = out.get("output", out.get("spikes", next(iter(out.values()))))
                out = out.float().reshape(1, -1)

            in_norm = magnitude
            out_norm = (out - base_out).pow(2).sum().sqrt().item()

            input_norms.append(in_norm)
            output_norms.append(out_norm)

            # Check if output stays within bounds
            if out.min().item() >= low and out.max().item() <= high:
                in_bounds_count += 1

        input_norms_t = torch.tensor(input_norms)
        output_norms_t = torch.tensor(output_norms)

        # Safe radius: largest tested magnitude where all samples are in bounds
        sorted_indices = input_norms_t.argsort()
        sorted_norms = input_norms_t[sorted_indices]
        sorted_outputs = output_norms_t[sorted_indices]

        # Find phase transition: where output first exceeds bounds
        failure_mask = sorted_outputs > (high - low)
        if failure_mask.any():
            phase_idx = failure_mask.nonzero(as_tuple=False)[0].item()
            safe_radius = sorted_norms[phase_idx].item() * 0.9  # 10% margin
            phase_transition = sorted_norms[phase_idx].item()
        else:
            safe_radius = sorted_norms.max().item()
            phase_transition = float("inf")

        max_safe_fraction = in_bounds_count / n_samples

        return {
            "safe_radius": safe_radius,
            "max_safe_fraction": max_safe_fraction,
            "perturbation_magnitudes": input_norms_t.cpu(),
            "output_magnitudes": output_norms_t.cpu(),
            "phase_transition": phase_transition,
        }

    # ------------------------------------------------------------------
    # 5. Emergency shutdown detection
    # ------------------------------------------------------------------

    def detect_emergency_shutdown(
        self,
        observations: torch.Tensor,
        baseline_stats: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """Detect anomalous network behavior requiring emergency halt.

        Monitors a sequence of activation observations and flags
        emergency conditions based on:
        1. Activation magnitude exceeding learned normal range.
        2. Rate of change (derivative) exceeding bounds.
        3. Entropy of spike patterns indicating loss of structure.

        A biological superintelligence system must have a reliable
        kill switch.  This detector provides the signal for that switch
        by identifying when the network has entered a regime outside
        its trained operating envelope.

        Args:
            observations: Tensor ``(time, n_features)`` — sequence of
                network activation snapshots.
            baseline_stats: Optional pre-computed statistics of normal
                behavior.  Keys: ``mean``, ``std``, ``max_rate``.
                If ``None``, estimated from the first 20% of observations.

        Returns:
            Dictionary with keys:

            - ``shutdown_requested``: ``True`` if any emergency condition
              was detected.
            - ``anomaly_scores``: ``(time,)`` z-score of activation
              magnitude at each time step.
            - ``rate_violations``: ``(time,)`` bool — where rate of
              change exceeded bounds.
            - ``entropy_values``: ``(time,)`` spike pattern entropy.
            - ``emergency_indices``: List of time steps where shutdown
              was triggered.
        """
        obs = observations.detach().float()
        T = obs.shape[0]

        # Baseline statistics from first 20% or provided
        if baseline_stats is None:
            split = max(T // 5, 1)
            baseline_data = obs[:split]
            baseline_mean = baseline_data.mean(dim=0)
            baseline_std = baseline_data.std(dim=0) + 1e-8
        else:
            baseline_mean = torch.tensor(baseline_stats.get("mean", 0.0))
            baseline_std = torch.tensor(baseline_stats.get("std", 1.0)) + 1e-8
            if baseline_mean.ndim == 0:
                baseline_mean = baseline_mean.expand(obs.shape[-1])
                baseline_std = baseline_std.expand(obs.shape[-1])

        # 1. Activation magnitude anomaly (z-score)
        deviations = (obs - baseline_mean) / baseline_std
        anomaly_scores = deviations.abs().mean(dim=-1)  # (T,)

        # 2. Rate of change detection
        if T > 1:
            diffs = obs[1:] - obs[:-1]
            rates = diffs.abs().mean(dim=-1)
            rate_threshold = rates.mean() + self.shutdown_threshold * (rates.std() + 1e-8)
            rate_violations = torch.zeros(T, dtype=torch.bool)
            rate_violations[1:] = rates > rate_threshold
        else:
            rate_violations = torch.zeros(T, dtype=torch.bool)

        # 3. Spike pattern entropy (Shannon entropy of binned activations)
        entropy_values = torch.zeros(T)
        n_bins = 10
        for t in range(T):
            # Bin activations into histogram
            obs_t = obs[t]
            if obs_t.numel() > 1:
                hist = torch.histc(obs_t, bins=n_bins)
                hist = hist / (hist.sum() + 1e-8)
                # Shannon entropy
                log_hist = torch.log(hist + 1e-8)
                entropy_values[t] = -(hist * log_hist).sum()
            else:
                entropy_values[t] = 0.0

        # Determine threshold from baseline entropy
        if T > 5:
            baseline_entropy = entropy_values[:max(T // 5, 1)].mean()
            entropy_std = entropy_values[:max(T // 5, 1)].std() + 1e-8
        else:
            baseline_entropy = entropy_values.mean()
            entropy_std = entropy_values.std() + 1e-8

        entropy_violations = (entropy_values - baseline_entropy).abs() > self.shutdown_threshold * entropy_std

        # Emergency conditions
        activation_anomaly = anomaly_scores > self.shutdown_threshold
        emergency_mask = activation_anomaly | rate_violations | entropy_violations
        emergency_indices = emergency_mask.nonzero(as_tuple=False).squeeze(-1).tolist()
        if isinstance(emergency_indices, int):
            emergency_indices = [emergency_indices]

        return {
            "shutdown_requested": len(emergency_indices) > 0,
            "anomaly_scores": anomaly_scores.cpu(),
            "rate_violations": rate_violations.cpu(),
            "entropy_values": entropy_values.cpu(),
            "emergency_indices": emergency_indices,
        }
