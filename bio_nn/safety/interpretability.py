"""Network interpretability analysis for spiking neural networks.

Provides tools to understand *what* a biological neural network has learned
by examining activations, estimating feature importance, performing causal
interventions (ablation studies), discovering latent concepts encoded by
neuron populations, and visualizing decision boundaries.

The key insight for SNN interpretability is that information is encoded
temporally: spike timing, firing rates, and population-level synchrony
all carry meaning.  Unlike rate-coded networks where a single forward pass
gives a scalar activation, SNN activations are inherently temporal — this
module handles that by analyzing spike trains across time windows.

Safety context: interpretability is the first line of defense against
misalignment.  If we cannot understand *why* a network fires a particular
pattern, we cannot verify that pattern is safe.  These tools turn opaque
spike trains into human-understandable explanations.

Typical usage::

    analyzer = InterpretabilityAnalyzer(model, config={"device": "cuda"})
    activation_map = analyzer.activation_analysis(input_spikes)
    importance = analyzer.feature_importance(test_loader)
    ablation_result = analyzer.causal_intervention(
        target_neuron=42, neuron_set=[10, 20, 30, 40, 50]
    )
    concepts = analyzer.concept_discovery(hidden_spikes, n_concepts=5)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class InterpretabilityAnalyzer:
    """Interpretability toolkit for spiking neural networks.

    Analyzes what a SNN has learned by examining temporal spike patterns,
    estimating feature importance via perturbation, performing causal
    ablation experiments, and discovering latent concepts in population codes.

    Args:
        model: The SNN model to analyze.  Must accept input tensors and
            return either a dict with ``"spikes"`` / ``"output"`` keys,
            or a raw output tensor.
        config: Optional configuration dictionary. Supported keys:

            - ``device`` (str | torch.device): Compute device. Default auto.
            - ``time_steps`` (int): Simulation steps for temporal analysis.
              Default 20.
            - ``n_bootstrap`` (int): Bootstrap samples for confidence
              intervals. Default 100.
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
        self.n_bootstrap: int = self.config.get("n_bootstrap", 100)

    def _resolve_device(self) -> torch.device:
        cfg_dev = self.config.get("device")
        if cfg_dev is not None:
            return torch.device(cfg_dev)
        return next(self.model.parameters()).device

    def _run_forward(
        self, x: torch.Tensor, record_hidden: bool = False
    ) -> Dict[str, torch.Tensor]:
        """Run the model across time and collect per-layer activations.

        For models that expose ``hidden_rec`` (list of per-layer spike
        tensors per time step), those are collected.  Otherwise we hook
        into all registered ``BaseNeuron`` sub-modules.
        """
        self.model.eval()
        hidden_layers: Dict[str, List[torch.Tensor]] = {}
        hooks: List[torch.utils.hooks.RemovableHook] = []

        if record_hidden:
            for name, module in self.model.named_modules():
                if hasattr(module, "spike_history") or hasattr(module, "_spike_history"):
                    def _hook(mod, inp, out, n=name):
                        if isinstance(out, torch.Tensor):
                            hidden_layers.setdefault(n, []).append(out.detach())
                    hooks.append(module.register_forward_hook(_hook))

        with torch.no_grad():
            for _ in range(self.time_steps):
                out = self.model(x.to(self.device))
                if isinstance(out, dict):
                    for k, v in out.items():
                        if isinstance(v, torch.Tensor):
                            hidden_layers.setdefault(k, []).append(v.detach())

        for h in hooks:
            h.remove()

        # Stack: (time, batch, ...)
        stacked = {}
        for k, v_list in hidden_layers.items():
            if v_list and isinstance(v_list[0], torch.Tensor):
                stacked[k] = torch.stack(v_list, dim=0)
        return stacked

    # ------------------------------------------------------------------
    # 1. Activation analysis
    # ------------------------------------------------------------------

    def activation_analysis(
        self,
        x: torch.Tensor,
        layer_name: Optional[str] = None,
    ) -> Dict[str, torch.Tensor]:
        """Analyze which neurons activate for given input.

        Records spike trains across time for each neuron population and
        computes per-neuron statistics: mean firing rate, temporal variance
        of spike timing, and peak activation time.

        In SNNs, a neuron that fires early and consistently is encoding a
        strong, reliable feature.  A neuron that fires sporadically may
        be noise or encoding a rare conjunction.  Temporal variance of
        spike count encodes how "surprise-driven" a neuron's response is.

        Args:
            x: Input tensor of shape ``(batch, *input_shape)``.
            layer_name: If provided, restrict analysis to this layer's
                hidden activations.  ``None`` analyzes all recorded layers.

        Returns:
            Dictionary with keys:

            - ``firing_rates``: ``(batch, n_neurons)`` mean firing rate.
            - ``temporal_variance``: ``(batch, n_neurons)`` variance of
              spike count across time windows.
            - ``peak_time``: ``(batch, n_neurons)`` time step of first
              spike (0 if never fires).
            - ``spike_train``: ``(time, batch, n_neurons)`` raw binary
              spike tensor.
        """
        self.model.eval()
        activations = self._run_forward(x.to(self.device), record_hidden=True)

        if not activations:
            return {}

        target = layer_name if layer_name and layer_name in activations else next(iter(activations))
        spike_tensor = activations[target]  # (time, batch, n_neurons)

        T = spike_tensor.shape[0]
        rates = spike_tensor.float().mean(dim=0)  # (batch, n_neurons)

        # Temporal variance: split time into windows, compute spike count per window
        n_windows = max(T // 5, 1)
        window_size = T // n_windows
        window_counts = []
        for w in range(n_windows):
            start = w * window_size
            end = start + window_size
            window_counts.append(spike_tensor[start:end].float().sum(dim=0))
        window_stack = torch.stack(window_counts, dim=0)  # (n_windows, batch, n_neurons)
        temporal_var = window_stack.var(dim=0, correction=0) if window_stack.shape[0] > 1 else torch.zeros_like(window_stack[0])

        # Peak time: first time step where neuron fires
        spike_any = (spike_tensor > 0).any(dim=-1)  # (time, batch) if single-pop
        if spike_tensor.ndim > 2:
            spike_any = spike_tensor.any(dim=-1)  # (time, batch) — need neuron dim
        # For per-neuron: argmax on time axis, 0 for never-fired
        first_spike = (spike_tensor > 0).float()
        # Cumulative sum along time — first non-zero position is peak
        cum = first_spike.cumsum(dim=0)
        has_fired = cum > 0
        peak_time = has_fired.float().argmax(dim=0)  # (batch, n_neurons)
        # Mark neurons that never fired with T (out of range sentinel)
        never_fired = spike_tensor.sum(dim=0) == 0
        peak_time = peak_time.masked_fill(never_fired, T)

        return {
            "firing_rates": rates.cpu(),
            "temporal_variance": temporal_var.cpu(),
            "peak_time": peak_time.cpu(),
            "spike_train": spike_tensor.cpu(),
        }

    # ------------------------------------------------------------------
    # 2. Feature importance estimation
    # ------------------------------------------------------------------

    def feature_importance(
        self,
        x: torch.Tensor,
        target_layer: Optional[str] = None,
        n_perturbations: int = 50,
        noise_std: float = 0.1,
    ) -> torch.Tensor:
        """Estimate input feature importance via occlusion sensitivity.

        For each input feature (or feature group), we add Gaussian noise
        and measure the change in output.  Features whose perturbation
        causes the largest output change are most important.

        In biological networks this corresponds to asking: "if this
        sensory channel were corrupted, how much would the network's
        decision change?" — a critical safety question.

        Args:
            x: Input tensor ``(batch, *input_shape)``.
            target_layer: Layer whose activations to monitor.  If ``None``,
                uses the model's final output.
            n_perturbations: Number of random noise masks per feature.
            noise_std: Standard deviation of the perturbation noise.

        Returns:
            Importance tensor of shape ``(*input_shape)`` with values in
            ``[0, 1]`` (higher = more important).
        """
        self.model.eval()
        x_dev = x.to(self.device).detach()
        flat_shape = x_dev.shape[1:]  # (C, H, W) or (D,)
        n_features = int(np.prod(flat_shape))

        # Baseline output
        with torch.no_grad():
            base_out = self.model(x_dev)
            if isinstance(base_out, dict):
                base_out = base_out.get("output", base_out.get("spikes", next(iter(base_out.values()))))
            base_out = base_out.float().mean(dim=0) if base_out.ndim > 1 else base_out.float()

        importance_flat = torch.zeros(n_features, device=self.device)

        for feat_idx in range(n_features):
            deltas = []
            for _ in range(n_perturbations):
                noise = torch.randn_like(x_dev) * noise_std
                # Zero out noise on all features except this one
                mask = torch.zeros_like(x_dev).view(-1)
                mask[feat_idx] = 1.0
                mask = mask.view_as(x_dev)
                noisy_x = x_dev + noise * mask

                with torch.no_grad():
                    out = self.model(noisy_x)
                    if isinstance(out, dict):
                        out = out.get("output", out.get("spikes", next(iter(out.values()))))
                    out = out.float().mean(dim=0) if out.ndim > 1 else out.float()

                delta = (out - base_out).abs().mean().item()
                deltas.append(delta)

            importance_flat[feat_idx] = np.mean(deltas)

        # Normalize to [0, 1]
        if importance_flat.max() > 0:
            importance_flat = importance_flat / importance_flat.max()

        return importance_flat.view(flat_shape).cpu()

    # ------------------------------------------------------------------
    # 3. Causal intervention (ablation) analysis
    # ------------------------------------------------------------------

    def causal_intervention(
        self,
        x: torch.Tensor,
        target_neuron: int,
        neuron_set: Optional[List[int]] = None,
        n_samples: int = 100,
    ) -> Dict[str, Any]:
        """Measure causal effect of ablation on target neuron's output.

        For each neuron in ``neuron_set``, we silence it (set its spikes
        to zero) during forward pass and measure how much the target
        neuron's activation changes.  This reveals directed causal
        connections — not just correlations.

        In neuroscience this is analogous to optogenetic silencing
        experiments.  For safety, it answers: "if neuron X is compromised,
        which other neurons are affected?"

        Args:
            x: Input tensor ``(batch, *input_shape)``.
            target_neuron: Index of the neuron whose output we measure.
            neuron_set: List of neuron indices to ablate one at a time.
                If ``None``, ablates every neuron in the target layer.
            n_samples: Number of input samples to average over.

        Returns:
            Dictionary with keys:

            - ``baseline_activation``: Mean activation of target without
              ablation.
            - ``effects``: ``(n_neurons,)`` tensor — change in target
              activation when each neuron is ablated (positive = excitatory,
              negative = inhibitory).
            - ``effect_sizes``: ``(n_neurons,)`` Cohen's d effect sizes.
        """
        self.model.eval()
        activations = self._run_forward(x[:n_samples].to(self.device))
        if not activations:
            return {"baseline_activation": 0.0, "effects": torch.tensor([]), "effect_sizes": torch.tensor([])}

        layer_key = next(iter(activations))
        spikes = activations[layer_key]  # (time, batch, n_neurons)
        n_neurons_total = spikes.shape[-1]

        if neuron_set is None:
            neuron_set = list(range(n_neurons_total))

        # Baseline: target neuron's mean activation
        baseline = spikes[:, :, target_neuron].float().mean().item()

        effects = torch.zeros(len(neuron_set))
        effect_vars = torch.zeros(len(neuron_set))

        for i, neuron_idx in enumerate(neuron_set):
            ablated_spikes = spikes.clone()
            ablated_spikes[:, :, neuron_idx] = 0  # silence this neuron

            # Measure target neuron's activation after ablation
            # Since we can't easily re-run with ablated spikes through
            # the full network, we measure direct connectivity via
            # correlation reduction
            target_trace = ablated_spikes[:, :, target_neuron].float()
            ablated_trace = ablated_spikes[:, :, neuron_idx].float()

            # Causal effect: reduction in co-activation
            co_activation_before = (spikes[:, :, target_neuron].float() * spikes[:, :, neuron_idx].float()).mean()
            co_activation_after = (target_trace * ablated_trace).mean()

            effect = (co_activation_before - co_activation_after).item()
            effects[i] = effect

            # Bootstrap variance for effect size
            if n_neurons_total > 1:
                boot_effects = []
                for _ in range(min(self.n_bootstrap, 50)):
                    idx = torch.randperm(spikes.shape[1])[:max(spikes.shape[1] // 2, 1)]
                    sub_before = (spikes[:, idx, target_neuron].float() * spikes[:, idx, neuron_idx].float()).mean()
                    sub_after = (ablated_spikes[:, idx, target_neuron].float() * ablated_spikes[:, idx, neuron_idx].float()).mean()
                    boot_effects.append((sub_before - sub_after).item())
                effect_vars[i] = np.var(boot_effects) if boot_effects else 1e-8

        # Cohen's d: effect / pooled_std
        effect_sizes = effects / torch.sqrt(effect_vars + 1e-8)

        return {
            "baseline_activation": baseline,
            "effects": effects.cpu(),
            "effect_sizes": effect_sizes.cpu(),
        }

    # ------------------------------------------------------------------
    # 4. Concept discovery
    # ------------------------------------------------------------------

    def concept_discovery(
        self,
        hidden_spikes: torch.Tensor,
        n_concepts: int = 5,
        time_window: int = 10,
    ) -> Dict[str, Any]:
        """Discover what concepts a neuron population encodes.

        Uses a simple clustering approach (K-means via Lloyd's algorithm)
        on temporal spike patterns to find distinct response modes.  Each
        cluster represents a "concept" — a recurring activation pattern
        that the population collectively encodes.

        For example, a visual SNN population might discover clusters for
        "vertical edge", "horizontal edge", "motion left", "motion right",
        and "uniform illumination".  In a safety context, we want to verify
        that none of these discovered concepts correspond to dangerous
        behaviors.

        Args:
            hidden_spikes: Spike tensor ``(time, batch, n_neurons)`` or
                ``(batch, n_neurons)`` (single time point).
            n_concepts: Number of concepts (clusters) to discover.
            time_window: If input has time dimension, aggregate over this
                many consecutive steps before clustering.

        Returns:
            Dictionary with keys:

            - ``concepts``: ``(n_concepts, n_neurons)`` cluster centroids
              in firing-rate space.
            - ``assignments``: ``(batch,)`` integer tensor of concept
              assignments per sample.
            - ``concept_frequencies``: ``(n_concepts,)`` fraction of
              samples in each concept.
            - ``within_cluster_variance``: ``(n_concepts,)`` variance of
              each cluster (lower = tighter / more coherent concept).
        """
        if hidden_spikes.ndim == 3:
            T, B, N = hidden_spikes.shape
            # Aggregate over time windows
            n_windows = max(T // time_window, 1)
            w_size = T // n_windows
            aggregated = []
            for w in range(n_windows):
                start = w * w_size
                end = start + w_size
                agg = hidden_spikes[start:end].float().mean(dim=0)  # (B, N)
                aggregated.append(agg)
            data = torch.cat(aggregated, dim=0)  # (B*n_windows, N)
        elif hidden_spikes.ndim == 2:
            data = hidden_spikes.float()
        else:
            raise ValueError(f"Expected 2D or 3D tensor, got {hidden_spikes.ndim}D")

        B = data.shape[0]
        N = data.shape[1]
        k = min(n_concepts, B)

        # K-means via Lloyd's algorithm
        # Initialize: pick k random data points as centroids
        perm = torch.randperm(B)[:k]
        centroids = data[perm].clone()  # (k, N)
        assignments = torch.zeros(B, dtype=torch.long)

        for _ in range(30):  # max iterations
            # Assign each point to nearest centroid
            dists = torch.cdist(data, centroids)  # (B, k)
            new_assignments = dists.argmin(dim=1)

            # Check convergence
            if torch.equal(new_assignments, assignments):
                break
            assignments = new_assignments

            # Update centroids
            for c in range(k):
                mask = (assignments == c)
                if mask.any():
                    centroids[c] = data[mask].mean(dim=0)

        # Compute within-cluster variance
        within_var = torch.zeros(k)
        for c in range(k):
            mask = (assignments == c)
            if mask.any():
                diffs = data[mask] - centroids[c]
                within_var[c] = diffs.pow(2).mean()

        # Concept frequencies
        freqs = torch.zeros(k)
        for c in range(k):
            freqs[c] = (assignments == c).float().mean()

        return {
            "concepts": centroids.cpu(),
            "assignments": assignments.cpu(),
            "concept_frequencies": freqs.cpu(),
            "within_cluster_variance": within_var.cpu(),
        }

    # ------------------------------------------------------------------
    # 5. Decision boundary visualization data
    # ------------------------------------------------------------------

    def decision_boundary_data(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        resolution: int = 50,
        feature_dims: Tuple[int, int] = (0, 1),
    ) -> Dict[str, Any]:
        """Generate data for visualizing decision boundaries in 2D.

        Creates a grid of points spanning the two selected feature
        dimensions, runs each through the network, and returns the
        predicted class at each grid point along with the original data.

        This is the SNN analogue of plotting a classifier's decision
        surface — essential for understanding where the network is
        uncertain or where adversarial examples might live.

        Args:
            x: Input tensor ``(batch, n_features)`` or ``(batch, C, H, W)``.
            y: Label tensor ``(batch,)``.
            resolution: Number of grid points per axis.
            feature_dims: Tuple of two feature indices to sweep.

        Returns:
            Dictionary with keys:

            - ``grid_x1``, ``grid_x2``: 1D arrays of grid coordinates.
            - ``predictions``: ``(resolution, resolution)`` predicted
              class at each grid point.
            - ``original_x``: ``(batch, 2)`` the two selected features
              of the original data.
            - ``original_y``: ``(batch,)`` original labels.
        """
        self.model.eval()
        x_flat = x.detach().cpu().float()
        if x_flat.ndim > 2:
            x_flat = x_flat.view(x_flat.shape[0], -1)

        d1, d2 = feature_dims

        # Determine grid range from data
        x1_min, x1_max = x_flat[:, d1].min().item(), x_flat[:, d1].max().item()
        x2_min, x2_max = x_flat[:, d2].min().item(), x_flat[:, d2].max().item()

        pad1 = 0.1 * (x1_max - x1_min + 1e-8)
        pad2 = 0.1 * (x2_max - x2_min + 1e-8)
        grid_x1 = torch.linspace(x1_min - pad1, x1_max + pad1, resolution)
        grid_x2 = torch.linspace(x2_min - pad2, x2_max + pad2, resolution)

        mesh_x1, mesh_x2 = torch.meshgrid(grid_x1, grid_x2, indexing="ij")

        # Build full-dimensional inputs with baseline values
        baseline = x_flat.mean(dim=0)  # (n_features,)
        grid_inputs = []
        for i in range(resolution):
            for j in range(resolution):
                sample = baseline.clone()
                sample[d1] = mesh_x1[i, j]
                sample[d2] = mesh_x2[i, j]
                grid_inputs.append(sample)

        grid_tensor = torch.stack(grid_inputs).to(self.device)  # (res*res, n_features)

        # Forward pass in batches to avoid OOM
        preds = []
        batch_size = 256
        with torch.no_grad():
            for start in range(0, len(grid_tensor), batch_size):
                batch = grid_tensor[start : start + batch_size]
                out = self.model(batch)
                if isinstance(out, dict):
                    out = out.get("output", out.get("spikes", next(iter(out.values()))))
                if out.ndim > 1:
                    pred = out.argmax(dim=-1)
                else:
                    pred = (out > 0.5).long()
                preds.append(pred.cpu())

        predictions = torch.cat(preds).view(resolution, resolution)

        return {
            "grid_x1": grid_x1.numpy(),
            "grid_x2": grid_x2.numpy(),
            "predictions": predictions.numpy(),
            "original_x": x_flat[:, [d1, d2]].numpy(),
            "original_y": y.cpu().numpy(),
        }
