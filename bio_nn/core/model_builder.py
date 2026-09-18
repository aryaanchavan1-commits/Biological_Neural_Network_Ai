"""Model builder — assembles a complete SNN / Bio-NN from configuration.

The ``build_model`` factory reads a config dict, resolves every
sub-component (encoder, layers, decoder, plasticity, topology), and
returns a fully wired ``BioNNModel`` ready for training.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from .registry import ComponentRegistry
from ..topology import build_topology
from ..neurons import create_neuron
from ..synapses import create_synapse
from ..encoders import create_encoder
from ..decoders import create_decoder

logger = logging.getLogger(__name__)


def build_model(config: Dict[str, Any]) -> nn.Module:
    """Build a complete SNN model from configuration.

    Config structure::

        model.type: "snn" | "bio_nn" | "ann"
        model.layers: [{type, size}]
        encoder: {type, ...}
        decoder: {type, ...}
        neuron: {model, tau_mem, threshold, ...}
        synapse: {type, weight_init, ...}
        plasticity: {enabled, rules: [...]}
        structural: {enabled, growth, pruning}
        topology: {type, sparsity}

    Args:
        config: Full experiment configuration.

    Returns:
        An ``nn.Module`` (always a ``BioNNModel``).
    """
    model_type = config.get("model", {}).get("type", "bio_nn")
    logger.info("Building model of type '%s'", model_type)
    return BioNNModel(config)


# ======================================================================
# BioNNModel
# ======================================================================


class BioNNModel(nn.Module):
    """Complete BIO-NN model combining all biologically-inspired components.

    Orchestrates encoding, layered spiking dynamics, plasticity,
    structural plasticity, topology masking, and decoding through a
    temporal simulation loop.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__()
        self.config = config

        model_cfg = config.get("model", {})
        neuron_cfg = config.get("neuron", {})
        synapse_cfg = config.get("synapse", {})
        encoder_cfg = config.get("encoder", {})
        decoder_cfg = config.get("decoder", {})
        plasticity_cfg = config.get("plasticity", {})
        structural_cfg = config.get("structural", {})
        topology_cfg = config.get("topology", {})
        self.time_steps: int = model_cfg.get("time_steps", 15)

        layer_specs = model_cfg.get("layers", [])
        if not layer_specs:
            layer_specs = [
                {"type": "lif_neuron", "size": 128},
                {"type": "lif_neuron", "size": 64},
            ]

        # --- encoder ---
        self.encoder = self._build_encoder(encoder_cfg)

        # --- layers (neurons + synapses) ---
        self.neuron_layers: nn.ModuleList = nn.ModuleList()
        self.synapse_layers: nn.ModuleList = nn.ModuleList()
        self.topology_masks: List[Optional[torch.Tensor]] = []
        self._layer_sizes: List[int] = []

        prev_size = model_cfg.get("input_size", layer_specs[0].get("size", 128) if layer_specs else 128)
        for spec in layer_specs:
            size = spec.get("size", 128)
            self._layer_sizes.append(size)

            neuron_type = neuron_cfg.get("model", "lif")
            neuron = create_neuron(neuron_type, size, config=neuron_cfg)
            self.neuron_layers.append(neuron)

            synapse_type = synapse_cfg.get("type", "static")
            synapse = create_synapse(synapse_type, prev_size, size, config=synapse_cfg)
            self.synapse_layers.append(synapse)

            if topology_cfg.get("type"):
                topo = build_topology(topology_cfg)
                mask = topo.generate(prev_size, size)
                self.topology_masks.append(mask)
            else:
                self.topology_masks.append(None)

            prev_size = size

        # --- decoder ---
        self.decoder = self._build_decoder(decoder_cfg)

        # --- plasticity ---
        self.plasticity_enabled: bool = plasticity_cfg.get("enabled", False)
        self.plasticity_rules: nn.ModuleList = nn.ModuleList()
        if self.plasticity_enabled:
            for rule_cfg in plasticity_cfg.get("rules", []):
                rule = ComponentRegistry.build(rule_cfg)
                self.plasticity_rules.append(rule)

        # --- structural plasticity ---
        self.structural_enabled: bool = structural_cfg.get("enabled", False)
        self.structural_plasticity = None
        if self.structural_enabled:
            self.structural_plasticity = ComponentRegistry.build(structural_cfg)

        # --- output layer (linear readout) ---
        last_size = self._layer_sizes[-1] if self._layer_sizes else 128
        n_classes = model_cfg.get("n_classes", 10)
        self.output_layer = nn.Linear(last_size, n_classes)

        # --- state ---
        self._spike_history: List[List[torch.Tensor]] = []
        self._membrane_history: List[List[torch.Tensor]] = []

    # ------------------------------------------------------------------
    # Component builders
    # ------------------------------------------------------------------

    def _build_encoder(self, cfg: dict) -> nn.Module:
        try:
            encoder_type = cfg.get("type", "rate")
            return create_encoder(encoder_type, cfg)
        except (KeyError, ValueError):
            from ..encoders.rate import RateEncoder
            return RateEncoder(cfg)

    def _build_decoder(self, cfg: dict) -> nn.Module:
        try:
            decoder_type = cfg.get("type", "rate")
            return create_decoder(decoder_type, cfg)
        except (KeyError, ValueError):
            from ..decoders.rate_decoder import RateDecoder
            return RateDecoder(cfg)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        data: torch.Tensor,
        return_details: bool = False,
    ) -> Any:
        """Full forward pass through time.

        Args:
            data: Input tensor ``(batch, features)``.
            return_details: If True, also return a dict of intermediate states.

        Returns:
            If ``return_details`` is False: ``(output,)`` where output is
            ``(batch, n_classes)``.
            If ``return_details`` is True: ``(output, details)``.
        """
        batch_size = data.shape[0]
        device = data.device

        # Encode input → (time_steps, batch, input_size)
        encoded = self.encoder(data, self.time_steps)

        # Initialise neuron states
        states: List[Dict[str, torch.Tensor]] = []
        for neuron in self.neuron_layers:
            states.append(neuron._get_initial_state(batch_size, device))

        self._spike_history = [[] for _ in self.neuron_layers]
        self._membrane_history = [[] for _ in self.neuron_layers]

        output_spikes: Optional[torch.Tensor] = None

        for t in range(self.time_steps):
            current_input = encoded[t]  # (batch, input_size)

            for layer_idx in range(len(self.neuron_layers)):
                synapse = self.synapse_layers[layer_idx]

                # Apply topology mask to synaptic input
                post_input = synapse(current_input)

                if self.topology_masks[layer_idx] is not None:
                    mask = self.topology_masks[layer_idx].to(device)
                    # Mask the weight matrix of this synapse
                    with torch.no_grad():
                        synapse.weights.mul_(mask.t())

                # Neuron dynamics
                spikes, membrane, new_state = self.neuron_layers[layer_idx](
                    post_input, states[layer_idx]
                )
                states[layer_idx] = new_state

                self._spike_history[layer_idx].append(spikes.detach())
                self._membrane_history[layer_idx].append(membrane.detach())

                # Plasticity
                if self.plasticity_enabled and layer_idx > 0:
                    pre_spikes_detached = self._spike_history[layer_idx - 1][-1]
                    for rule in self.plasticity_rules:
                        with torch.no_grad():
                            new_w = rule(
                                synapse.weights.detach(),
                                pre_spikes_detached,
                                spikes.detach(),
                            )
                            synapse.weights.copy_(new_w)

                # Structural plasticity
                if self.structural_enabled and self.structural_plasticity is not None:
                    with torch.no_grad():
                        activity = spikes.mean(dim=0)
                        new_w = self.structural_plasticity(
                            synapse.weights.detach(), activity
                        )
                        synapse.weights.copy_(new_w)

                current_input = spikes

            output_spikes = spikes  # last layer spikes

        # Decode output
        if output_spikes is None:
            output = torch.zeros(batch_size, self.output_layer.out_features, device=device)
        else:
            decoded = self.decoder(output_spikes.unsqueeze(0)).squeeze(0)
            output = self.output_layer(decoded)

        if return_details:
            details = {
                "spike_history": self._spike_history,
                "membrane_history": self._membrane_history,
                "weights": [s.get_weights() for s in self.synapse_layers],
                "connectivity": [m for m in self.topology_masks],
                "states": states,
            }
            return output, details

        return (output,)

    # ------------------------------------------------------------------
    # Introspection helpers
    # ------------------------------------------------------------------

    def get_neuron_state(self) -> List[Dict[str, torch.Tensor]]:
        """Return the latest state dict for each neuron layer."""
        return [n.get_state() for n in self.neuron_layers]

    def get_membrane_potential(self) -> List[torch.Tensor]:
        """Return the latest membrane potential for each neuron layer."""
        result = []
        for n in self.neuron_layers:
            state = n.get_state()
            result.append(state.get("membrane", torch.empty(0)))
        return result

    def get_spike_history(self) -> List[List[torch.Tensor]]:
        """Return the full spike history for every neuron layer."""
        return self._spike_history

    def get_synaptic_weights(self) -> List[torch.Tensor]:
        """Return a detached weight matrix for each synapse layer."""
        return [s.get_weights() for s in self.synapse_layers]

    def get_connectivity(self) -> List[Optional[torch.Tensor]]:
        """Return the topology mask for each synapse layer."""
        return self.topology_masks

    def get_plasticity_state(self) -> Dict[str, Any]:
        """Return plasticity rule state summary."""
        return {
            "enabled": self.plasticity_enabled,
            "n_rules": len(self.plasticity_rules),
            "rule_names": [type(r).__name__ for r in self.plasticity_rules],
        }

    def get_network_statistics(self) -> Dict[str, Any]:
        """Aggregate statistics across all layers."""
        total_neurons = sum(self._layer_sizes)
        total_weights = sum(
            s.weights.numel() for s in self.synapse_layers
        )
        active_weights = 0
        for s in self.synapse_layers:
            active_weights += (s.weights.detach() != 0).sum().item()

        firing_rates = []
        for history in self._spike_history:
            if history:
                all_spikes = torch.stack(history)
                firing_rates.append(all_spikes.mean().item())
            else:
                firing_rates.append(0.0)

        return {
            "total_neurons": total_neurons,
            "total_weights": total_weights,
            "active_weights": active_weights,
            "connectivity_ratio": active_weights / max(total_weights, 1),
            "firing_rates": firing_rates,
            "n_layers": len(self.neuron_layers),
            "time_steps": self.time_steps,
        }

    def reset_state(self) -> None:
        """Clear all recorded history and neuron state."""
        for neuron in self.neuron_layers:
            neuron.reset_state()
        for synapse in self.synapse_layers:
            if hasattr(synapse, "reset_traces"):
                synapse.reset_traces()
        self._spike_history.clear()
        self._membrane_history.clear()
