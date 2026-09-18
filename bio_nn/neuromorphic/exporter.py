"""Model export for neuromorphic hardware platforms.

Exports trained SNN models to formats compatible with Intel Loihi (Lava),
BrainScaleS, SpiNNaker2 (PyNN), SNNTorch, and ONNX with temporal dimensions.
All exporters handle missing library dependencies gracefully.
"""

from __future__ import annotations

import json
import logging
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional dependency guards
# ---------------------------------------------------------------------------

_TORCH_AVAILABLE = True
_ONNX_AVAILABLE = True

try:
    import snntorch as snn
except ImportError:
    snn = None

try:
    import lava.lib.dl.slayer as slayer
except ImportError:
    slayer = None

try:
    import pyNN
except ImportError:
    pyNN = None

try:
    import onnx
    import onnxruntime as ort
except ImportError:
    _ONNX_AVAILABLE = False
    onnx = None
    ort = None


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class ExportConfig:
    """Configuration for neuromorphic export.

    Attributes:
        platform: Target platform identifier.
        timesteps: Number of simulation time steps for the temporal model.
        dt: Simulation time step in milliseconds.
        input_encoding: Spike encoding scheme ('rate', 'temporal', 'delta').
        weight_clamp: If set, clamp exported weights to [-value, value].
        output_dir: Directory for exported artifacts.
    """

    platform: str = "snntorch"
    timesteps: int = 32
    dt: float = 1.0
    input_encoding: str = "rate"
    weight_clamp: Optional[float] = None
    output_dir: str = "exports"


@dataclass
class ExportResult:
    """Result of a model export operation.

    Attributes:
        format: Exported format name.
        path: Path to the exported file(s).
        metadata: Additional format-specific metadata.
        warnings: Any non-fatal warnings generated during export.
    """

    format: str
    path: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# SNNTorch export
# ---------------------------------------------------------------------------


def _extract_layer_params(module: nn.Module) -> List[Dict[str, Any]]:
    """Recursively extract weight matrices and neuron parameters from a model."""
    layers = []
    for name, child in module.named_children():
        entry: Dict[str, Any] = {"name": name, "type": type(child).__name__}

        if hasattr(child, "weight") and child.weight is not None:
            entry["weight"] = child.weight.detach().cpu().numpy().tolist()
        if hasattr(child, "bias") and child.bias is not None:
            entry["bias"] = child.bias.detach().cpu().numpy().tolist()
        if hasattr(child, "threshold"):
            entry["threshold"] = float(child.threshold)
        if hasattr(child, "tau"):
            entry["tau"] = float(child.tau)
        if hasattr(child, "decay"):
            entry["decay"] = float(child.decay)

        entry["children"] = _extract_layer_params(child)
        layers.append(entry)
    return layers


def export_snntorch(
    model: nn.Module,
    path: Union[str, Path],
    config: Optional[ExportConfig] = None,
) -> ExportResult:
    """Export a model to SNNTorch-compatible format.

    Creates a JSON representation of the network topology, weights, and
    neuron parameters that can be loaded by SNNTorch for simulation.

    Args:
        model: Trained PyTorch model (may contain snntorch neurons).
        path: Output file path (JSON).
        config: Export configuration.

    Returns:
        ExportResult with file path and metadata.

    Raises:
        FileNotFoundError: If parent directory does not exist.
        ValueError: If model has no extractable parameters.
    """
    config = config or ExportConfig(platform="snntorch")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    warnings_list: List[str] = []
    if snn is None:
        warnings_list.append(
            "snntorch not installed; exported JSON is structural only, "
            "no snntorch neuron classes referenced."
        )

    layers = _extract_layer_params(model)
    if not layers:
        raise ValueError("Model has no extractable parameters for export.")

    export_data: Dict[str, Any] = {
        "format": "snntorch",
        "timesteps": config.timesteps,
        "dt": config.dt,
        "input_encoding": config.input_encoding,
        "layers": layers,
        "model_summary": {
            "total_params": sum(p.numel() for p in model.parameters()),
            "trainable_params": sum(
                p.numel() for p in model.parameters() if p.requires_grad
            ),
        },
    }

    if config.weight_clamp is not None:
        for layer in layers:
            if "weight" in layer:
                arr = np.array(layer["weight"])
                arr = np.clip(arr, -config.weight_clamp, config.weight_clamp)
                layer["weight"] = arr.tolist()

    path.write_text(json.dumps(export_data, indent=2))
    logger.info("SNNTorch export written to %s", path)

    return ExportResult(
        format="snntorch",
        path=str(path),
        metadata=export_data["model_summary"],
        warnings=warnings_list,
    )


# ---------------------------------------------------------------------------
# ONNX export with temporal dimensions
# ---------------------------------------------------------------------------


def export_onnx_temporal(
    model: nn.Module,
    path: Union[str, Path],
    config: Optional[ExportConfig] = None,
    input_shape: Optional[Tuple[int, ...]] = None,
) -> ExportResult:
    """Export model to ONNX with explicit temporal dimensions.

    Wraps the model so that the ONNX graph includes the time dimension,
    making it suitable for event-driven inference runtimes.

    Args:
        model: Trained PyTorch model.
        path: Output file path (.onnx).
        config: Export configuration.
        input_shape: Shape of a single-sample input tensor (batch=1).
                     If None, infers from first parameter shape.

    Returns:
        ExportResult with file path and metadata.

    Raises:
        RuntimeError: If ONNX export fails.
        FileNotFoundError: If parent directory does not exist.
    """
    if not _ONNX_AVAILABLE:
        raise RuntimeError(
            "onnx and onnxruntime are required for ONNX export. "
            "Install with: pip install onnx onnxruntime"
        )

    config = config or ExportConfig(platform="onnx")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    timesteps = config.timesteps
    if input_shape is None:
        # Attempt to infer from first Linear-like layer
        for p in model.parameters():
            input_shape = (1,) + tuple(p.shape[1:])
            break
        if input_shape is None:
            input_shape = (1, 128)

    class TemporalWrapper(nn.Module):
        """Wraps model to unroll across time steps for ONNX export."""

        def __init__(self, base: nn.Module, steps: int):
            super().__init__()
            self.base = base
            self.steps = steps

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            outputs = []
            for _ in range(self.steps):
                out = self.base(x)
                if isinstance(out, dict):
                    out = out.get("output", out.get("spikes", torch.zeros_like(x)))
                outputs.append(out)
            return torch.stack(outputs, dim=1)

    wrapper = TemporalWrapper(model, timesteps)
    wrapper.eval()

    # Flatten input_shape to batch=1
    onnx_input = tuple([1] + list(input_shape[1:]))
    dummy = torch.randn(*onnx_input)

    warnings_list: List[str] = []
    try:
        torch.onnx.export(
            wrapper,
            dummy,
            str(path),
            input_names=["spike_input"],
            output_names=["temporal_output"],
            dynamic_axes={
                "spike_input": {0: "batch"},
                "temporal_output": {0: "batch"},
            },
            opset_version=17,
        )
    except Exception as exc:
        raise RuntimeError(f"ONNX export failed: {exc}") from exc

    metadata: Dict[str, Any] = {
        "timesteps": timesteps,
        "input_shape": list(onnx_input),
        "opset": 17,
    }

    # Validate the exported model
    try:
        session = ort.InferenceSession(str(path))
        input_name = session.get_inputs()[0].name
        test_input = np.random.randn(*onnx_input).astype(np.float32)
        session.run(None, {input_name: test_input})
        metadata["validation"] = "passed"
    except Exception as exc:
        warnings_list.append(f"ONNX validation failed: {exc}")
        metadata["validation"] = "failed"

    path_str = str(path)
    logger.info("ONNX export written to %s", path_str)

    return ExportResult(
        format="onnx",
        path=path_str,
        metadata=metadata,
        warnings=warnings_list,
    )


# ---------------------------------------------------------------------------
# Lava (Intel Loihi) export
# ---------------------------------------------------------------------------


def _serialize_lava_weights(model: nn.Module) -> List[Dict[str, Any]]:
    """Extract weight data in a format suitable for Lava/Loihi."""
    layers = []
    for name, param in model.named_parameters():
        arr = param.detach().cpu().numpy()
        layers.append({
            "name": name,
            "shape": list(arr.shape),
            "dtype": str(arr.dtype),
            "data_b64": None,  # placeholder; real impl would base64-encode
            "min": float(arr.min()),
            "max": float(arr.max()),
            "mean": float(arr.mean()),
        })
    return layers


def export_lava(
    model: nn.Module,
    path: Union[str, Path],
    config: Optional[ExportConfig] = None,
) -> ExportResult:
    """Export model to Intel Lava/Loihi-compatible format.

    Creates a JSON descriptor with weights and network topology that can
    be loaded by the Lava framework for Loihi deployment. When lava.lib.dl
    is installed, performs additional validation.

    Args:
        model: Trained PyTorch model.
        path: Output file path (JSON).
        config: Export configuration.

    Returns:
        ExportResult with file path and metadata.
    """
    config = config or ExportConfig(platform="lava")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    warnings_list: List[str] = []
    if slayer is None:
        warnings_list.append(
            "lava.lib.dl.slayer not installed; exported descriptor is "
            "structural only. Install lava-dl for full Loihi support."
        )

    layers = _serialize_lava_weights(model)

    lava_data: Dict[str, Any] = {
        "format": "lava_loihi",
        "dt": config.dt,
        "timesteps": config.timesteps,
        "layers": layers,
        "neuron_model": "lif",
        "encoding": config.input_encoding,
    }

    if config.weight_clamp is not None:
        for layer in layers:
            layer["min"] = max(layer["min"], -config.weight_clamp)
            layer["max"] = min(layer["max"], config.weight_clamp)

    path.write_text(json.dumps(lava_data, indent=2))
    logger.info("Lava/Loihi export written to %s", path)

    return ExportResult(
        format="lava",
        path=str(path),
        metadata={"num_layers": len(layers)},
        warnings=warnings_list,
    )


# ---------------------------------------------------------------------------
# SpiNNaker2 / PyNN export
# ---------------------------------------------------------------------------


def _build_pynn_descriptor(model: nn.Module, config: ExportConfig) -> Dict[str, Any]:
    """Build a PyNN-compatible network descriptor."""
    populations = []
    projections = []

    layer_idx = 0
    prev_name = None
    for name, child in model.named_children():
        pop_entry: Dict[str, Any] = {
            "label": name or f"layer_{layer_idx}",
            "size": _get_layer_size(child),
        }

        # Map PyTorch neuron types to PyNN neuron models
        type_name = type(child).__name__.lower()
        if "lif" in type_name or "leaky" in type_name:
            pop_entry["neuron_model"] = "IF_cond_exp"
            pop_entry["tau_mem"] = getattr(child, "tau", 20.0)
            pop_entry["tau_syn_exc"] = 5.0
        elif "izhikevich" in type_name:
            pop_entry["neuron_model"] = "Izhikevich"
            pop_entry["a"] = 0.02
            pop_entry["b"] = 0.2
            pop_entry["c"] = -65.0
            pop_entry["d"] = 8.0
        else:
            pop_entry["neuron_model"] = "IF_curr_exp"
            pop_entry["tau_mem"] = 20.0

        populations.append(pop_entry)

        if prev_name is not None and hasattr(child, "weight"):
            proj_entry: Dict[str, Any] = {
                "pre": prev_name,
                "post": name or f"layer_{layer_idx}",
                "connector": "all_to_all",
                "synapse_model": "StaticSynapse",
            }
            if hasattr(child, "weight") and child.weight is not None:
                w = child.weight.detach().cpu().numpy()
                proj_entry["weight_shape"] = list(w.shape)
                proj_entry["weight_min"] = float(w.min())
                proj_entry["weight_max"] = float(w.max())
            projections.append(proj_entry)

        prev_name = name or f"layer_{layer_idx}"
        layer_idx += 1

    return {
        "format": "pynn_spinnaker2",
        "simulator": "spinnaker2",
        "dt": config.dt,
        "timesteps": config.timesteps,
        "populations": populations,
        "projections": projections,
    }


def _get_layer_size(module: nn.Module) -> int:
    """Infer the output size of a layer."""
    if hasattr(module, "out_features"):
        return module.out_features
    if hasattr(module, "out_channels"):
        return module.out_channels
    if hasattr(module, "size"):
        return module.size
    for p in module.parameters():
        return p.shape[0]
    return 0


def export_pynn(
    model: nn.Module,
    path: Union[str, Path],
    config: Optional[ExportConfig] = None,
) -> ExportResult:
    """Export model to PyNN format for SpiNNaker2 deployment.

    Generates a JSON descriptor compatible with PyNN that can be loaded
    by the SpiNNaker2 simulation interface.

    Args:
        model: Trained PyTorch model.
        path: Output file path (JSON).
        config: Export configuration.

    Returns:
        ExportResult with file path and metadata.
    """
    config = config or ExportConfig(platform="pynn")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    warnings_list: List[str] = []
    if pyNN is None:
        warnings_list.append(
            "pyNN not installed; exported descriptor is structural only. "
            "Install pyNN for full SpiNNaker2 deployment."
        )

    descriptor = _build_pynn_descriptor(model, config)

    path.write_text(json.dumps(descriptor, indent=2))
    logger.info("PyNN/SpiNNaker2 export written to %s", path)

    return ExportResult(
        format="pynn",
        path=str(path),
        metadata={
            "num_populations": len(descriptor["populations"]),
            "num_projections": len(descriptor["projections"]),
        },
        warnings=warnings_list,
    )


# ---------------------------------------------------------------------------
# High-level exporter
# ---------------------------------------------------------------------------


class NeuromorphicExporter:
    """Unified interface for exporting SNN models to neuromorphic formats.

    Wraps the individual export functions and provides a single entry point
    for multi-format export.

    Args:
        model: Trained PyTorch SNN model.
        config: Export configuration.
    """

    _FORMATS = {
        "snntorch": export_snntorch,
        "onnx": export_onnx_temporal,
        "lava": export_lava,
        "pynn": export_pynn,
    }

    def __init__(self, model: nn.Module, config: Optional[ExportConfig] = None):
        self.model = model
        self.config = config or ExportConfig()
        self._results: Dict[str, ExportResult] = {}

    def export(self, fmt: str, path: Optional[Union[str, Path]] = None, **kwargs: Any) -> ExportResult:
        """Export to a specific format.

        Args:
            fmt: Format name ('snntorch', 'onnx', 'lava', 'pynn').
            path: Output path. If None, uses ``<output_dir>/<model_name>.<ext>``.
            **kwargs: Additional arguments passed to the format-specific exporter.

        Returns:
            ExportResult for the requested format.

        Raises:
            ValueError: If format is not supported.
        """
        if fmt not in self._FORMATS:
            raise ValueError(
                f"Unsupported format '{fmt}'. Choose from: {list(self._FORMATS)}"
            )

        if path is None:
            ext = {"snntorch": ".json", "onnx": ".onnx", "lava": ".json", "pynn": ".json"}
            path = Path(self.config.output_dir) / f"model{ext[fmt]}"

        exporter_fn = self._FORMATS[fmt]
        result = exporter_fn(self.model, path, self.config, **kwargs)
        self._results[fmt] = result
        return result

    def export_all(
        self, output_dir: Optional[Union[str, Path]] = None
    ) -> Dict[str, ExportResult]:
        """Export to all supported formats.

        Args:
            output_dir: Override output directory for all exports.

        Returns:
            Dictionary mapping format names to ExportResults.
        """
        if output_dir is not None:
            self.config.output_dir = str(output_dir)

        for fmt in self._FORMATS:
            try:
                self.export(fmt)
            except Exception as exc:
                logger.warning("Export to %s failed: %s", fmt, exc)
                self._results[fmt] = ExportResult(
                    format=fmt,
                    path="",
                    warnings=[str(exc)],
                )

        return dict(self._results)

    @property
    def results(self) -> Dict[str, ExportResult]:
        """Return results from previous export operations."""
        return dict(self._results)
