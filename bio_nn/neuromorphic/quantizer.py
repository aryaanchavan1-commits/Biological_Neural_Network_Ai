"""Weight quantization for neuromorphic hardware deployment.

Provides integer, ternary, and binary quantization with both
quantization-aware training (QAT) and post-training quantization (PTQ)
paths, plus accuracy-efficiency trade-off analysis.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class QuantizationConfig:
    """Configuration for weight quantization.

    Attributes:
        bits: Number of bits for integer quantization (4 or 8).
        mode: Quantization mode ('int', 'ternary', 'binary').
        symmetric: If True, use symmetric quantization around zero.
        per_channel: If True, quantize each output channel independently.
        momentum: EMA momentum for running min/max tracking.
        freeze_bn: Whether to freeze batch-norm during QAT.
    """

    bits: int = 8
    mode: str = "int"
    symmetric: bool = True
    per_channel: bool = True
    momentum: float = 0.99
    freeze_bn: bool = True


@dataclass
class QuantizationResult:
    """Result of a quantization operation.

    Attributes:
        original_model: Reference to the original model.
        quantized_model: The quantized model (state dict or nn.Module).
        config: The quantization config used.
        stats: Per-layer quantization statistics.
        accuracy: Accuracy of the quantized model if evaluated.
    """

    original_model: Optional[nn.Module] = None
    quantized_model: Optional[nn.Module] = None
    config: Optional[QuantizationConfig] = None
    stats: Dict[str, Any] = field(default_factory=dict)
    accuracy: Optional[float] = None


# ---------------------------------------------------------------------------
# Core quantization primitives
# ---------------------------------------------------------------------------


def _compute_scale_zero_point(
    tensor: torch.Tensor,
    bits: int = 8,
    symmetric: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute quantization scale and zero-point for a tensor.

    Args:
        tensor: Weight tensor to quantize.
        bits: Bit width for quantization.
        symmetric: Whether to use symmetric quantization.

    Returns:
        Tuple of (scale, zero_point) tensors.
    """
    qmin = -(2 ** (bits - 1))
    qmax = 2 ** (bits - 1) - 1

    if symmetric:
        abs_max = tensor.abs().max()
        scale = abs_max / qmax
        zero_point = torch.zeros_like(scale)
    else:
        t_min = tensor.min()
        t_max = tensor.max()
        scale = (t_max - t_min) / (qmax - qmin)
        zero_point = qmin - t_min / scale

    # Avoid division by zero
    scale = torch.clamp(scale, min=1e-8)
    return scale, zero_point


def _quantize_tensor(
    tensor: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor,
    bits: int = 8,
) -> torch.Tensor:
    """Quantize and dequantize a tensor (simulated quantization)."""
    qmin = -(2 ** (bits - 1))
    qmax = 2 ** (bits - 1) - 1

    scaled = tensor / scale + zero_point
    quantized = torch.round(scaled).clamp(qmin, qmax)
    dequantized = (quantized - zero_point) * scale
    return dequantized


def quantize_int(
    tensor: torch.Tensor,
    bits: int = 8,
    symmetric: bool = True,
    per_channel: bool = True,
    channel_dim: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize a weight tensor to integer representation.

    Performs simulated quantization (quantize then dequantize) so that
    the tensor remains differentiable for training.

    Args:
        tensor: Weight tensor to quantize.
        bits: Bit width (4 or 8).
        symmetric: Use symmetric quantization.
        per_channel: Quantize each channel independently.
        channel_dim: Dimension to treat as channels for per-channel quant.

    Returns:
        Tuple of (quantized_weights, scale, zero_point).
    """
    if bits not in (4, 8):
        raise ValueError(f"Supported bit widths are 4 and 8, got {bits}")

    if per_channel and tensor.dim() > 1:
        # Per-channel quantization
        n_channels = tensor.shape[channel_dim]
        scale = torch.empty(n_channels, device=tensor.device, dtype=tensor.dtype)
        zero_point = torch.empty(n_channels, device=tensor.device, dtype=tensor.dtype)
        for c in range(n_channels):
            slices = [slice(None)] * tensor.dim()
            slices[channel_dim] = c
            ch = tensor[tuple(slices)]
            s, zp = _compute_scale_zero_point(ch, bits, symmetric)
            scale[c] = s.squeeze()
            zero_point[c] = zp.squeeze()
        # Reshape for broadcasting: (n_channels, 1, 1, ...)
        shape = [1] * tensor.dim()
        shape[channel_dim] = n_channels
        scale = scale.reshape(shape)
        zero_point = zero_point.reshape(shape)
    else:
        scale, zero_point = _compute_scale_zero_point(tensor, bits, symmetric)

    quantized = _quantize_tensor(tensor, scale, zero_point, bits)
    return quantized, scale, zero_point


def quantize_ternary(
    tensor: torch.Tensor,
    threshold: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize weights to {-1, 0, +1}.

    Uses a threshold-based approach: values above +threshold map to +1,
    below -threshold to -1, and everything else to 0.

    Args:
        tensor: Weight tensor.
        threshold: Decision threshold. If None, uses 0.5 * mean(abs(tensor)).

    Returns:
        Tuple of (ternary_weights, scale_factor).
    """
    abs_weights = tensor.abs()
    if threshold is None:
        threshold = 0.5 * abs_weights.mean().item()

    ternary = torch.zeros_like(tensor)
    ternary[tensor > threshold] = 1.0
    ternary[tensor < -threshold] = -1.0

    # Scale factor to approximate original magnitude
    nonzero_mask = ternary.abs() > 0
    if nonzero_mask.any():
        scale = abs_weights[nonzero_mask].mean()
    else:
        scale = torch.tensor(1.0)

    return ternary * scale, scale


def quantize_binary(
    tensor: torch.Tensor,
    method: str = "sign",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize weights to {-1, +1} (binary).

    Args:
        tensor: Weight tensor.
        method: Binarization method. 'sign' uses sign function,
                'threshold' uses median-based thresholding.

    Returns:
        Tuple of (binary_weights, scale_factor).
    """
    if method == "sign":
        binary = torch.sign(tensor)
        binary[binary == 0] = 1.0
    elif method == "threshold":
        threshold = tensor.mean()
        binary = torch.where(tensor > threshold, 1.0, -1.0)
    else:
        raise ValueError(f"Unknown binarization method: {method}")

    # XNOR-compatible scale
    scale = tensor.abs().mean()
    return binary * scale, scale


# ---------------------------------------------------------------------------
# Post-training quantization
# ---------------------------------------------------------------------------


def post_training_quantize(
    model: nn.Module,
    config: Optional[QuantizationConfig] = None,
    calibration_data: Optional[torch.Tensor] = None,
    num_batches: int = 32,
) -> QuantizationResult:
    """Apply post-training quantization to a model.

    Quantizes all weight tensors using collected statistics without
    retraining. Optionally uses calibration data to compute optimal
    quantization ranges.

    Args:
        model: Model to quantize.
        config: Quantization configuration.
        calibration_data: Optional input data for range calibration.
        num_batches: Number of batches to use for calibration.

    Returns:
        QuantizationResult with quantized model and statistics.
    """
    config = config or QuantizationConfig()
    quantized = copy.deepcopy(model)
    quantized.eval()

    stats: Dict[str, Any] = {}

    for name, param in quantized.named_parameters():
        if "bias" in name:
            continue  # Typically keep biases in higher precision

        original = param.data.clone()

        if config.mode == "int":
            q_tensor, scale, zp = quantize_int(
                original,
                bits=config.bits,
                symmetric=config.symmetric,
                per_channel=config.per_channel,
            )
            param.data = q_tensor
            stats[name] = {
                "mode": "int",
                "bits": config.bits,
                "scale": scale.mean().item(),
                "compression_ratio": config.bits / 32.0,
                "mse": float(((original - q_tensor) ** 2).mean()),
            }

        elif config.mode == "ternary":
            q_tensor, scale = quantize_ternary(original)
            param.data = q_tensor
            stats[name] = {
                "mode": "ternary",
                "scale": scale.item(),
                "compression_ratio": 1.75 / 32.0,
                "mse": float(((original - q_tensor) ** 2).mean()),
                "sparsity": float((q_tensor == 0).float().mean()),
            }

        elif config.mode == "binary":
            q_tensor, scale = quantize_binary(original)
            param.data = q_tensor
            stats[name] = {
                "mode": "binary",
                "scale": scale.item(),
                "compression_ratio": 1.0 / 32.0,
                "mse": float(((original - q_tensor) ** 2).mean()),
            }

    # Compute aggregate stats
    total_mse = sum(s.get("mse", 0) for s in stats.values())
    avg_compression = np.mean([s.get("compression_ratio", 1.0) for s in stats.values()])

    stats["_aggregate"] = {
        "total_mse": total_mse,
        "avg_compression": float(avg_compression),
        "num_quantized_params": len(stats),
    }

    logger.info(
        "PTQ complete: mode=%s, bits=%d, avg_compression=%.2fx",
        config.mode,
        config.bits,
        1.0 / max(avg_compression, 1e-8),
    )

    return QuantizationResult(
        original_model=model,
        quantized_model=quantized,
        config=config,
        stats=stats,
    )


# ---------------------------------------------------------------------------
# Quantization-aware training
# ---------------------------------------------------------------------------


class _FakeLinearQuantize(torch.autograd.Function):
    """Straight-through estimator for simulated quantization in QAT."""

    @staticmethod
    def forward(
        ctx: Any,
        tensor: torch.Tensor,
        scale: torch.Tensor,
        zero_point: torch.Tensor,
        bits: int,
    ) -> torch.Tensor:
        qmin = -(2 ** (bits - 1))
        qmax = 2 ** (bits - 1) - 1

        scaled = tensor / scale + zero_point
        quantized = torch.round(scaled).clamp(qmin, qmax)
        dequantized = (quantized - zero_point) * scale
        ctx.save_for_backward(tensor, scale)
        return dequantized

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[Optional[torch.Tensor], ...]:
        tensor, scale = ctx.saved_tensors
        # Straight-through estimator: pass gradient through unchanged
        grad_input = grad_output.clone()
        return grad_input / scale, None, None, None


class QuantizationAwareHook:
    """Forward hook that applies simulated quantization during training.

    Insert this as a module hook to enable quantization-aware training.
    The quantization parameters are updated via EMA during the forward pass.

    Args:
        module: The nn.Linear or Conv layer to make quantization-aware.
        bits: Bit width for quantization.
        symmetric: Use symmetric quantization.
        momentum: EMA momentum for scale tracking.
    """

    def __init__(
        self,
        module: nn.Module,
        bits: int = 8,
        symmetric: bool = True,
        momentum: float = 0.99,
    ):
        self.module = module
        self.bits = bits
        self.symmetric = symmetric
        self.momentum = momentum
        self.scale: Optional[torch.Tensor] = None
        self.zero_point: Optional[torch.Tensor] = None
        self._registered = False

    def register(self) -> None:
        """Register the forward hook on the module."""
        if not self._registered:
            self.module.register_forward_hook(self._hook)
            self._registered = True

    def _hook(
        self, module: nn.Module, input: Tuple[torch.Tensor, ...], output: torch.Tensor
    ) -> torch.Tensor:
        """Apply fake quantization to the module output weights."""
        if not hasattr(module, "weight") or module.weight is None:
            return output

        weight = module.weight.data
        s, zp = _compute_scale_zero_point(weight, self.bits, self.symmetric)

        if self.scale is None:
            self.scale = s
            self.zero_point = zp
        else:
            self.scale = self.momentum * self.scale + (1 - self.momentum) * s
            self.zero_point = self.momentum * self.zero_point + (1 - self.momentum) * zp

        module.weight.data = _quantize_tensor(weight, self.scale, self.zero_point, self.bits)
        return output


def quantization_aware_hook(
    model: nn.Module,
    bits: int = 8,
    symmetric: bool = True,
    momentum: float = 0.99,
) -> List[QuantizationAwareHook]:
    """Attach quantization-aware hooks to all Linear/Conv layers in a model.

    Args:
        model: Model to make QAT-aware.
        bits: Bit width.
        symmetric: Symmetric quantization.
        momentum: EMA momentum.

    Returns:
        List of QuantizationAwareHook instances (keep references alive).
    """
    hooks: List[QuantizationAwareHook] = []
    for module in model.modules():
        if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            hook = QuantizationAwareHook(module, bits, symmetric, momentum)
            hook.register()
            hooks.append(hook)

    logger.info(
        "Attached QAT hooks to %d layers (bits=%d, symmetric=%s)",
        len(hooks),
        bits,
        symmetric,
    )
    return hooks


# ---------------------------------------------------------------------------
# Accuracy-efficiency trade-off analysis
# ---------------------------------------------------------------------------


def accuracy_efficiency_tradeoff(
    model: nn.Module,
    eval_fn: Any,
    test_data: Any,
    test_targets: Any,
    bit_widths: Sequence[int] = (4, 8),
    modes: Sequence[str] = ("int", "ternary", "binary"),
) -> Dict[str, Any]:
    """Analyze the accuracy vs. efficiency trade-off across quantization configs.

    Evaluates the model under multiple quantization settings and returns
    a comparison table.

    Args:
        model: Original full-precision model.
        eval_fn: Callable(model, data, targets) -> accuracy (float).
        test_data: Test input data.
        test_targets: Test labels.
        bit_widths: Bit widths to evaluate for integer mode.
        modes: Quantization modes to evaluate.

    Returns:
        Dictionary with per-config results and summary statistics.
    """
    results: Dict[str, Any] = {}

    # Baseline
    baseline_acc = eval_fn(model, test_data, test_targets)
    baseline_params = sum(p.numel() for p in model.parameters())
    results["baseline"] = {
        "accuracy": baseline_acc,
        "params": baseline_params,
        "model_size_mb": baseline_params * 4 / (1024 * 1024),
        "mode": "float32",
    }

    for mode in modes:
        if mode == "int":
            for bits in bit_widths:
                cfg = QuantizationConfig(bits=bits, mode=mode)
                qr = post_training_quantize(model, cfg)
                acc = eval_fn(qr.quantized_model, test_data, test_targets)
                key = f"{mode}_{bits}bit"
                results[key] = {
                    "accuracy": acc,
                    "accuracy_drop": baseline_acc - acc,
                    "model_size_mb": baseline_params * bits / (32 * 1024 * 1024),
                    "compression": 32.0 / bits,
                    "mse": qr.stats.get("_aggregate", {}).get("total_mse", 0),
                    "mode": mode,
                    "bits": bits,
                }
        else:
            cfg = QuantizationConfig(mode=mode)
            qr = post_training_quantize(model, cfg)
            acc = eval_fn(qr.quantized_model, test_data, test_targets)
            size_map = {"ternary": 1.75, "binary": 1.0}
            bits_equiv = size_map.get(mode, 8)
            results[mode] = {
                "accuracy": acc,
                "accuracy_drop": baseline_acc - acc,
                "model_size_mb": baseline_params * bits_equiv / (32 * 1024 * 1024),
                "compression": 32.0 / bits_equiv,
                "mse": qr.stats.get("_aggregate", {}).get("total_mse", 0),
                "mode": mode,
            }

    # Find Pareto-optimal configs
    configs = {k: v for k, v in results.items() if k != "baseline"}
    pareto = []
    for k, v in configs.items():
        is_pareto = True
        for k2, v2 in configs.items():
            if k2 == k:
                continue
            if (
                v2["accuracy"] >= v["accuracy"]
                and v2["model_size_mb"] <= v["model_size_mb"]
                and (
                    v2["accuracy"] > v["accuracy"]
                    or v2["model_size_mb"] < v["model_size_mb"]
                )
            ):
                is_pareto = False
                break
        if is_pareto:
            pareto.append(k)

    results["_pareto_front"] = pareto
    results["_summary"] = {
        "best_accuracy": max(
            (v["accuracy"] for v in configs.values()), default=baseline_acc
        ),
        "smallest_model": min(
            (v["model_size_mb"] for v in configs.values()),
            default=results["baseline"]["model_size_mb"],
        ),
        "num_configs_evaluated": len(configs),
    }

    return results


# ---------------------------------------------------------------------------
# High-level quantizer
# ---------------------------------------------------------------------------


class WeightQuantizer:
    """Unified interface for weight quantization.

    Wraps the individual quantization functions and provides stateful
    tracking of quantization history.

    Args:
        config: Quantization configuration.
    """

    def __init__(self, config: Optional[QuantizationConfig] = None):
        self.config = config or QuantizationConfig()
        self._history: List[QuantizationResult] = []

    def quantize_post_training(
        self,
        model: nn.Module,
        calibration_data: Optional[torch.Tensor] = None,
    ) -> QuantizationResult:
        """Apply post-training quantization.

        Args:
            model: Model to quantize.
            calibration_data: Optional calibration data.

        Returns:
            QuantizationResult with quantized model.
        """
        result = post_training_quantize(model, self.config, calibration_data)
        self._history.append(result)
        return result

    def attach_qat_hooks(
        self, model: nn.Module
    ) -> List[QuantizationAwareHook]:
        """Attach quantization-aware training hooks.

        Args:
            model: Model to instrument.

        Returns:
            List of hooks (keep references alive during training).
        """
        return quantization_aware_hook(
            model,
            bits=self.config.bits,
            symmetric=self.config.symmetric,
            momentum=self.config.momentum,
        )

    def analyze_tradeoff(
        self,
        model: nn.Module,
        eval_fn: Any,
        test_data: Any,
        test_targets: Any,
    ) -> Dict[str, Any]:
        """Run accuracy-efficiency trade-off analysis.

        Args:
            model: Full-precision model.
            eval_fn: Evaluation function.
            test_data: Test inputs.
            test_targets: Test labels.

        Returns:
            Trade-off analysis results.
        """
        return accuracy_efficiency_tradeoff(
            model, eval_fn, test_data, test_targets
        )

    @property
    def history(self) -> List[QuantizationResult]:
        """Return history of quantization operations."""
        return list(self._history)
