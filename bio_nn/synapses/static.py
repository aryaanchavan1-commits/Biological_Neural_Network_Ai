import torch
import torch.nn as nn

from .base import BaseSynapse


class StaticSynapse(BaseSynapse):
    """Synapse with fixed (non-plastic) weights.

    Supports multiple weight initialisation schemes and optional
    magnitude clipping.

    Config keys:
        init (str): Weight initialisation. One of {"xavier", "kaiming", "normal", "uniform"}.
                    Default: "xavier".
        clip_min (float | None): Minimum allowed weight value. Default: None (disabled).
        clip_max (float | None): Maximum allowed weight value. Default: None (disabled).
    """

    _INIT_METHODS = ("xavier", "kaiming", "normal", "uniform")

    def __init__(self, in_features: int, out_features: int, config: dict) -> None:
        super().__init__(in_features, out_features, config)

        init_method = config.get("init", "xavier")
        if init_method not in self._INIT_METHODS:
            raise ValueError(
                f"Unknown init method '{init_method}'. Must be one of {self._INIT_METHODS}"
            )

        self._clip_min = config.get("clip_min", None)
        self._clip_max = config.get("clip_max", None)

        self._init_weights(init_method)

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _init_weights(self, method: str) -> None:
        if method == "xavier":
            nn.init.xavier_uniform_(self.weights)
        elif method == "kaiming":
            nn.init.kaiming_uniform_(self.weights, nonlinearity="linear")
        elif method == "normal":
            nn.init.normal_(self.weights)
        elif method == "uniform":
            nn.init.uniform_(self.weights, -1.0, 1.0)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, pre_spikes: torch.Tensor) -> torch.Tensor:
        """Multiply pre-synaptic spikes by the weight matrix.

        Args:
            pre_spikes: (batch, in_features)

        Returns:
            (batch, out_features)
        """
        w = self.weights
        if self._clip_min is not None or self._clip_max is not None:
            w = torch.clamp(w, min=self._clip_min, max=self._clip_max)
        return pre_spikes @ w.t()

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def get_statistics(self) -> dict:
        stats = super().get_statistics()
        stats["clip_min"] = self._clip_min
        stats["clip_max"] = self._clip_max
        return stats
