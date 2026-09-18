"""Checkpoint management for BIO-NN.

Handles saving/loading full training state, rotation, and resume logic.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn


class CheckpointManager:
    """Manage model checkpoints with rotation and metadata tracking.

    Args:
        checkpoint_dir: Root directory for checkpoints.
        max_checkpoints: Maximum saved checkpoints (0 = unlimited).
        save_latest: Always save a ``latest.pt`` symlink/copy.
    """

    def __init__(
        self,
        checkpoint_dir: str | Path,
        max_checkpoints: int = 5,
        save_latest: bool = True,
    ) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.max_checkpoints = max_checkpoints
        self.save_latest = save_latest
        self._checkpoints: List[Path] = sorted(
            self.checkpoint_dir.glob("checkpoint_*.pt"),
            key=lambda p: p.stat().st_mtime,
        )

    def save(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        metrics: Optional[Dict[str, float]] = None,
        config: Optional[Dict[str, Any]] = None,
        extra: Optional[Dict[str, Any]] = None,
        scaler: Optional[torch.amp.GradScaler] = None,
    ) -> Path:
        """Save a full checkpoint.

        Args:
            model:     Model state_dict.
            optimizer: Optimizer state_dict.
            epoch:     Current epoch number.
            metrics:   Validation metrics to record.
            config:    Training config to embed.
            extra:     Arbitrary extra state (scheduler, etc.).
            scaler:    Optional AMP GradScaler state.

        Returns:
            Path to saved checkpoint.
        """
        state: Dict[str, Any] = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "timestamp": time.time(),
            "metrics": metrics or {},
            "config": config or {},
        }
        if extra is not None:
            state["extra"] = extra
        if scaler is not None:
            state["scaler_state_dict"] = scaler.state_dict()

        path = self.checkpoint_dir / f"checkpoint_{epoch:06d}.pt"
        torch.save(state, path)
        self._checkpoints.append(path)

        if self.save_latest:
            self._save_latest(model, optimizer, epoch, metrics, config, extra, scaler)

        self._rotate()
        self._save_metadata(path, state)
        return path

    def load(
        self,
        path: Optional[str | Path] = None,
        model: Optional[nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scaler: Optional[torch.amp.GradScaler] = None,
    ) -> Dict[str, Any]:
        """Load a checkpoint.

        Args:
            path:      Path to checkpoint. None = latest.
            model:     If provided, loads state_dict.
            optimizer: If provided, loads state_dict.
            scaler:    If provided, loads scaler state.

        Returns:
            Full checkpoint state dict.
        """
        if path is None:
            path = self._find_latest()
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        state = torch.load(path, map_location="cpu", weights_only=False)

        if model is not None:
            model.load_state_dict(state["model_state_dict"])
        if optimizer is not None:
            optimizer.load_state_dict(state["optimizer_state_dict"])
        if scaler is not None and "scaler_state_dict" in state:
            scaler.load_state_dict(state["scaler_state_dict"])

        return state

    def resume(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scaler: Optional[torch.amp.GradScaler] = None,
    ) -> Dict[str, Any]:
        """Auto-resume from the latest checkpoint.

        Returns checkpoint state including epoch, metrics, config.
        Raises FileNotFoundError if no checkpoint exists.
        """
        latest = self._find_latest()
        return self.load(latest, model, optimizer, scaler)

    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """List all saved checkpoints with metadata."""
        results = []
        for p in self._checkpoints:
            meta_path = p.with_suffix(".json")
            if meta_path.exists():
                with open(meta_path, "r") as f:
                    meta = json.load(f)
            else:
                meta = {"epoch": self._extract_epoch(p)}
            meta["path"] = str(p)
            results.append(meta)
        return results

    def delete(self, path: str | Path) -> None:
        """Delete a specific checkpoint."""
        path = Path(path)
        if path.exists():
            path.unlink()
        meta = path.with_suffix(".json")
        if meta.exists():
            meta.unlink()
        self._checkpoints = [p for p in self._checkpoints if p != path]

    def _save_latest(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        metrics: Optional[Dict[str, float]],
        config: Optional[Dict[str, Any]],
        extra: Optional[Dict[str, Any]],
        scaler: Optional[torch.amp.GradScaler],
    ) -> None:
        state: Dict[str, Any] = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "timestamp": time.time(),
            "metrics": metrics or {},
            "config": config or {},
        }
        if extra is not None:
            state["extra"] = extra
        if scaler is not None:
            state["scaler_state_dict"] = scaler.state_dict()
        torch.save(state, self.checkpoint_dir / "latest.pt")

    def _rotate(self) -> None:
        if self.max_checkpoints <= 0:
            return
        while len(self._checkpoints) > self.max_checkpoints:
            oldest = self._checkpoints.pop(0)
            self.delete(oldest)

    def _find_latest(self) -> Path:
        latest = self.checkpoint_dir / "latest.pt"
        if latest.exists():
            return latest
        if self._checkpoints:
            return self._checkpoints[-1]
        raise FileNotFoundError(
            f"No checkpoints found in {self.checkpoint_dir}"
        )

    def _save_metadata(self, path: Path, state: Dict[str, Any]) -> None:
        meta = {
            "epoch": state.get("epoch"),
            "timestamp": state.get("timestamp"),
            "metrics": state.get("metrics", {}),
        }
        meta_path = path.with_suffix(".json")
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2, default=str)

    @staticmethod
    def _extract_epoch(path: Path) -> int:
        name = path.stem  # checkpoint_000042
        parts = name.split("_")
        try:
            return int(parts[-1])
        except (ValueError, IndexError):
            return 0
