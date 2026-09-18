"""File I/O helpers — JSON, directories, and PyTorch checkpoints."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Union

import torch


def ensure_dir(path: Union[str, Path]) -> Path:
    """Create a directory (and parents) if it doesn't exist.

    Args:
        path: Directory path.

    Returns:
        The resolved ``Path`` object.
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(data: Any, path: Union[str, Path], *, indent: int = 2) -> None:
    """Serialize *data* to a JSON file.

    Args:
        data:   JSON-serialisable object.
        path:   Destination file path.
        indent: JSON indentation level.
    """
    p = Path(path)
    ensure_dir(p.parent)
    with p.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def load_json(path: Union[str, Path]) -> Dict[str, Any]:
    """Load a JSON file and return the parsed object.

    Args:
        path: Path to the JSON file.

    Returns:
        Parsed dictionary / list.

    Raises:
        FileNotFoundError: If *path* does not exist.
        json.JSONDecodeError: If the file contains invalid JSON.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"No such file: {p}")
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)  # type: ignore[no-any-return]


def save_checkpoint(state: Dict[str, Any], path: Union[str, Path]) -> None:
    """Save a training checkpoint via ``torch.save``.

    Args:
        state: Dictionary to persist (must be picklable).
        path:  Destination file path.
    """
    p = Path(path)
    ensure_dir(p.parent)
    torch.save(state, p)


def load_checkpoint(path: Union[str, Path]) -> Dict[str, Any]:
    """Load a training checkpoint.

    Args:
        path: Path to the checkpoint file.

    Returns:
        The restored state dictionary.

    Raises:
        FileNotFoundError: If *path* does not exist.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"No such checkpoint: {p}")
    return torch.load(p, weights_only=False)  # type: ignore[no-any-return]
