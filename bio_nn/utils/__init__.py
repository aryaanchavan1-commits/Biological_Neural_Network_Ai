"""Utility package — seeds, timing, memory, hardware detection, I/O, datasets."""

from .seeds import set_seed, get_rng
from .timing import Timer, TimerContext
from .memory import get_peak_memory, get_current_memory, format_bytes
from .hardware import detect_device, get_hardware_info
from .io import save_json, load_json, ensure_dir, save_checkpoint, load_checkpoint
from .datasets import load_dataset, create_split_dataset

__all__ = [
    "set_seed",
    "get_rng",
    "Timer",
    "TimerContext",
    "get_peak_memory",
    "get_current_memory",
    "format_bytes",
    "detect_device",
    "get_hardware_info",
    "save_json",
    "load_json",
    "ensure_dir",
    "save_checkpoint",
    "load_checkpoint",
    "load_dataset",
    "create_split_dataset",
]
