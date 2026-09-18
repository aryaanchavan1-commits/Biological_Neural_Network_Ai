"""Timing utilities for wall-clock measurement."""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Generator, Optional


class Timer:
    """Simple wall-clock timer.

    Usage::

        timer = Timer()
        timer.start()
        ... work ...
        elapsed = timer.stop()  # seconds
    """

    def __init__(self) -> None:
        self._start: Optional[float] = None
        self._elapsed: float = 0.0

    def start(self) -> Timer:
        """Start the timer.  Returns *self* for chaining."""
        self._start = time.perf_counter()
        return self

    def stop(self) -> float:
        """Stop the timer and return elapsed seconds.

        Raises:
            RuntimeError: If the timer was never started.
        """
        if self._start is None:
            raise RuntimeError("Timer was never started")
        self._elapsed = time.perf_counter() - self._start
        self._start = None
        return self._elapsed

    @property
    def elapsed(self) -> float:
        """Elapsed seconds (from last ``start`` / ``stop`` pair)."""
        return self._elapsed

    def reset(self) -> Timer:
        """Reset elapsed time.  Returns *self* for chaining."""
        self._elapsed = 0.0
        self._start = None
        return self

    def __repr__(self) -> str:
        return f"Timer(elapsed={self._elapsed:.6f}s)"


@contextmanager
def TimerContext(label: Optional[str] = None) -> Generator[Timer, None, None]:
    """Context-manager wrapper around :class:`Timer`.

    Usage::

        with TimerContext("inference") as t:
            model(x)
        print(t.elapsed)
    """
    timer = Timer().start()
    try:
        yield timer
    finally:
        timer.stop()
        if label:
            print(f"[Timer] {label}: {timer.elapsed:.6f}s")
