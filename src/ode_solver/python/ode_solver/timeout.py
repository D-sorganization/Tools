"""Timeout utilities for ODE solver computations.

Provides a cross-platform timeout wrapper using ``threading.Thread``
so that long-running or divergent ODE integrations can be interrupted
before they hang the application indefinitely.

Design notes:
- Uses daemon threads: the background thread is abandoned on timeout,
  not killed. This is safe for scipy ``solve_ivp`` which holds no OS
  resources beyond CPU time.
- ``signal.SIGALRM`` is intentionally avoided: it is UNIX-only and
  incompatible with ``pytest-xdist`` parallel workers.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from typing import Any, TypeVar

_log = logging.getLogger(__name__)

_T = TypeVar("_T")


class SolverTimeoutError(TimeoutError):
    """Raised when a solver computation exceeds its allotted time budget."""


def with_timeout(
    seconds: float,
    func: Callable[..., _T],
    /,
    *args: Any,
    **kwargs: Any,
) -> _T:
    """Run ``func(*args, **kwargs)`` and raise ``SolverTimeoutError`` if it exceeds *seconds*.

    Preconditions:
        ``seconds`` must be a positive numeric value (int or float).
        ``func`` must be callable.

    Args:
        seconds: Maximum wall-clock time allowed for the computation.
        func: Callable to execute.
        *args: Positional arguments forwarded to *func*.
        **kwargs: Keyword arguments forwarded to *func*.

    Returns:
        Whatever *func* returns.

    Raises:
        TypeError: If *seconds* is not numeric or *func* is not callable.
        ValueError: If *seconds* is not positive.
        SolverTimeoutError: If the computation does not complete within *seconds*.
        Exception: Any exception raised by *func* is re-raised in the caller's thread.
    """
    if not isinstance(seconds, (int, float)):
        raise TypeError(
            f"seconds must be a numeric value (int or float), got {type(seconds).__name__}"
        )
    if seconds <= 0:
        raise ValueError(f"seconds must be > 0, got {seconds!r}")
    if not callable(func):
        raise TypeError(f"func must be callable, got {type(func).__name__}")

    result: list[_T] = []
    exc_holder: list[BaseException] = []

    def _target() -> None:
        try:
            result.append(func(*args, **kwargs))
        except Exception as err:  # noqa: BLE001
            exc_holder.append(err)

    worker = threading.Thread(target=_target, daemon=True, name="ode-solver-worker")
    worker.start()
    worker.join(timeout=seconds)

    if worker.is_alive():
        _log.warning("ODE solver timed out after %.1f s", seconds)
        raise SolverTimeoutError(
            f"ODE solver computation timed out after {seconds:.1f} s. "
            "Consider simplifying the system, reducing the time span, or increasing the timeout."
        )

    if exc_holder:
        raise exc_holder[0]

    return result[0]
