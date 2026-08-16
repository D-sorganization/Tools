"""Design by Contract shim for rate_of_closure.

Re-exports from the monorepo's shared ``contracts`` module when available.
Falls back to lightweight standalone implementations so the package works
outside the monorepo (e.g. vendored into UpstreamDrift or copied out).

Consumers inside rate_of_closure should always import from here::

    from ._contracts import require, ensure, require_finite
"""

from __future__ import annotations

import math
from typing import Any

__all__ = [
    "PostconditionError",
    "PreconditionError",
    "ensure",
    "require",
    "require_finite",
]

try:
    # Monorepo path — full-featured DbC from src/shared/python/contracts.py
    from shared.python.contracts import (
        PostconditionError,
        PreconditionError,
        ensure,
        require,
        require_finite,
    )
except ImportError:
    # ── Standalone fallback ──────────────────────────────────────

    class PreconditionError(AssertionError, ValueError):  # type: ignore[no-redef]
        """Raised when a pre-condition is violated."""

        def __init__(self, message: str, value: Any = None) -> None:
            detail = f"[DbC pre-condition] {message}"
            if value is not None:
                detail += f" (got: {value!r})"
            super().__init__(detail)

    class PostconditionError(AssertionError, ValueError):  # type: ignore[no-redef]
        """Raised when a post-condition is violated."""

        def __init__(self, message: str, value: Any = None) -> None:
            detail = f"[DbC post-condition] {message}"
            if value is not None:
                detail += f" (got: {value!r})"
            super().__init__(detail)

    def require(condition: bool, message: str, value: Any = None) -> None:
        """Raise :class:`PreconditionError` when *condition* is false."""
        if not condition:
            raise PreconditionError(message, value)

    def ensure(condition: bool, message: str, value: Any = None) -> None:
        """Raise :class:`PostconditionError` when *condition* is false."""
        if not condition:
            raise PostconditionError(message, value)

    def require_finite(value: float, name: str = "value") -> None:
        """Raise :class:`PreconditionError` when *value* is NaN or infinite."""
        if not math.isfinite(value):
            raise PreconditionError(f"{name} must be finite", value)
