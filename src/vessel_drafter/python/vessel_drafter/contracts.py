"""Design by Contract helpers for the vessel_drafter package.

Re-exports the fleet-standard contract primitives from the shared
``contracts`` module (``src/shared/python/contracts.py``).  Domain-specific
helpers (``require_nonnegative``, ``require_fraction``, etc.) are kept here
as thin wrappers around ``require()``.

Signature fix (closes #1930): ``require_positive(value, name)`` now matches
the canonical signature in ``src/shared/python/contracts.py``.

De-duplicated per https://github.com/D-sorganization/Tools/issues/1926.
"""

from __future__ import annotations

import logging
from math import isfinite
from typing import Any

logger = logging.getLogger(__name__)

# ── Re-export shared contracts primitives (monorepo path) ──────────────────

try:
    from contracts import (
        PreconditionError,
        ensure,
        require,
        require_positive,
    )
except ImportError:
    # ── Standalone fallback ────────────────────────────────────────────────

    class PreconditionError(AssertionError, ValueError):  # type: ignore[no-redef]
        """Raised when a pre-condition is violated."""

        def __init__(self, message: str, value: Any = None) -> None:
            detail = f"[DbC pre-condition] {message}"
            if value is not None:
                detail += f" (got: {value!r})"
            super().__init__(detail)

    def require(condition: bool, message: str, value: Any = None) -> None:  # type: ignore[misc]
        """Assert a pre-condition (standard bool-style API)."""
        if not condition:
            raise PreconditionError(message, value)

    def ensure(condition: bool, message: str, value: Any = None) -> None:  # type: ignore[misc]
        """Assert a post-condition (standard bool-style API)."""
        if not condition:
            raise ValueError(f"[DbC post-condition] {message}")

    def require_positive(value: float, name: str = "value") -> None:  # type: ignore[misc]
        """Require that *value* is strictly positive."""
        if value <= 0:
            raise PreconditionError(f"{name} must be positive (got {value})")


# ── Domain-specific wrapper helpers ──────────────────────────────────────


def require_nonnegative(name: str, value: float) -> None:
    """Assert that *value* is non-negative (>= 0).

    Args:
        name: Human-readable parameter name for error messages.
        value: The numeric value to check.

    Raises:
        PreconditionError: If *value* < 0.
    """
    if value < 0.0:
        raise PreconditionError(f"{name} must be nonnegative, got {value!r}", value)


def require_fraction(name: str, value: float) -> None:
    """Assert that *value* is a fraction in [0.0, 1.0].

    Args:
        name: Human-readable parameter name for error messages.
        value: The numeric value to check.

    Raises:
        PreconditionError: If *value* < 0.0 or *value* > 1.0.
    """
    if value < 0.0 or value > 1.0:
        raise PreconditionError(
            f"{name} must be between 0.0 and 1.0, got {value!r}", value
        )


def require_integer_at_least(name: str, value: int, minimum: int) -> None:
    """Assert that integer *value* is >= *minimum*.

    Args:
        name: Human-readable parameter name for error messages.
        value: The integer value to check.
        minimum: Inclusive lower bound.

    Raises:
        PreconditionError: If *value* < *minimum*.
    """
    if value < minimum:
        raise PreconditionError(f"{name} must be >= {minimum}, got {value!r}", value)


def require_less_or_equal(name: str, value: float, maximum: float) -> None:
    """Assert that *value* is <= *maximum*.

    Args:
        name: Human-readable parameter name for error messages.
        value: The numeric value to check.
        maximum: Inclusive upper bound.

    Raises:
        PreconditionError: If *value* > *maximum*.
    """
    if value > maximum:
        raise PreconditionError(f"{name} must be <= {maximum}, got {value!r}", value)


def require_finite(name: str, value: float) -> None:
    """Assert that *value* is a finite real number (no NaN or Inf).

    Args:
        name: Human-readable parameter name for error messages.
        value: The numeric value to check.

    Raises:
        PreconditionError: If *value* is NaN or infinite.
    """
    if not isfinite(value):
        raise PreconditionError(f"{name} must be finite, got {value!r}", value)


__all__ = [
    # Shared primitives
    "PreconditionError",
    "ensure",
    "require",
    "require_positive",
    # Domain-specific (name, value) wrappers
    "require_finite",
    "require_fraction",
    "require_integer_at_least",
    "require_less_or_equal",
    "require_nonnegative",
]
