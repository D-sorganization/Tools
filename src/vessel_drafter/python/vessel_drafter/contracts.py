"""Design by Contract helpers for the vessel_drafter package.

Re-exports the fleet-standard contract primitives from the shared
``contracts`` module (``src/shared/python/contracts.py``) with lightweight
wrappers that preserve the legacy ``(name: str, value: float)`` parameter
order used throughout vessel_drafter source code.

Converged API (closes #1862)
-----------------------------
All callers inside vessel_drafter use the wrapper functions below, which
accept ``(name, value)`` positionally.  The shared ``require``/``ensure``
primitives and exception types are re-exported directly so callers may
also use the standard ``require(bool_expr, msg, value)`` style.

Fallback
--------
When the shared ``contracts`` module is not importable (e.g. standalone
pip-install without the monorepo) a minimal inline implementation is
provided so the package remains self-contained.
"""

from __future__ import annotations

import logging
from math import isfinite
from typing import Any

logger = logging.getLogger(__name__)

# ── Re-export shared contracts primitives (monorepo path) ──────────────────

try:
    from shared.python.contracts import (
        PostconditionError,
        PreconditionError,
        ensure,
        require,
    )
except ImportError:
    # ── Standalone fallback ────────────────────────────────────────────────
    import enum
    import os

    class ContractLevel(enum.Enum):
        OFF = "off"
        WARN = "warn"
        ENFORCE = "enforce"

    _LEVEL = ContractLevel(
        os.environ.get("DBC_LEVEL", "enforce").lower()
        if os.environ.get("DBC_LEVEL", "").lower() in ("off", "warn", "enforce")
        else "enforce"
    )

    class ContractViolationError(AssertionError, ValueError):
        """Base error for standalone vessel-drafter contract violations."""

        def __init__(self, kind: str, message: str, value: Any = None) -> None:
            detail = f"[DbC {kind}] {message}"
            if value is not None:
                detail += f" (got: {value!r})"
            super().__init__(detail)

    class PreconditionError(ContractViolationError):  # type: ignore[no-redef]
        """Raised when a pre-condition is violated."""

        def __init__(self, message: str, value: Any = None) -> None:
            super().__init__("pre-condition", message, value)

    class PostconditionError(ContractViolationError):  # type: ignore[no-redef]
        """Raised when a post-condition is violated."""

        def __init__(self, message: str, value: Any = None) -> None:
            super().__init__("post-condition", message, value)

    def _fail(kind: str, message: str, value: Any = None) -> None:
        if _LEVEL == ContractLevel.ENFORCE:
            error_type = {
                "pre-condition": PreconditionError,
                "post-condition": PostconditionError,
            }[kind]
            raise error_type(message, value)

    def require(condition: bool, message: str, value: Any = None) -> None:
        """Assert a pre-condition (standard bool-style API)."""
        if _LEVEL == ContractLevel.OFF:
            return
        if not condition:
            _fail("pre-condition", message, value)

    def ensure(condition: bool, message: str, value: Any = None) -> None:
        """Assert a post-condition (standard bool-style API)."""
        if _LEVEL == ContractLevel.OFF:
            return
        if not condition:
            _fail("post-condition", message, value)


# ── Legacy (name, value) wrapper helpers ───────────────────────────────────
# These preserve the parameter order used throughout vessel_drafter source
# files.  They delegate to ``require()`` so that the enforcement level set
# via ``set_contract_level()`` is respected.


def _coerce_name_value(
    first: object, second: object, helper_name: str
) -> tuple[str, float]:
    """Accept both legacy ``(name, value)`` and shared ``(value, name)`` order."""
    if isinstance(first, str) and isinstance(second, int | float):
        return first, float(second)
    if isinstance(second, str) and isinstance(first, int | float):
        return second, float(first)
    raise TypeError(
        f"{helper_name} expects (name: str, value: number) or "
        "(value: number, name: str)"
    )


def require_positive(name: str | float, value: float | str) -> None:
    """Assert that *value* is strictly positive.

    Args:
        name: Human-readable parameter name, or the numeric value when using
            the shared ``(value, name)`` call order.
        value: Numeric value to check, or the parameter name when using the
            shared ``(value, name)`` call order.

    Raises:
        PreconditionError: If *value* <= 0.
    """
    parameter_name, numeric_value = _coerce_name_value(name, value, "require_positive")
    require(
        numeric_value > 0.0,
        f"{parameter_name} must be positive, got {numeric_value!r}",
        numeric_value,
    )


def require_nonnegative(name: str, value: float) -> None:
    """Assert that *value* is non-negative (>= 0).

    Args:
        name: Human-readable parameter name for error messages.
        value: The numeric value to check.

    Raises:
        PreconditionError: If *value* < 0.
    """
    require(value >= 0.0, f"{name} must be nonnegative, got {value!r}", value)


def require_fraction(name: str, value: float) -> None:
    """Assert that *value* is a fraction in [0.0, 1.0].

    Args:
        name: Human-readable parameter name for error messages.
        value: The numeric value to check.

    Raises:
        PreconditionError: If *value* < 0.0 or *value* > 1.0.
    """
    require(
        0.0 <= value <= 1.0,
        f"{name} must be between 0.0 and 1.0, got {value!r}",
        value,
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
    require(value >= minimum, f"{name} must be >= {minimum}, got {value!r}", value)


def require_less_or_equal(name: str, value: float, maximum: float) -> None:
    """Assert that *value* is <= *maximum*.

    Args:
        name: Human-readable parameter name for error messages.
        value: The numeric value to check.
        maximum: Inclusive upper bound.

    Raises:
        PreconditionError: If *value* > *maximum*.
    """
    require(value <= maximum, f"{name} must be <= {maximum}, got {value!r}", value)


def require_finite(name: str, value: float) -> None:
    """Assert that *value* is a finite real number (no NaN or Inf).

    Args:
        name: Human-readable parameter name for error messages.
        value: The numeric value to check.

    Raises:
        PreconditionError: If *value* is NaN or infinite.
    """
    require(isfinite(value), f"{name} must be finite, got {value!r}", value)


__all__ = [
    # Shared primitives
    "PostconditionError",
    "PreconditionError",
    "ensure",
    "require",
    # Legacy (name, value) wrappers
    "require_finite",
    "require_fraction",
    "require_integer_at_least",
    "require_less_or_equal",
    "require_nonnegative",
    "require_positive",
]
