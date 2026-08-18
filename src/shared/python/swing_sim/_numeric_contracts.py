"""Strict scalar-domain validators shared by swing simulation contracts."""

from __future__ import annotations

import math
from numbers import Integral, Real
from typing import cast

from shared.python.contracts import require


def finite_real(value: object, name: str) -> float:
    """Return one finite real scalar without Boolean or string coercion."""
    require(
        isinstance(value, Real) and not isinstance(value, bool),
        f"{name} must be a real non-boolean scalar",
        value,
    )
    normalized = float(cast(Real, value))
    require(math.isfinite(normalized), f"{name} must be finite", value)
    return normalized


def integer(value: object, name: str, *, minimum: int = 0) -> int:
    """Return one integer scalar without Boolean, float, or string coercion."""
    require(
        isinstance(value, Integral) and not isinstance(value, bool),
        f"{name} must be an integer",
        value,
    )
    normalized = int(cast(Integral, value))
    require(normalized >= minimum, f"{name} must be >= {minimum}", value)
    return normalized


__all__ = ["finite_real", "integer"]
