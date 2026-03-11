"""Simple design-by-contract helpers used across drafting models."""

from __future__ import annotations

from math import isfinite


def require_positive(name: str, value: float) -> None:
    if value <= 0.0:
        raise ValueError(f"{name} must be positive, got {value!r}")


def require_nonnegative(name: str, value: float) -> None:
    if value < 0.0:
        raise ValueError(f"{name} must be nonnegative, got {value!r}")


def require_fraction(name: str, value: float) -> None:
    if value < 0.0 or value > 1.0:
        raise ValueError(f"{name} must be between 0.0 and 1.0, got {value!r}")


def require_integer_at_least(name: str, value: int, minimum: int) -> None:
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value!r}")


def require_less_or_equal(name: str, value: float, maximum: float) -> None:
    if value > maximum:
        raise ValueError(f"{name} must be <= {maximum}, got {value!r}")


def require_finite(name: str, value: float) -> None:
    if not isfinite(value):
        raise ValueError(f"{name} must be finite, got {value!r}")
