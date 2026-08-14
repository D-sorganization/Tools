"""Shared boundary validation for spatial target domain and persistence code."""

from __future__ import annotations

import math
from numbers import Real
from typing import Literal, TypeAlias

TargetFrame: TypeAlias = Literal["app", "flight"]
TargetKind: TypeAlias = Literal["landing_area", "aerial_waypoint"]
ElevationSource: TypeAlias = Literal["course_surface", "absolute"]
Vector3: TypeAlias = tuple[float, float, float]

_VALID_FRAMES = frozenset(("app", "flight"))


def finite_float(value: object, name: str) -> float:
    """Return a finite real value, rejecting booleans and coercive strings."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def positive_float(value: object, name: str) -> float:
    """Return a finite strictly positive real value."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be finite and > 0")
    return number


def vector3(value: object, name: str) -> Vector3:
    """Defensively copy exactly three finite coordinates into a tuple."""
    if isinstance(value, (str, bytes)):
        raise TypeError(f"{name} must be an iterable of three coordinates")
    try:
        values: tuple[object, ...] = tuple(value)  # type: ignore[arg-type]
    except TypeError as exc:
        raise TypeError(f"{name} must be an iterable of three coordinates") from exc
    if len(values) != 3:
        raise ValueError(f"{name} must contain exactly three coordinates")
    return tuple(finite_float(item, name) for item in values)  # type: ignore[return-value]


def target_frame(value: object, name: str = "frame") -> TargetFrame:
    """Validate an app/flight frame discriminator."""
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if value not in _VALID_FRAMES:
        raise ValueError(f"{name} must be 'app' or 'flight'; got {value!r}")
    return value  # type: ignore[return-value]


def nonempty_text(value: object, name: str) -> str:
    """Validate a non-empty string with no ambiguous surrounding whitespace."""
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if not value or value != value.strip():
        raise ValueError(f"{name} must be non-empty and trimmed")
    return value


__all__: list[str] = []
