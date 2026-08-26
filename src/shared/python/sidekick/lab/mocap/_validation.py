"""Private validation helpers for markerless-mocap contracts."""

from __future__ import annotations

import math
import re
from collections.abc import Iterable

_SEMVER_PATTERN = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+(?:[-+][A-Za-z0-9.-]+)?$")
__all__: list[str] = []


def require_text(value: str, field_name: str) -> str:
    """Return a stripped non-empty string or raise ``ValueError``."""
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must be non-empty")
    return normalized


def require_finite(value: float, field_name: str) -> float:
    """Return a finite float or raise ``TypeError``/``ValueError``."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be a real number")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{field_name} must be finite")
    return normalized


def require_nonnegative_integer(value: int, field_name: str) -> int:
    """Return a non-negative integer or raise a contract error."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be an integer")
    if value < 0:
        raise ValueError(f"{field_name} must be non-negative")
    return value


def require_unique_text(values: Iterable[str], field_name: str) -> tuple[str, ...]:
    """Return stripped, non-empty, unique strings preserving order."""
    normalized = tuple(require_text(value, field_name) for value in values)
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{field_name} must contain unique values")
    return normalized


def require_semver(value: str, field_name: str) -> str:
    """Return a semantic version string or raise ``ValueError``."""
    normalized = require_text(value, field_name)
    if not _SEMVER_PATTERN.fullmatch(normalized):
        raise ValueError(f"{field_name} must be semantic version text")
    return normalized
