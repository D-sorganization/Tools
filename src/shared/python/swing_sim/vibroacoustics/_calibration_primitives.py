"""Unconditional calibration boundaries; identity text is never a file path."""

from __future__ import annotations

import re

from .measurement import _finite as finite_scalar


def identity_text(value: object, label: str) -> str:
    """Require trimmed text without control characters or lone surrogates."""
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{label} must be nonempty trimmed text")
    if any(
        ord(c) < 32 or 127 <= ord(c) <= 159 or 0xD800 <= ord(c) <= 0xDFFF for c in value
    ):
        raise ValueError(f"{label} cannot contain controls or surrogates")
    return value


def sha256_reference(value: object, label: str) -> str:
    """Validate a digest reference without authenticating its alleged source."""
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def interval(value: object, label: str) -> tuple[float, float]:
    """Require an immutable finite interval with distinct ordered endpoints."""
    if not isinstance(value, tuple) or len(value) != 2:
        raise TypeError(f"{label} must be a two-element tuple")
    lower, upper = (finite_scalar(item, label) for item in value)
    if upper <= lower:
        raise ValueError(f"{label} must increase strictly")
    return lower, upper


def instance(value: object, expected: type, label: str) -> None:
    """Refuse implicit mapping/array conversion at record boundaries."""
    if not isinstance(value, expected):
        raise TypeError(f"{label} must be {expected.__name__}")


__all__ = ()
