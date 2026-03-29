"""Shared internal helpers for the Modern Robotics submodules."""

from __future__ import annotations


def _near_zero(val: float, tol: float = 1e-12) -> bool:
    """Check if a scalar is effectively zero."""
    return abs(val) < tol
