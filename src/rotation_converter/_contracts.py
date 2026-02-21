"""Design by Contract shim for rotation_converter.

Re-exports from the monorepo's shared ``contracts`` module when available.
Falls back to lightweight standalone implementations so the package works
outside the monorepo (e.g. pip-installed or copied into another project).

Consumers inside rotation_converter should always import from here::

    from rotation_converter._contracts import require, ensure, ...

This keeps the package self-contained while still using the full DbC
infrastructure when running inside the Tools monorepo.
"""

from __future__ import annotations

from typing import Any

try:
    # Monorepo path — full-featured DbC from src/shared/python/contracts.py
    from contracts import (  # type: ignore[import-untyped]
        PostconditionError,
        PreconditionError,
        ensure,
        require,
        require_finite,
        require_unit_vector,
    )
except ImportError:
    # ── Standalone fallback ──────────────────────────────────────
    import numpy as _np

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

    def require(condition: bool, message: str, value: Any = None) -> None:  # type: ignore[no-redef]
        """Assert a pre-condition."""
        if not condition:
            raise PreconditionError(message, value)

    def ensure(condition: bool, message: str, value: Any = None) -> None:  # type: ignore[no-redef]
        """Assert a post-condition."""
        if not condition:
            raise PostconditionError(message, value)

    def require_finite(array: Any, name: str = "array") -> None:  # type: ignore[no-redef]
        """Require all elements to be finite (no NaN / Inf)."""
        if not _np.all(_np.isfinite(array)):
            raise PreconditionError(f"{name} contains NaN or Inf values")

    def require_unit_vector(  # type: ignore[no-redef]
        vector: Any, name: str = "vector", tol: float = 1e-6
    ) -> None:
        """Require vector to have unit length."""
        norm = float(_np.linalg.norm(vector))
        if abs(norm - 1.0) > tol:
            raise PreconditionError(f"{name} must be a unit vector (norm = {norm})")


__all__ = [
    "PreconditionError",
    "PostconditionError",
    "ensure",
    "require",
    "require_finite",
    "require_unit_vector",
]
