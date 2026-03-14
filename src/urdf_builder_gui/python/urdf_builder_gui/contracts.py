"""Lightweight Design by Contract support for urdf_builder_gui.

Provides require() and ensure() for precondition and postcondition
enforcement without depending on the data_processor package.
"""

from __future__ import annotations

import os
from typing import Any


class PreconditionError(ValueError):
    """Raised when a precondition is violated."""


class PostconditionError(ValueError):
    """Raised when a postcondition is violated."""


# DBC_LEVEL controls enforcement: "enforce" (default), "warn", "disabled"
_DBC_LEVEL = os.environ.get("DBC_LEVEL", "enforce")


def require(condition: bool, message: str, *args: Any) -> None:
    """Assert a precondition.

    Args:
        condition: The boolean condition that must be true.
        message: Error message if violated.
        *args: Additional context values to include in the error.
    """
    if _DBC_LEVEL == "disabled":
        return
    if not condition:
        detail = f"{message}"
        if args:
            detail += f" (got: {', '.join(str(a) for a in args)})"
        raise PreconditionError(detail)


def ensure(condition: bool, message: str, *args: Any) -> None:
    """Assert a postcondition.

    Args:
        condition: The boolean condition that must be true.
        message: Error message if violated.
        *args: Additional context values.
    """
    if _DBC_LEVEL == "disabled":
        return
    if not condition:
        detail = f"{message}"
        if args:
            detail += f" (got: {', '.join(str(a) for a in args)})"
        raise PostconditionError(detail)


__all__ = [
    "PreconditionError",
    "PostconditionError",
    "ensure",
    "require",
]
