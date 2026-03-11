"""Vessel drafter project builders."""

from __future__ import annotations

from typing import Any

__all__ = [
    "build_vessel_drafter_shape",
]


def build_vessel_drafter_shape(*args: Any, **kwargs: Any) -> Any:
    from .vessel_drafter_layout import build_vessel_drafter_shape as _fn

    return _fn(*args, **kwargs)
