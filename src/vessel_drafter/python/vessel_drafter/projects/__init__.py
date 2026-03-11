"""Vessel drafter project builders."""

__all__ = [
    "build_vessel_drafter_shape",
]


def build_vessel_drafter_shape(*args, **kwargs):
    from .vessel_drafter_layout import build_vessel_drafter_shape as _fn

    return _fn(*args, **kwargs)
