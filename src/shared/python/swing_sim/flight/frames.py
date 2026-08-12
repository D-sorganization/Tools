"""Frame adapters between the app frame and the UpstreamDrift flight frame.

- Flight frame (UpstreamDrift physics stack, and everything inside
  :mod:`shared.python.swing_sim.flight`): x forward, y left, z up.
- App frame (Tools AffineDrift / ``tools_core.ball_flight`` Rust kernel):
  x target (downrange), y up, z right.

Both are right-handed, related by a pure rotation, so the same adapter
applies to positions, velocities, angular velocities, and spin axes.

Mapping: app x = flight x; app y = flight z; app z = -flight y.
"""

from __future__ import annotations

import numpy as np


def _validated(vec: np.ndarray, name: str) -> np.ndarray:
    """Return ``vec`` as a float array of shape (3,) or (N, 3)."""
    arr = np.asarray(vec, dtype=float)
    if arr.shape != (3,) and not (arr.ndim == 2 and arr.shape[1] == 3):
        raise ValueError(f"{name} must have shape (3,) or (N, 3); got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be finite")
    return arr


def to_flight_frame(vec_app: np.ndarray) -> np.ndarray:
    """Convert vector(s) from the app frame to the flight frame.

    Args:
        vec_app: Vector(s) in the app frame (x target, y up, z right),
            shape (3,) or (N, 3).

    Returns:
        Vector(s) in the flight frame (x forward, y left, z up), same shape.
    """
    arr = _validated(vec_app, "vec_app")
    out = np.empty_like(arr)
    out[..., 0] = arr[..., 0]
    out[..., 1] = -arr[..., 2]
    out[..., 2] = arr[..., 1]
    return out


def from_flight_frame(vec_flight: np.ndarray) -> np.ndarray:
    """Convert vector(s) from the flight frame to the app frame.

    Args:
        vec_flight: Vector(s) in the flight frame (x forward, y left, z up),
            shape (3,) or (N, 3).

    Returns:
        Vector(s) in the app frame (x target, y up, z right), same shape.
    """
    arr = _validated(vec_flight, "vec_flight")
    out = np.empty_like(arr)
    out[..., 0] = arr[..., 0]
    out[..., 1] = arr[..., 2]
    out[..., 2] = -arr[..., 1]
    return out


__all__ = ["from_flight_frame", "to_flight_frame"]
