"""Thin adapter over the rotation converter's screw-axis extraction.

Per the #4108 recon, ALL instantaneous-screw-axis (ISA) computation for
the simulation goes through this one module:

* ``rotation_converter`` emits a ``DeprecationWarning`` at import (the
  package is migrating to the Rust ``tools_core.math_primitives``, which
  has no se(3)/screw surface yet) — the import is confined here and the
  warning suppressed, so the eventual migration touches one file.
* ``extract_screw_axes_from_trajectory`` returns per-step rotation
  ANGLES, not rates — this adapter divides ``theta`` by the sampling
  ``dt`` to report deg/s.
* The ISA is ill-conditioned as the per-step ``theta`` approaches zero,
  so callers should sample densely near impact (the session's uniform
  1 ms grid) and treat near-zero-rate entries as "no meaningful axis".
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Sequence
from typing import Any

import numpy as np

from rate_of_closure._contracts import require

__all__ = ["MIN_RATE_DPS", "screw_axis_samples"]

with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    from rotation_converter.screw_visualization import (
        extract_screw_axes_from_trajectory,
    )

#: Below this rotation rate the extracted axis direction is numerically
#: meaningless (theta -> 0 ill-conditioning); callers should skip it.
MIN_RATE_DPS = 1.0


def screw_axis_samples(
    poses: Sequence[np.ndarray] | np.ndarray, dt: float
) -> list[dict[str, Any]]:
    """Instantaneous screw axes for a uniformly sampled pose trajectory.

    Args:
        poses: N SE(3) matrices (sequence of 4x4 arrays or an (N, 4, 4)
            array), uniformly spaced ``dt`` seconds apart, N >= 2.
        dt: Sampling interval [s], > 0.

    Returns:
        N-1 dicts with keys ``axis`` (unit 3-vector, world frame),
        ``point`` (3-vector on the axis, world frame), ``pitch``
        (translation per radian; ``inf`` for pure translation),
        ``rate_dps`` (rotation rate, degrees per second), ``r_isa_m``
        (distance from the segment midpoint to the axis, meters;
        ``inf`` when the rate is below :data:`MIN_RATE_DPS`), and
        ``midpoint``.
    """
    require(math.isfinite(dt) and dt > 0.0, "dt must be finite and > 0", dt)
    pose_list = [np.asarray(p, dtype=float) for p in poses]
    require(len(pose_list) >= 2, "need at least 2 poses", len(pose_list))

    out: list[dict[str, Any]] = []
    for raw in extract_screw_axes_from_trajectory(pose_list):
        rate_dps = math.degrees(float(raw["theta"])) / dt
        axis = np.asarray(raw["axis"], dtype=float)
        point = np.asarray(raw["point"], dtype=float)
        midpoint = np.asarray(raw["midpoint"], dtype=float)
        if rate_dps >= MIN_RATE_DPS:
            to_axis = midpoint - point
            r_isa = float(np.linalg.norm(to_axis - float(to_axis @ axis) * axis))
        else:
            r_isa = math.inf
        out.append(
            {
                "axis": axis,
                "point": point,
                "pitch": float(raw["pitch"]),
                "rate_dps": rate_dps,
                "r_isa_m": r_isa,
                "midpoint": midpoint,
            }
        )
    return out
