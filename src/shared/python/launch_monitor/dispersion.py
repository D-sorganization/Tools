"""Target-relative shot-dispersion statistics.

Ported from UpstreamDrift ``src/shared/python/launch_monitor/dispersion.py``
(70 lines) under ADR-0046 Stage 1 — step **P1** of the ADR-0046 G1 port plan
(UpstreamDrift ``docs/adr/0048-launch-monitor-port-plan.md``). The
implementation is UpstreamDrift's, carried over unchanged rather than
reimplemented; its authors retain authorship. No behaviour is added, removed,
or limited by the move.

**This is not** ``rate_of_closure.launch_monitor_performance.analyze_dispersion``.
The two share a name and nothing else, and ADR-0046 G0 pinned the gap
numerically on the shared ``adr0046_cross_stack_session_v1`` fixture:

* D6 — the two result dataclasses share **zero** field names. This module
  reports the median centre, the 95% covariance ellipse
  (major/minor/angle/area) and radial error about that centre; the
  ``rate_of_closure`` function reports lateral mean/sd/RMS plus
  left/centre/right counts. Neither is a subset of the other.
* D7 — "RMS" is a different estimand in each. ``radial_rmse`` here is 2-D
  about the median centre (11.364728588362174 on the gate fixture);
  ``rms_yards`` there is 1-D about zero lateral (8.39694421985684). Same
  fixture, 35% apart, and *not* a unit factor. They must never be reconciled
  by renaming.
* D8 — this module requires three complete shots (it needs a covariance);
  the ``rate_of_closure`` one accepts a single shot.
* D9 — this module declares and converts **no unit**: results come back in
  whatever unit the frame carries. The ``rate_of_closure`` one validates its
  declared unit and always reports yards.

Keeping the canonical layer in ``shared.python.launch_monitor`` rather than in
``rate_of_closure`` is what holds those two definitions apart. Do not add a
convenience re-export of either into the other package.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

__all__ = ["DispersionResult", "analyze_dispersion"]


@dataclass(frozen=True)
class DispersionResult:
    """Robust center, covariance ellipse, and radial-error summary."""

    sample_count: int
    center_forward: float
    center_lateral: float
    mean_forward: float
    mean_lateral: float
    ellipse_major: float
    ellipse_minor: float
    ellipse_angle_rad: float
    area_95: float
    radial_rmse: float
    radial_p50: float
    radial_p90: float


def analyze_dispersion(
    frame: pd.DataFrame,
    *,
    forward: str = "carry_distance",
    lateral: str = "lateral_carry",
) -> DispersionResult:
    """Compute a 95% covariance ellipse and robust dispersion metrics."""
    missing = {forward, lateral} - set(frame.columns)
    if missing:
        raise ValueError(f"Dispersion columns not present: {sorted(missing)}")
    values = frame[[forward, lateral]].apply(pd.to_numeric, errors="coerce").dropna()
    if len(values) < 3:
        raise ValueError("At least three complete shots are required for dispersion")
    points = values.to_numpy(float)
    robust_center = np.median(points, axis=0)
    mean_center = np.mean(points, axis=0)
    covariance = np.cov(points, rowvar=False, ddof=1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    eigenvectors = eigenvectors[:, order]
    chi_square_95 = 5.991464547
    radii = np.sqrt(eigenvalues * chi_square_95)
    major, minor = 2 * radii
    vector = eigenvectors[:, 0]
    angle = float(np.arctan2(vector[1], vector[0]))
    delta = points - robust_center
    radial = np.hypot(delta[:, 0], delta[:, 1])
    return DispersionResult(
        len(points),
        float(robust_center[0]),
        float(robust_center[1]),
        float(mean_center[0]),
        float(mean_center[1]),
        float(major),
        float(minor),
        angle,
        float(np.pi * radii[0] * radii[1]),
        float(np.sqrt(np.mean(radial**2))),
        float(np.quantile(radial, 0.5)),
        float(np.quantile(radial, 0.9)),
    )
