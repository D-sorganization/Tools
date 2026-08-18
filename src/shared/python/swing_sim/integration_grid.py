"""Canonical fixed-step integration-grid calculations."""

from __future__ import annotations

from shared.python.contracts import require

from ._numeric_contracts import finite_real

DEFAULT_SWING_RK4_DT_S = 1e-3


def effective_rk4_duration(
    duration_s: object, dt_s: object = DEFAULT_SWING_RK4_DT_S
) -> float:
    """Return the duration represented by the nearest fixed-step RK4 grid."""
    duration = finite_real(duration_s, "duration_s")
    dt = finite_real(dt_s, "dt_s")
    require(duration > 0.0, "duration_s must be > 0", duration_s)
    require(dt > 0.0, "dt_s must be > 0", dt_s)
    require(dt <= duration, "dt_s must not exceed duration_s", (dt_s, duration_s))
    return float(int(round(duration / dt)) * dt)


__all__ = ["DEFAULT_SWING_RK4_DT_S", "effective_rk4_duration"]
