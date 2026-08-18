"""Tuning parameters for the impact-parameter solver (epic #4103, #4109).

Scaffolding modeled on UpstreamDrift's
``movement_optimizer/trajectory/tuning.py`` (named, documented constants
instead of magic numbers inside the driver), with golf-impact semantics.

All values below are **tuning parameters**: the residual scales normalise
heterogeneous goal units (degrees, mph, RPM, metres) so a default-weighted
goal contributes comparably to the least-squares cost per "one perceptible
unit" of error; the multi-start counts balance robustness against runtime.
They are not derived from first principles.

Rationale:

- Residual scales: 1 unit of each scale is roughly the smallest difference
  a launch monitor reports (0.5 deg, 1 mph, 100 RPM, 1 m of carry), so the
  optimizer trades goals off at launch-monitor resolution by default.
- ``DEFAULT_N_STARTS``: the delivery->impact map is smooth and mostly
  monotone in each variable, so a handful of Latin-hypercube starts is
  enough to escape the rare fold (e.g. loft/speed trade-offs).
- ``DEFAULT_MAX_NFEV_PER_START``: trf on <= ~12 variables converges in a
  few dozen evaluations; the cap only guards against pathological goals.
"""

from __future__ import annotations

# -- Unit conversions (exact definitions) ----------------------------------
MPH_TO_MPS: float = 0.44704
"""Miles per hour to metres per second (exact, 1 mph = 0.44704 m/s)."""

# -- Residual scales per goal quantity (see module docstring) --------------
SCALE_ANGLE_DEG: float = 0.5
"""Residual scale for angular goals [deg]: path/face/AoA/loft/launch/azimuth."""

SCALE_BALL_SPEED_MPH: float = 1.0
"""Residual scale for ball-speed goals [mph]."""

SCALE_SPIN_RPM: float = 100.0
"""Residual scale for total-spin goals [RPM]."""

SCALE_SPIN_AXIS_DEG: float = 1.0
"""Residual scale for spin-axis tilt goals [deg]."""

SCALE_CARRY_M: float = 1.0
"""Residual scale for carry-distance goals [m]."""

# -- Multi-start driver -----------------------------------------------------
DEFAULT_N_STARTS: int = 6
"""Default number of multi-start seeds (start 0 = midpoint/x0 baseline)."""

DEFAULT_MAX_NFEV_PER_START: int = 200
"""Default cap on residual evaluations per ``scipy`` least-squares start."""

DEFAULT_XTOL: float = 1e-10
"""``scipy.optimize.least_squares`` step tolerance."""

DEFAULT_FTOL: float = 1e-10
"""``scipy.optimize.least_squares`` cost-reduction tolerance."""

DEFAULT_GTOL: float = 1e-10
"""``scipy.optimize.least_squares`` gradient tolerance."""

PROGRESS_EMIT_EVERY: int = 20
"""Emit a ProgressReport to the callback every N recorded evaluations."""

STALL_WINDOW: int = 80
"""Evaluations examined by the stall heuristic (movement_optimizer value)."""

STALL_THRESHOLD: float = 1e-4
"""Relative cost change below which the stall heuristic fires."""

# -- Swing-source evaluation (pendulum candidates) --------------------------
DEFAULT_SWING_DURATION_S: float = 1.0
"""Integrated swing duration [s] per pendulum candidate evaluation."""

DEFAULT_SWING_DT_S: float = 2e-3
"""RK4 step [s] for pendulum candidate evaluations (pure-Python path)."""

SWING_TIME_SEARCH_SAMPLES: int = 64
"""Grid samples used to locate the peak-clubhead-speed nominal impact time."""

MIN_CLUBHEAD_SPEED_MPS: float = 1e-3
"""Floor applied to swing-derived clubhead speed [m/s] (delivery requires > 0)."""

# -- Flight evaluation (carry goals) ----------------------------------------
DEFAULT_FLIGHT_MODEL: str = "waterloo_penner"
"""Registry flight model used when carry goals are present."""

DEFAULT_FLIGHT_MAX_TIME_S: float = 12.0
"""Maximum simulated flight time [s]."""

DEFAULT_FLIGHT_DT_S: float = 0.02
"""Flight trajectory sampling interval [s]."""
