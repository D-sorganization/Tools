"""
Shared physical constants for the pendulum simulator.

Single source of truth for all physical constants used across
the simulator's physics engines, GUI panels, and test fixtures.

Following the DRY principle (Don't Repeat Yourself):
these values were previously defined as magic numbers in 28+
locations across the codebase.

Reference values from NIST:
  https://physics.nist.gov/cgi-bin/cuu/Value?gn
"""

from __future__ import annotations

# ── Gravitational Constants ─────────────────────────────────────────────

#: Standard gravitational acceleration (m/s²).
#: Used as the default ``g`` parameter in physics engines and GUI panels.
#: This is the conventional standard value, not the precise value at any
#: particular latitude.
GRAVITY_MSS: float = 9.81

#: Standard acceleration of gravity (m/s²) — exact SI definition.
#: Used for unit conversions (e.g. kgf → N) where the precise value matters.
try:
    # Prefer the fleet-canonical constant (issue #3994) so this stays in
    # sync with the rest of the fleet instead of drifting independently.
    from shared.python.sidekick.utils.unit_constants import (
        STANDARD_GRAVITY as GRAVITY_STANDARD,
    )
except ImportError:
    # pendulum_simulator ships as a standalone wheel (see Tools CLAUDE.md);
    # shared.python isn't guaranteed to be on the path there. Same value.
    GRAVITY_STANDARD: float = 9.80665  # type: ignore[no-redef]

# ── Conversion Factors ──────────────────────────────────────────────────

#: Newtons per kilogram-force (N/kgf).
NM_PER_KGFM: float = GRAVITY_STANDARD

#: Pounds-force per Newton.
LBF_PER_N: float = 0.224809

#: Inches per meter.
INCHES_PER_M: float = 39.3701

#: Meters per inch.
M_PER_INCH: float = 0.0254
