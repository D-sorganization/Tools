"""Shared constants for the electrode advisor.

Centralizes magic numbers that were previously hard-coded throughout the
codebase. Issues #1429-#1434.
"""

from __future__ import annotations

# -- Electrode geometry --
ELECTRODE_COUNT: int = 3
ELECTRODE_ANGLES_DEG: list[int] = [0, 120, 240]
SHELL_THICKNESS: float = 0.5

# -- Mesh resolution --
SPHERE_U_RESOLUTION: int = 20
SPHERE_V_RESOLUTION: int = 15
CYLINDER_THETA_SEGMENTS: int = 20
CYLINDER_LENGTH_SEGMENTS: int = 30
CYLINDER_CIRCUM_SEGMENTS: int = 16

# -- Default electrode colors --
ELECTRODE_COLORS: list[str] = ["silver", "#C0C0C0", "#E5E5E5"]
