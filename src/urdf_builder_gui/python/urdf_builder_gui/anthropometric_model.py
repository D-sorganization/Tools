"""Anthropometric body model — shared constants for URDF generation.

Provides scientifically-backed body proportion ratios based on
de Leva (1996) anthropometric data. A single source of truth for
all consumer modules (GUI, generator, tests).

References:
    de Leva, P. (1996). Adjustments to Zatsiorsky-Seluyanov's
    segment inertia parameters. J Biomech, 29(9), 1223-1230.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from urdf_builder_gui.contracts import require

logger = logging.getLogger(__name__)

# ── Height-to-segment-length ratios (fraction of total standing height) ──

HEIGHT_RATIOS: dict[str, float] = {
    "pelvis": 0.078,
    "torso": 0.278,
    "head": 0.139,
    "thigh": 0.245,
    "shin": 0.246,
    "upper_arm": 0.186,
    "forearm": 0.146,
    "hand": 0.058,
    "foot": 0.039,
}

# ── Mass distribution ratios (fraction of total body mass) ──

MASS_RATIOS: dict[str, float] = {
    "pelvis": 0.112,
    "lumbar": 0.139,
    "thorax": 0.216,
    "torso": 0.355,  # lumbar + thorax combined
    "neck": 0.024,
    "head": 0.069,
    "upper_arm": 0.027,  # per arm
    "forearm": 0.016,  # per arm
    "hand": 0.006,  # per arm
    "thigh": 0.142,  # per leg
    "shin": 0.043,  # per leg
    "foot": 0.014,  # per leg
}

# ── Gender-dependent width scaling factors ──

GENDER_SHOULDER_SCALE = {"min": 0.85, "max": 1.15}  # Female → Male
GENDER_HIP_SCALE = {"min": 1.10, "max": 0.90}  # Female → Male


@dataclass(frozen=True)
class SegmentDimensions:
    """Computed dimensions for a single body segment."""

    name: str
    length: float  # meters
    mass: float  # kg
    width: float  # meters (cross-section)
    depth: float  # meters (cross-section)
    ixx: float  # inertia tensor (kg·m²)
    iyy: float
    izz: float


@dataclass(frozen=True)
class URDFConfig:
    """Configuration for URDF generation.

    Collects all user-settable parameters in one place so that
    the generator never needs to reach into GUI widgets (Law of Demeter).
    """

    robot_name: str = "humanoid"
    height_m: float = 1.75
    mass_kg: float = 70.0
    gender_factor: float = 0.5  # 0 = female, 1 = male
    template: str = "Full Humanoid"
    geometry_type: str = "box"
    collision_geometry: str = "Same as Visual"
    inertia_mode: str = "Primitive"
    damping: float = 0.5
    friction: float = 0.0
    density: float = 1050.0
    proportions: dict[str, float] = field(
        default_factory=lambda: {
            "shoulder_width": 1.0,
            "hip_width": 1.0,
            "arm_length": 1.0,
            "leg_length": 1.0,
            "torso_length": 1.0,
            "head_size": 1.0,
        }
    )


def compute_segment_length(
    total_height: float, segment_key: str, proportion_factor: float = 1.0
) -> float:
    """Compute segment length from total height and proportion factor.

    **Pre-conditions** (DbC):
      - ``total_height`` must be positive.
      - ``segment_key`` must be in HEIGHT_RATIOS.
      - ``proportion_factor`` must be positive.
    """
    require(total_height > 0, "total_height must be positive", total_height)
    require(
        segment_key in HEIGHT_RATIOS,
        f"Unknown segment key: {segment_key}",
        segment_key,
    )
    require(
        proportion_factor > 0,
        "proportion_factor must be positive",
        proportion_factor,
    )
    return total_height * HEIGHT_RATIOS[segment_key] * proportion_factor


def compute_segment_mass(total_mass: float, segment_key: str, count: int = 1) -> float:
    """Compute segment mass from total body mass.

    **Pre-conditions** (DbC):
      - ``total_mass`` must be positive.
      - ``segment_key`` must be in MASS_RATIOS.
      - ``count`` must be >= 1.
    """
    require(total_mass > 0, "total_mass must be positive", total_mass)
    require(
        segment_key in MASS_RATIOS,
        f"Unknown segment key: {segment_key}",
        segment_key,
    )
    require(count >= 1, "count must be >= 1", count)
    return total_mass * MASS_RATIOS[segment_key] * count


def compute_box_inertia(
    mass: float, width: float, height: float, depth: float
) -> tuple[float, float, float]:
    """Compute box inertia tensor (ixx, iyy, izz).

    **Pre-conditions** (DbC):
      - All parameters must be positive.

    Returns:
        Tuple of (ixx, iyy, izz) in kg·m².
    """
    require(mass > 0, "mass must be positive", mass)
    require(width > 0, "width must be positive", width)
    require(height > 0, "height must be positive", height)
    require(depth > 0, "depth must be positive", depth)
    ixx = (1.0 / 12.0) * mass * (height**2 + depth**2)
    iyy = (1.0 / 12.0) * mass * (width**2 + depth**2)
    izz = (1.0 / 12.0) * mass * (width**2 + height**2)
    return ixx, iyy, izz


def compute_cylinder_inertia(
    mass: float, radius: float, length: float
) -> tuple[float, float, float]:
    """Compute cylinder inertia tensor (ixx, iyy, izz).

    z-axis is the cylinder axis.
    """
    require(mass > 0, "mass must be positive", mass)
    require(radius > 0, "radius must be positive", radius)
    require(length > 0, "length must be positive", length)
    ixx = (1.0 / 12.0) * mass * (3 * radius**2 + length**2)
    iyy = ixx
    izz = 0.5 * mass * radius**2
    return ixx, iyy, izz


def compute_sphere_inertia(mass: float, radius: float) -> tuple[float, float, float]:
    """Compute sphere inertia tensor (ixx, iyy, izz)."""
    require(mass > 0, "mass must be positive", mass)
    require(radius > 0, "radius must be positive", radius)
    i = (2.0 / 5.0) * mass * radius**2
    return i, i, i


def interpolate_gender_factor(
    factor: float, female_val: float, male_val: float
) -> float:
    """Linearly interpolate between female and male values.

    ``factor`` = 0 → female, ``factor`` = 1 → male.
    """
    clamped = max(0.0, min(1.0, factor))
    return female_val + clamped * (male_val - female_val)


# ── Template definitions ────────────────────────────────────────────────

TEMPLATE_SEGMENTS: dict[str, list[str]] = {
    "Full Humanoid": [
        "pelvis",
        "torso",
        "head",
        "upper_arm_l",
        "forearm_l",
        "hand_l",
        "upper_arm_r",
        "forearm_r",
        "hand_r",
        "thigh_l",
        "shin_l",
        "foot_l",
        "thigh_r",
        "shin_r",
        "foot_r",
    ],
    "Upper Body Only": [
        "pelvis",
        "torso",
        "head",
        "upper_arm_l",
        "forearm_l",
        "hand_l",
        "upper_arm_r",
        "forearm_r",
        "hand_r",
    ],
    "Lower Body Only": [
        "pelvis",
        "thigh_l",
        "shin_l",
        "foot_l",
        "thigh_r",
        "shin_r",
        "foot_r",
    ],
    "Torso + Arms": [
        "pelvis",
        "torso",
        "upper_arm_l",
        "forearm_l",
        "hand_l",
        "upper_arm_r",
        "forearm_r",
        "hand_r",
    ],
    "Torso + Legs": [
        "pelvis",
        "torso",
        "thigh_l",
        "shin_l",
        "foot_l",
        "thigh_r",
        "shin_r",
        "foot_r",
    ],
    "Custom": ["pelvis", "torso", "head"],
}


def get_template_segments(template_name: str) -> list[str]:
    """Return the list of segment names for a given template.

    **Pre-conditions** (DbC):
      - ``template_name`` must be a known template.
    """
    require(
        template_name in TEMPLATE_SEGMENTS,
        f"Unknown template: {template_name}",
        template_name,
    )
    return TEMPLATE_SEGMENTS[template_name]


__all__ = [
    "HEIGHT_RATIOS",
    "MASS_RATIOS",
    "TEMPLATE_SEGMENTS",
    "SegmentDimensions",
    "URDFConfig",
    "compute_box_inertia",
    "compute_cylinder_inertia",
    "compute_segment_length",
    "compute_segment_mass",
    "compute_sphere_inertia",
    "get_template_segments",
    "interpolate_gender_factor",
]
