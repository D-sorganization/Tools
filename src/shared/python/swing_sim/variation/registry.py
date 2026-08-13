"""Namespaced variable registry for the variation engine (#4120 V3).

The single shared vocabulary of perturbable variables: built-in
``swing_sim`` categories (delivery, swing, club, ball setup, launch) plus the
:func:`register_variable` extension seam other packages use to adopt
the same scheme. See :mod:`.spec` for the study value types.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from shared.python.contracts import require

from ..ball_setup import DEFAULT_DRIVER_TEE_HEIGHT_M
from .contextual_registry import LOCALIZED_TORQUE_VARIABLES

MODES: tuple[str, ...] = ("delivery", "swing", "launch")
"""Pipeline slices a plan can exercise (see :class:`VariationPlan`)."""

CATEGORY_DELIVERY = "swing_sim.impact.delivery"
CATEGORY_SWING = "swing_sim.swing"
CATEGORY_CLUB = "swing_sim.club"
CATEGORY_LAUNCH = "swing_sim.flight.launch"
CATEGORY_BALL_SETUP = "swing_sim.ball_setup"

APPLICABILITIES: tuple[str, ...] = (
    "always",
    "tee_only",
    "localized_torque_only",
)


@dataclass(frozen=True)
class VariableDef:
    """One registry entry: a variable other packages can perturb.

    The fields bind a stable key to presentation metadata, a default and
    typical scale, guidance, and any execution-context applicability gate.
    """

    key: str
    label: str
    unit: str
    default: float
    typical_scale: float
    guidance: str
    applicability: str = "always"

    def __post_init__(self) -> None:
        require("." in self.key, "key must be namespaced (category.name)", self.key)
        require(math.isfinite(self.default), "default must be finite", self.default)
        require(
            math.isfinite(self.typical_scale) and self.typical_scale > 0.0,
            "typical_scale must be finite and > 0",
            self.typical_scale,
        )
        require(
            self.applicability in APPLICABILITIES,
            f"applicability must be one of {APPLICABILITIES}",
            self.applicability,
        )

    @property
    def category(self) -> str:
        """The namespace portion of :attr:`key` (everything but the name)."""
        return self.key.rsplit(".", 1)[0]

    @property
    def name(self) -> str:
        """The short name portion of :attr:`key`."""
        return self.key.rsplit(".", 1)[1]


_REGISTRY: dict[str, VariableDef] = {}


def register_variable(definition: VariableDef) -> None:
    """Add a variable to the shared registry (extension seam).

    Raises:
        ContractViolationError: If the key is already registered.
    """
    require(
        definition.key not in _REGISTRY,
        "variable key already registered",
        definition.key,
    )
    _REGISTRY[definition.key] = definition


def variable_registry() -> Mapping[str, VariableDef]:
    """Read-only view of the full registry (insertion-ordered)."""
    return MappingProxyType(_REGISTRY)


def variables_in_category(category: str) -> tuple[VariableDef, ...]:
    """All registry entries whose category equals ``category``."""
    return tuple(d for d in _REGISTRY.values() if d.category == category)


def _register_builtins() -> None:
    """Populate the built-in swing_sim categories (import-time, once)."""
    src_lm = "Source: openly published tour launch-monitor averages."
    src_swing = (
        "Source: 3-D motion-capture swing-plane studies collected in the "
        "AffineDrift dossier."
    )
    src_club = (
        "Source: shared swing_sim impact constants (driver head, USGA "
        "COR limit region)."
    )
    entries: tuple[tuple[str, str, str, float, float, str], ...] = (
        # ── swing_sim.impact.delivery (goals.DELIVERY_VARIABLE_DEFAULTS) ──
        (
            f"{CATEGORY_DELIVERY}.clubhead_speed_mps",
            "Clubhead Speed",
            "m/s",
            45.0,
            0.5,
            f"Typical shot-to-shot variation: 0.3-1 m/s. {src_lm}",
        ),
        (
            f"{CATEGORY_DELIVERY}.club_path_deg",
            "Club Path",
            "deg",
            0.0,
            1.0,
            f"Typical shot-to-shot variation: 0.5-2 deg. {src_lm}",
        ),
        (
            f"{CATEGORY_DELIVERY}.face_angle_deg",
            "Face Angle",
            "deg",
            0.0,
            1.0,
            f"Typical shot-to-shot variation: 0.5-2 deg (the dominant "
            f"start-line input). {src_lm}",
        ),
        (
            f"{CATEGORY_DELIVERY}.attack_angle_deg",
            "Attack Angle",
            "deg",
            0.0,
            0.8,
            f"Typical shot-to-shot variation: 0.5-1.5 deg. {src_lm}",
        ),
        (
            f"{CATEGORY_DELIVERY}.dynamic_loft_deg",
            "Dynamic Loft",
            "deg",
            10.5,
            0.8,
            f"Typical shot-to-shot variation: 0.5-1.5 deg. {src_lm}",
        ),
        (
            f"{CATEGORY_DELIVERY}.lie_deg",
            "Residual Lie Rotation",
            "deg",
            0.0,
            0.5,
            "Typical variation: within 1 deg of square. Source: "
            "AffineDrift launch-monitor frame conventions.",
        ),
        (
            f"{CATEGORY_DELIVERY}.impact_offset_toe_mm",
            "Impact Toward Toe",
            "mm",
            0.0,
            4.0,
            "Typical strike dispersion: 3-8 mm across the face. Source: "
            "published robot-test impact maps.",
        ),
        (
            f"{CATEGORY_DELIVERY}.impact_offset_high_mm",
            "Impact Above Center",
            "mm",
            0.0,
            3.0,
            "Typical strike dispersion: 2-6 mm vertically. Source: "
            "published robot-test impact maps.",
        ),
        # ── swing_sim.swing (goals.SWING_VARIABLE_DEFAULTS) ──────────────
        (
            f"{CATEGORY_SWING}.yaw_deg",
            "Swing-Plane Yaw",
            "deg",
            0.0,
            1.5,
            f"Typical variation: 1-3 deg about vertical. {src_swing}",
        ),
        (
            f"{CATEGORY_SWING}.side_tilt_deg",
            "Swing-Plane Side Tilt",
            "deg",
            -45.0,
            1.5,
            f"Typical variation: 1-3 deg about the plane lean. {src_swing}",
        ),
        (
            f"{CATEGORY_SWING}.forward_tilt_deg",
            "Swing-Plane Forward Tilt",
            "deg",
            0.0,
            1.5,
            f"Typical variation: 1-3 deg toward/away from target. {src_swing}",
        ),
        (
            f"{CATEGORY_SWING}.impact_time_offset_s",
            "Impact-Time Offset",
            "s",
            0.0,
            0.002,
            "Typical timing jitter: 1-5 ms about the peak-speed instant. "
            "Source: pendulum swing-model timing in the shared swing_sim "
            "package.",
        ),
        (
            f"{CATEGORY_SWING}.damping_shoulder",
            "Shoulder Damping",
            "N·m·s",
            0.4,
            0.05,
            "Typical variation: 0.02-0.1 N·m·s about the 0.4 golf "
            "default. Source: double-pendulum golf-swing literature "
            "parameters used by swing_sim.",
        ),
        (
            f"{CATEGORY_SWING}.damping_wrist",
            "Wrist Damping",
            "N·m·s",
            0.25,
            0.05,
            "Typical variation: 0.02-0.1 N·m·s about the 0.25 golf "
            "default. Source: double-pendulum golf-swing literature "
            "parameters used by swing_sim.",
        ),
        # ── swing_sim.club (impact constants) ────────────────────────────
        (
            f"{CATEGORY_CLUB}.head_mass_kg",
            "Clubhead Mass",
            "kg",
            0.200,
            0.002,
            f"Manufacturing tolerance: a few grams about 200 g. {src_club}",
        ),
        (
            f"{CATEGORY_CLUB}.head_moi_kg_m2",
            "Clubhead MOI",
            "kg·m²",
            4.5e-4,
            2.0e-5,
            f"Typical driver MOI spread about 4.5e-4 kg·m². {src_club}",
        ),
        (
            f"{CATEGORY_CLUB}.cor",
            "Coefficient of Restitution",
            "",
            0.83,
            0.005,
            f"Face-to-face COR spread near the 0.83 limit. {src_club}",
        ),
        # ── swing_sim.flight.launch (LaunchConditions front-end) ─────────
        (
            f"{CATEGORY_LAUNCH}.ball_speed_mph",
            "Ball Speed",
            "mph",
            150.0,
            1.0,
            f"Typical shot-to-shot variation: 0.5-2 mph. {src_lm}",
        ),
        (
            f"{CATEGORY_LAUNCH}.launch_angle_deg",
            "Launch Angle",
            "deg",
            12.0,
            0.5,
            f"Typical shot-to-shot variation: 0.3-1 deg. {src_lm}",
        ),
        (
            f"{CATEGORY_LAUNCH}.launch_azimuth_deg",
            "Launch Direction",
            "deg",
            0.0,
            0.8,
            f"Positive = right of the target line. {src_lm}",
        ),
        (
            f"{CATEGORY_LAUNCH}.spin_rpm",
            "Total Spin",
            "rpm",
            2600.0,
            100.0,
            f"Typical shot-to-shot variation: 50-300 rpm. {src_lm}",
        ),
        (
            f"{CATEGORY_LAUNCH}.spin_axis_deg",
            "Spin-Axis Tilt",
            "deg",
            0.0,
            1.5,
            f"Positive = fade/slice side. {src_lm}",
        ),
    )
    for key, label, unit, default, scale, guidance in entries:
        register_variable(
            VariableDef(
                key=key,
                label=label,
                unit=unit,
                default=default,
                typical_scale=scale,
                guidance=guidance,
            )
        )
    for definition in LOCALIZED_TORQUE_VARIABLES:
        register_variable(
            VariableDef(
                key=f"{CATEGORY_SWING}.{definition.name}",
                label=definition.label,
                unit=definition.unit,
                default=definition.default,
                typical_scale=definition.typical_scale,
                guidance=definition.guidance,
                applicability=definition.applicability,
            )
        )
    register_variable(
        VariableDef(
            key=f"{CATEGORY_BALL_SETUP}.tee_height_m",
            label="Tee Height",
            unit="m",
            default=DEFAULT_DRIVER_TEE_HEIGHT_M,
            typical_scale=0.003,
            guidance=(
                "Applicable only when Ball Support is Tee. Height is measured "
                "from the ground plane to the bottom of the ball."
            ),
            applicability="tee_only",
        )
    )


_register_builtins()

#: Registry categories whose variables are legal per pipeline mode.
MODE_CATEGORIES: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "delivery": (CATEGORY_DELIVERY, CATEGORY_CLUB, CATEGORY_BALL_SETUP),
        "swing": (
            CATEGORY_SWING,
            CATEGORY_DELIVERY,
            CATEGORY_CLUB,
            CATEGORY_BALL_SETUP,
        ),
        "launch": (CATEGORY_LAUNCH,),
    }
)

#: Delivery variables that are derived from the swing in ``swing`` mode
#: (same set as ``solver.goals.SWING_DERIVED_VARIABLES``, namespaced).
SWING_DERIVED_KEYS: tuple[str, ...] = (
    f"{CATEGORY_DELIVERY}.clubhead_speed_mps",
    f"{CATEGORY_DELIVERY}.club_path_deg",
    f"{CATEGORY_DELIVERY}.attack_angle_deg",
)


def keys_for_mode(mode: str) -> tuple[str, ...]:
    """Registry keys legal as base/noise variables for a pipeline mode."""
    require(mode in MODES, "unknown mode", mode)
    keys: list[str] = []
    for category in MODE_CATEGORIES[mode]:
        keys.extend(d.key for d in variables_in_category(category))
    if mode == "swing":
        keys = [k for k in keys if k not in SWING_DERIVED_KEYS]
    return tuple(keys)


__all__ = [
    "APPLICABILITIES",
    "CATEGORY_BALL_SETUP",
    "CATEGORY_CLUB",
    "CATEGORY_DELIVERY",
    "CATEGORY_LAUNCH",
    "CATEGORY_SWING",
    "MODES",
    "MODE_CATEGORIES",
    "SWING_DERIVED_KEYS",
    "VariableDef",
    "keys_for_mode",
    "register_variable",
    "variable_registry",
    "variables_in_category",
]
