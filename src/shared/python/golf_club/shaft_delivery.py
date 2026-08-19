"""Shaft forward dynamics → delivered-state deltas (club-tester C2, #4551).

The clubfitting question this module answers: *for the same swing input,
how does the shaft's stiffness distribution change what the head actually
delivers?* The mechanism is the one Milne & Davis (1992) identified as
dominant and MacKenzie & Sprigings (2009) confirmed with full forward
dynamics: near impact the shaft's response is governed by **centrifugal
alignment** — the head's center of gravity, offset from the shaft axis,
is pulled outward by ``F_c = m·ω²·R`` and rotates the compliant tip until
the CG approaches the pull line — plus the **tangential inertial load**
``F_t = m·α·R`` from the grip's angular acceleration or release.

Model ``quasi_static_centrifugal_alignment/1``:

- The CG offset *behind* the shaft axis (``cg_back_m``) makes ``F_c``
  bend the tip forward: **dynamic loft add** and lead deflection.
- The CG offset *toe-ward* (``cg_toe_m``) makes ``F_c`` bend the tip
  down: **toe droop** (delivered-lie change).
- ``F_t`` acting at the toe-ward CG twists the shaft: **face closure**
  when the grip decelerates through release (``α < 0``), face held open
  while still accelerating.
- **Tension stiffening**: the centrifugal pull puts the whole shaft in
  axial tension ``N = F_c`` — several times the cantilever buckling
  scale at driver speeds — which raises the effective bending stiffness
  by the first-order Rayleigh factor ``1 + N/P_cr`` with
  ``P_cr = π²·EI_eff/(4L²)``. Without this term the linear model
  overpredicts droop several-fold against published fitting data.
- **Alignment restoring**: the alignment torque is not constant. With
  the CG an axial distance ``cg_drop_m`` below the tip, a tip rotation
  ``θ`` shrinks the perpendicular offset to ``d − cg_drop·θ``, so the
  equilibrium solves ``k_θ·θ = F_c·(d − cg_drop·θ)`` rather than the
  unbounded ``k_θ·θ = F_c·d``.
- Quasi-static responses are amplified by ``DAF = 1/(1 − β²)`` with
  ``β = f_force / f₁``: a half-sine downswing of duration ``T`` forces at
  ``f_force = 1/(2T)``, and ``f₁`` is the head-loaded first bending
  frequency from a Rayleigh estimate (untensioned tip stiffness from the
  statics compliances — conservative, refusing earlier; effective shaft
  mass ``33/140`` of the flexible span, the cantilever consistent-mass
  fraction). Validity requires ``β ≤ 0.8``; beyond it a quasi-static
  model is not honest and the solver refuses.
- **Kick speed**: the recovering tip returns stored lead deflection at
  first-mode velocity scale, ``v = release_recovery · 2π·f₁ · |δ_lead|``.

Every bending/torsion compliance comes from the *public* statics API
(:func:`solve_cantilever_tip_response`), so this model, the statics
reference, and the modal FE stay mutually consistent — enforced by tests:
the rigid-shaft limit produces exactly zero deltas, the static limit
reproduces the cantilever response bit-for-bit, and the Rayleigh ``f₁``
matches the modal FE's first frequency within 2 % for the closed-form
uniform shaft. Magnitudes for a representative driver land inside
published fitting ranges (≈1–5° dynamic loft add, ≈0.5–4° droop).

Intentionally outside v1 (documented extension points, not omissions):
shear deformation, large-deflection geometry, spine/FLO asymmetry beyond
the per-axis EI stations, grip-side compliance, and calibrated composite
damping. The modal FE (:mod:`.shaft_dynamics`) remains the reference for
fidelity upgrades.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ._validation import require_finite_float
from .shaft_profile import ShaftProfile
from .shaft_statics import ShaftTipLoad, solve_cantilever_tip_response

_MODEL_NAME = "quasi_static_centrifugal_alignment/1"
_MAX_FREQUENCY_RATIO = 0.8
_CANTILEVER_MASS_FRACTION = 33.0 / 140.0
_ASSUMPTIONS = (
    "quasi-static centrifugal alignment with first-mode dynamic amplification",
    "small deflections; compliances from the Euler-Bernoulli statics reference",
    "head treated as a tip point mass with planar CG offsets",
    "half-sine downswing forcing; validity bounded at beta <= 0.8",
    "kick recovery as a release_recovery fraction of first-mode tip velocity",
)

__all__ = [
    "GripKinematics",
    "ShaftDeliveryDeltas",
    "ShaftTipMass",
    "solve_shaft_delivery",
]


@dataclass(frozen=True)
class ShaftTipMass:
    """The head as the shaft tip sees it: point mass with planar CG offsets.

    Attributes:
        mass_kg: Head mass (> 0).
        cg_back_m: CG distance behind the shaft axis, face-normal
            direction (>= 0; drives dynamic loft and lead).
        cg_toe_m: CG distance toe-ward of the shaft axis (>= 0; drives
            droop and, with tangential load, face twist).
        cg_drop_m: CG axial distance below the tip along the shaft axis
            (>= 0; the alignment-restoring lever).
    """

    mass_kg: float
    cg_back_m: float
    cg_toe_m: float
    cg_drop_m: float

    def __post_init__(self) -> None:
        for name in ("mass_kg", "cg_back_m", "cg_toe_m", "cg_drop_m"):
            object.__setattr__(
                self, name, require_finite_float(getattr(self, name), name)
            )
        if self.mass_kg <= 0.0:
            raise ValueError("mass_kg must be > 0")
        if self.cg_back_m < 0.0 or self.cg_toe_m < 0.0 or self.cg_drop_m < 0.0:
            raise ValueError("cg offsets must be >= 0")


@dataclass(frozen=True)
class GripKinematics:
    """Grip-driven swing state near impact, SI.

    Attributes:
        omega_rad_s: Club angular speed about the swing center (>= 0).
        alpha_rad_s2: Club angular acceleration; negative while the grip
            decelerates through release (the usual state at impact).
        swing_radius_m: Swing-center-to-head distance (> 0).
        downswing_duration_s: Downswing time, sets the half-sine forcing
            frequency (> 0).
        release_recovery: Fraction of stored lead deflection returned as
            tip velocity at impact (0..1). 0.5 is a conservative,
            well-timed-swing default.
    """

    omega_rad_s: float
    alpha_rad_s2: float
    swing_radius_m: float
    downswing_duration_s: float = 0.30
    release_recovery: float = 0.5

    def __post_init__(self) -> None:
        for name in (
            "omega_rad_s",
            "alpha_rad_s2",
            "swing_radius_m",
            "downswing_duration_s",
            "release_recovery",
        ):
            object.__setattr__(
                self, name, require_finite_float(getattr(self, name), name)
            )
        if self.omega_rad_s < 0.0:
            raise ValueError("omega_rad_s must be >= 0")
        if self.swing_radius_m <= 0.0:
            raise ValueError("swing_radius_m must be > 0")
        if self.downswing_duration_s <= 0.0:
            raise ValueError("downswing_duration_s must be > 0")
        if not 0.0 <= self.release_recovery <= 1.0:
            raise ValueError("release_recovery must lie in [0, 1]")


@dataclass(frozen=True)
class ShaftDeliveryDeltas:
    """Delivered-state changes attributable to shaft compliance.

    Attributes:
        dynamic_loft_add_deg: Face-tilt from lead-plane tip rotation;
            positive = added loft (tip kicked forward).
        face_closure_deg: Torsional face rotation; positive = closed.
        lie_toe_down_deg: Droop-plane tip rotation; positive = toe down
            (delivered lie flatter).
        kick_speed_mps: Tip speed recovered from lead deflection (>= 0).
        lead_deflection_m: Signed lead(+)/lag(−) tip deflection.
        droop_deflection_m: Toe-down tip deflection (>= 0).
        first_mode_hz: Head-loaded first bending frequency (Rayleigh).
        dynamic_amplification: The applied ``1/(1 − β²)`` factor.
        model_name: Versioned model identifier.
        assumptions: The model's stated validity envelope.
    """

    dynamic_loft_add_deg: float
    face_closure_deg: float
    lie_toe_down_deg: float
    kick_speed_mps: float
    lead_deflection_m: float
    droop_deflection_m: float
    first_mode_hz: float
    dynamic_amplification: float
    model_name: str = _MODEL_NAME
    assumptions: tuple[str, ...] = _ASSUMPTIONS


def _flexible_shaft_mass_kg(profile: ShaftProfile) -> float:
    """Trapezoidal mass of the flexible span from the station densities."""
    start = profile.butt_trim_m
    end = profile.raw_length_m - profile.tip_trim_m - profile.insertion_depth_m
    positions = [start]
    positions.extend(
        station.position_m
        for station in profile.stations
        if start < station.position_m < end
    )
    positions.append(end)
    ordered = sorted(positions)
    total = 0.0
    for left, right in zip(ordered, ordered[1:], strict=False):
        rho_left = float(profile.station_at(left).linear_density_kg_m)
        rho_right = float(profile.station_at(right).linear_density_kg_m)
        total += 0.5 * (rho_left + rho_right) * (right - left)
    return total


def solve_shaft_delivery(
    profile: ShaftProfile,
    tip_mass: ShaftTipMass,
    grip: GripKinematics,
) -> ShaftDeliveryDeltas:
    """Solve the quasi-static centrifugal-alignment delivery response.

    Raises:
        TypeError: If any argument has the wrong type.
        ValueError: If the forcing ratio ``β`` exceeds 0.8 — the
            quasi-static model is not valid there and refuses rather
            than extrapolate.
    """
    if not isinstance(profile, ShaftProfile):
        raise TypeError("profile must be ShaftProfile")
    if not isinstance(tip_mass, ShaftTipMass):
        raise TypeError("tip_mass must be ShaftTipMass")
    if not isinstance(grip, GripKinematics):
        raise TypeError("grip must be GripKinematics")

    # Unit-load compliances through the public statics API, x = lead plane.
    unit_force = solve_cantilever_tip_response(profile, ShaftTipLoad(force_x_n=1.0))
    unit_moment = solve_cantilever_tip_response(
        profile, ShaftTipLoad(moment_about_y_nm=1.0)
    )
    unit_twist = solve_cantilever_tip_response(
        profile, ShaftTipLoad(torque_about_shaft_nm=1.0)
    )
    compliance_twist = unit_twist.twist_about_shaft_rad
    flexible_length_m = unit_force.flexible_length_m

    # Head-loaded first mode: Rayleigh with the cantilever mass fraction,
    # untensioned (conservative for the validity bound below).
    tip_stiffness = 1.0 / unit_force.deflection_x_m
    effective_mass = (
        tip_mass.mass_kg + _CANTILEVER_MASS_FRACTION * _flexible_shaft_mass_kg(profile)
    )
    first_mode_hz = math.sqrt(tip_stiffness / effective_mass) / (2.0 * math.pi)

    forcing_hz = 1.0 / (2.0 * grip.downswing_duration_s)
    beta = forcing_hz / first_mode_hz
    if beta > _MAX_FREQUENCY_RATIO:
        raise ValueError(
            "forcing-to-first-mode ratio "
            f"{beta:.3f} exceeds the quasi-static validity bound "
            f"{_MAX_FREQUENCY_RATIO}; this shaft/swing pair needs the modal model"
        )
    amplification = 1.0 / (1.0 - beta * beta)

    centrifugal_n = tip_mass.mass_kg * grip.omega_rad_s**2 * grip.swing_radius_m
    tangential_n = tip_mass.mass_kg * grip.alpha_rad_s2 * grip.swing_radius_m

    # Centrifugal tension stiffening: first-order Rayleigh factor on the
    # bending compliances. EI_eff is recovered from the measured force
    # compliance (delta = L^3/3EI for the uniform reference).
    ei_effective = flexible_length_m**3 / (3.0 * unit_force.deflection_x_m)
    buckling_scale_n = (math.pi**2 * ei_effective) / (4.0 * flexible_length_m**2)
    tension_scale = 1.0 + centrifugal_n / buckling_scale_n
    compliance_delta_force = unit_force.deflection_x_m / tension_scale
    compliance_theta_force = unit_force.rotation_about_y_rad / tension_scale
    compliance_delta_moment = unit_moment.deflection_x_m / tension_scale
    compliance_theta_moment = unit_moment.rotation_about_y_rad / tension_scale

    # Lead plane: release (alpha < 0) kicks the tip forward; the back-offset
    # CG's centrifugal moment tilts the face into added loft, bounded by the
    # alignment-restoring lever (theta shrinks the offset by cg_drop*theta).
    lead_force_n = -tangential_n
    lead_moment_nm = centrifugal_n * tip_mass.cg_back_m
    lead_restoring = (
        1.0
        + amplification * centrifugal_n * tip_mass.cg_drop_m * compliance_theta_moment
    )
    lead_rotation_rad = (
        amplification
        * (
            lead_force_n * compliance_theta_force
            + lead_moment_nm * compliance_theta_moment
        )
        / lead_restoring
    )
    lead_moment_effective = (
        lead_moment_nm - centrifugal_n * tip_mass.cg_drop_m * lead_rotation_rad
    )
    lead_deflection_m = amplification * (
        lead_force_n * compliance_delta_force
        + lead_moment_effective * compliance_delta_moment
    )

    # Droop plane: the toe-ward CG's centrifugal moment bends the tip down,
    # with the same restoring lever.
    droop_moment_nm = centrifugal_n * tip_mass.cg_toe_m
    droop_rotation_rad = (
        amplification * droop_moment_nm * compliance_theta_moment / lead_restoring
    )
    droop_moment_effective = (
        droop_moment_nm - centrifugal_n * tip_mass.cg_drop_m * droop_rotation_rad
    )
    droop_deflection_m = (
        amplification * droop_moment_effective * compliance_delta_moment
    )

    # Torsion: tangential load at the toe-ward CG; deceleration closes.
    twist_torque_nm = tangential_n * tip_mass.cg_toe_m
    face_closure_rad = amplification * (-twist_torque_nm) * compliance_twist

    kick_speed_mps = (
        grip.release_recovery * 2.0 * math.pi * first_mode_hz * abs(lead_deflection_m)
    )

    return ShaftDeliveryDeltas(
        dynamic_loft_add_deg=math.degrees(lead_rotation_rad),
        face_closure_deg=math.degrees(face_closure_rad),
        lie_toe_down_deg=math.degrees(droop_rotation_rad),
        kick_speed_mps=kick_speed_mps,
        lead_deflection_m=lead_deflection_m,
        droop_deflection_m=droop_deflection_m,
        first_mode_hz=first_mode_hz,
        dynamic_amplification=amplification,
    )
