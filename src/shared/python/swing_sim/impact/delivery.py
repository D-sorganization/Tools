"""Delivery front-end: launch-monitor parameters -> impact-model inputs.

New in Tools (epic #4103 / issue #4106); no upstream equivalent exists in
UpstreamDrift.

Frame and sign conventions (AffineDrift frame)
----------------------------------------------
All vectors are expressed in the AffineDrift frame:

- ``x`` = target line, ``y`` = up, ``z`` = right (right-handed).
- Club path (+) = in-to-out: the clubhead travels right of the target
  line at impact (+z velocity component, right-handed player).
- Face angle (+) = open: the face normal points right of the target
  (+z component).
- Attack angle (+) = hitting up (+y velocity component).
- Dynamic loft (+) tilts the face normal upward (+y component).
- Impact offset: toe (+) and high (+) in millimetres, converted to the
  impact model's ``[horizontal, vertical]`` metre offsets. For a
  right-handed player the toe axis is +z (see
  :func:`.models.face_basis`).

The impact model itself (:mod:`.models`) is frame-agnostic — it consumes
whatever consistent right-handed SI frame the vectors are given in — so
the vectors built here feed it directly with no further mapping. (For
reference, UpstreamDrift's physics stack uses x forward / y left / z up;
that adapter concern lives at the UD boundary, not here.)

Angle composition:

    v_hat = [cos(AoA) cos(path), sin(AoA), cos(AoA) sin(path)]
    n_hat = [cos(loft) cos(face), sin(loft), cos(loft) sin(face)]

Dynamic loft and face angle are launch-monitor MEASUREMENTS of the
delivered normal, so delivered lie is already reflected in them;
``lie_deg`` here is the residual toe-up(+)/toe-down(-) rotation of the
face about its own normal, used only to orient the toe/high offset axes
(``[h, v] = R(lie) [toe, high]``).

Spin loft and D-plane diagnostics
---------------------------------
Spin loft is the 3-D angle between the delivered face normal and the club
path vector: ``spin_loft = arccos(v_hat . n_hat)``. The D-plane is the
plane spanned by ``v_hat`` and ``n_hat``; in the classic approximation
(Jorgensen, *The Physics of Golf*; TrackMan D-plane literature) the ball
launches close to the face normal and spins about the D-plane's normal:

    spin_axis = unit(v_hat x n_hat)

For a square face this axis is horizontal (+z, pure backspin). A
face-minus-path difference tilts it; the reported tilt angle is

    tilt = atan2(-axis_y_component, in-plane horizontal component)

signed so positive tilt = fade/slice-side spin (curves right for a
right-handed player) and negative = draw/hook-side.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from shared.python.contracts import require

from .constants import DRIVER_COR, DRIVER_MASS_KG, DRIVER_MOI_KG_M2
from .dplane import DPlaneAnalysis, analyze_dplane
from .models import RigidBodyImpactModel
from .types import ImpactParameters, PreImpactState

_MM_TO_M = 1e-3
_MAX_ANGLE_DEG = 89.0


@dataclass(frozen=True)
class ToolsDeliveryEstimate:
    """Delivered club and ball flight estimate with documented model basis.

    Carries kinematic D-plane quantities along with physical launch angle and
    spin rate derived from the rigid-body impulse-momentum collision model
    (:class:`~shared.python.swing_sim.impact.models.RigidBodyImpactModel` with
    COR impulse and Coulomb friction with 2/7 sphere rolling cap).

    Attributes:
        club_path_deg: Clubhead travel direction azimuth [deg] (+ = in-to-out)
        attack_angle_deg: Clubhead travel elevation angle [deg] (+ = hitting up)
        face_angle_deg: Delivered face normal azimuth [deg] (+ = open / right)
        dynamic_loft_deg: Delivered face normal elevation [deg]
        face_to_path_deg: Horizontal face-to-path angle [deg] (face minus path)
        spin_loft_3d_deg: 3-D angle between travel vector and face normal [deg]
        planar_spin_loft_deg: Planar spin loft [deg]
        spin_axis_tilt_deg: Signed D-plane spin axis tilt [deg] (+ = fade side)
        dplane_status: Status of the D-plane geometry ('defined', etc.)
        launch_angle_deg: Ball launch elevation angle [deg] relative to horizontal,
            computed via rigid-body impulse-momentum exchange with restitution (COR).
        spin_rate_rpm: Total ball spin magnitude [rpm] (revolutions per minute),
            computed via tangential Coulomb friction impulse capped at the 2/7
            rolling-without-slip limit for a uniform sphere.
        spin_rate_rad_per_s: Total ball spin magnitude [rad/s].
        label: Provenance label identifying model basis.
    """

    club_path_deg: float | None
    attack_angle_deg: float | None
    face_angle_deg: float | None
    dynamic_loft_deg: float
    face_to_path_deg: float | None
    spin_loft_3d_deg: float
    planar_spin_loft_deg: float | None
    spin_axis_tilt_deg: float | None
    dplane_status: str
    launch_angle_deg: float
    spin_rate_rpm: float
    spin_rate_rad_per_s: float = 0.0
    label: str = "model estimate (Tools D-plane, rigid-body COR & 2/7 friction spin)"


@dataclass(frozen=True)
class DeliveryParameters:
    """Launch-monitor-style delivery numbers (degrees, m/s, mm).

    Attributes:
        clubhead_speed_mps: Clubhead speed at impact [m/s] (> 0)
        club_path_deg: Club path [deg]; + = in-to-out
        face_angle_deg: Face angle [deg]; + = open (right of target)
        attack_angle_deg: Attack angle [deg]; + = hitting up
        dynamic_loft_deg: Delivered (dynamic) loft [deg]
        lie_deg: Residual delivered lie rotation about the face normal
            [deg]; + = toe up. Orients the offset axes only (see module
            docstring).
        impact_offset_toe_mm: Impact offset toward the toe [mm]
        impact_offset_high_mm: Impact offset up the face [mm]
    """

    clubhead_speed_mps: float
    club_path_deg: float = 0.0
    face_angle_deg: float = 0.0
    attack_angle_deg: float = 0.0
    dynamic_loft_deg: float = 10.5
    lie_deg: float = 0.0
    impact_offset_toe_mm: float = 0.0
    impact_offset_high_mm: float = 0.0

    def __post_init__(self) -> None:
        require(
            math.isfinite(self.clubhead_speed_mps) and self.clubhead_speed_mps > 0,
            "clubhead_speed_mps must be positive and finite",
            self.clubhead_speed_mps,
        )
        for name in (
            "club_path_deg",
            "face_angle_deg",
            "attack_angle_deg",
            "dynamic_loft_deg",
            "lie_deg",
        ):
            value = getattr(self, name)
            require(
                math.isfinite(value) and abs(value) <= _MAX_ANGLE_DEG,
                f"{name} must be finite and within +/-{_MAX_ANGLE_DEG} deg",
                value,
            )
        for name in ("impact_offset_toe_mm", "impact_offset_high_mm"):
            require(
                math.isfinite(getattr(self, name)),
                f"{name} must be finite",
                getattr(self, name),
            )


@dataclass(frozen=True)
class DeliveryDerived:
    """Impact-model inputs plus D-plane diagnostics (AffineDrift frame).

    Attributes:
        clubhead_velocity: Clubhead velocity vector [m/s] (3,)
        face_normal: Unit delivered face normal (3,)
        impact_offset: (2,) [horizontal, vertical] offset [m] in the
            impact model's convention (lie-rotated toe/high offsets)
        clubhead_angular_velocity: Head angular velocity [rad/s] (3,);
            zeros unless supplied by a richer swing source
        spin_loft_deg: 3-D angle between face normal and club path [deg]
        face_to_path_deg: Face angle minus club path [deg] (horizontal)
        spin_axis: Unit D-plane spin axis (3,); +z = pure backspin
        spin_axis_tilt_deg: Signed spin-axis tilt [deg]; + = fade side
        dplane: Typed, frame-explicit full 3-D geometry.  Its normal is
            unavailable for collinear vectors; the legacy ``spin_axis`` field
            retains the historical pure-backspin fallback for compatibility.
    """

    clubhead_velocity: np.ndarray
    face_normal: np.ndarray
    impact_offset: np.ndarray
    clubhead_angular_velocity: np.ndarray
    spin_loft_deg: float
    face_to_path_deg: float
    spin_axis: np.ndarray
    spin_axis_tilt_deg: float
    dplane: DPlaneAnalysis


def derive_delivery(
    params: DeliveryParameters,
    clubhead_angular_velocity: np.ndarray | None = None,
) -> DeliveryDerived:
    """Convert delivery parameters into impact-model inputs + diagnostics.

    Args:
        params: Launch-monitor-style delivery parameters.
        clubhead_angular_velocity: Optional head angular velocity [rad/s]
            (3,) from a richer swing source; defaults to zeros.

    Returns:
        :class:`DeliveryDerived` in the AffineDrift frame.
    """
    path = math.radians(params.club_path_deg)
    face = math.radians(params.face_angle_deg)
    aoa = math.radians(params.attack_angle_deg)
    loft = math.radians(params.dynamic_loft_deg)
    lie = math.radians(params.lie_deg)

    v_hat = np.array(
        [
            math.cos(aoa) * math.cos(path),
            math.sin(aoa),
            math.cos(aoa) * math.sin(path),
        ]
    )
    n_hat = np.array(
        [
            math.cos(loft) * math.cos(face),
            math.sin(loft),
            math.cos(loft) * math.sin(face),
        ]
    )
    clubhead_velocity = params.clubhead_speed_mps * v_hat

    # Lie rotates the toe/high axes about the face normal (+ = toe up):
    # [h, v] = R(lie) [toe, high].
    toe_m = params.impact_offset_toe_mm * _MM_TO_M
    high_m = params.impact_offset_high_mm * _MM_TO_M
    cos_l, sin_l = math.cos(lie), math.sin(lie)
    impact_offset = np.array(
        [toe_m * cos_l - high_m * sin_l, toe_m * sin_l + high_m * cos_l]
    )

    dplane = analyze_dplane(clubhead_velocity, n_hat)
    assert dplane.spin_loft_3d_deg is not None
    spin_loft_deg = dplane.spin_loft_3d_deg
    if dplane.dplane_normal_unit is None:
        # Compatibility only: the typed D-plane result above explicitly marks
        # this axis unavailable instead of presenting the fallback as geometry.
        spin_axis = np.array([0.0, 0.0, 1.0])
        spin_axis_tilt_deg = 0.0
    else:
        spin_axis = np.asarray(dplane.dplane_normal_unit)
        assert dplane.dplane_tilt_deg is not None
        spin_axis_tilt_deg = dplane.dplane_tilt_deg

    omega = (
        np.asarray(clubhead_angular_velocity, dtype=float)
        if clubhead_angular_velocity is not None
        else np.zeros(3)
    )

    return DeliveryDerived(
        clubhead_velocity=clubhead_velocity,
        face_normal=n_hat,
        impact_offset=impact_offset,
        clubhead_angular_velocity=omega,
        spin_loft_deg=spin_loft_deg,
        face_to_path_deg=params.face_angle_deg - params.club_path_deg,
        spin_axis=spin_axis,
        spin_axis_tilt_deg=spin_axis_tilt_deg,
        dplane=dplane,
    )


def estimate_delivery(params: DeliveryParameters) -> ToolsDeliveryEstimate:
    """Evaluate delivered club parameters into a ToolsDeliveryEstimate.

    Computes 3D D-plane kinematics and simulates instantaneous rigid-body
    COR impulse and friction spin to produce launch angle and spin rate.

    Args:
        params: Launch-monitor delivery parameters (speed, path, face,
            attack angle, dynamic loft, impact offsets).


    Returns:
        :class:`ToolsDeliveryEstimate` carrying angles, launch angle, and spin rate.
    """
    derived = derive_delivery(params)
    dplane = derived.dplane

    pre_state = PreImpactState(
        clubhead_velocity=derived.clubhead_velocity,
        clubhead_angular_velocity=derived.clubhead_angular_velocity,
        clubhead_orientation=derived.face_normal,
        ball_position=np.zeros(3),
        ball_velocity=np.zeros(3),
        ball_angular_velocity=np.zeros(3),
        clubhead_mass=DRIVER_MASS_KG,
        clubhead_loft=math.radians(params.dynamic_loft_deg),
        clubhead_moi=DRIVER_MOI_KG_M2,
        impact_offset=derived.impact_offset,
    )
    model = RigidBodyImpactModel()
    post_state = model.solve(
        pre_state, ImpactParameters(cor=DRIVER_COR, friction_coefficient=0.4)
    )

    v_ball = post_state.ball_velocity
    v_ball_horiz = math.hypot(float(v_ball[0]), float(v_ball[2]))
    launch_angle_deg = (
        math.degrees(math.atan2(float(v_ball[1]), v_ball_horiz))
        if (v_ball_horiz > 1e-12 or abs(float(v_ball[1])) > 1e-12)
        else 0.0
    )
    omega_ball = post_state.ball_angular_velocity
    spin_rate_rad_s = float(np.linalg.norm(omega_ball))
    spin_rate_rpm = spin_rate_rad_s * 60.0 / (2.0 * math.pi)

    return ToolsDeliveryEstimate(
        club_path_deg=dplane.club_path_deg,
        attack_angle_deg=dplane.attack_angle_deg,
        face_angle_deg=dplane.face_angle_deg,
        dynamic_loft_deg=float(dplane.dynamic_loft_deg),
        face_to_path_deg=derived.face_to_path_deg,
        spin_loft_3d_deg=float(derived.spin_loft_deg),
        planar_spin_loft_deg=dplane.planar_spin_loft_deg,
        spin_axis_tilt_deg=derived.spin_axis_tilt_deg,
        dplane_status=str(dplane.status),
        launch_angle_deg=launch_angle_deg,
        spin_rate_rpm=spin_rate_rpm,
        spin_rate_rad_per_s=spin_rate_rad_s,
    )


def to_pre_impact_state(
    params: DeliveryParameters,
    clubhead_mass: float = DRIVER_MASS_KG,
    clubhead_moi: float = DRIVER_MOI_KG_M2,
    clubhead_moi_tensor: np.ndarray | None = None,
    ball_velocity: np.ndarray | None = None,
    ball_angular_velocity: np.ndarray | None = None,
    clubhead_angular_velocity: np.ndarray | None = None,
) -> PreImpactState:
    """Build a :class:`.types.PreImpactState` from delivery parameters.

    Args:
        params: Launch-monitor-style delivery parameters.
        clubhead_mass: Clubhead mass [kg]
        clubhead_moi: Scalar clubhead MOI about CG [kg.m^2]
        clubhead_moi_tensor: Optional 3x3 clubhead MOI tensor (enables the
            full 3-D effective-mass treatment in :mod:`.models`)
        ball_velocity: Ball velocity [m/s] (default: at rest)
        ball_angular_velocity: Ball spin [rad/s] (default: none)
        clubhead_angular_velocity: Head angular velocity [rad/s]
            (default: zeros)

    Returns:
        A pre-impact state carrying the delivered vectors AND the impact
        offset, so off-center hits get the MOI effective-mass reduction.
    """
    derived = derive_delivery(params, clubhead_angular_velocity)
    return PreImpactState(
        clubhead_velocity=derived.clubhead_velocity,
        clubhead_angular_velocity=derived.clubhead_angular_velocity,
        clubhead_orientation=derived.face_normal,
        ball_position=np.zeros(3),
        ball_velocity=(
            np.asarray(ball_velocity, dtype=float)
            if ball_velocity is not None
            else np.zeros(3)
        ),
        ball_angular_velocity=(
            np.asarray(ball_angular_velocity, dtype=float)
            if ball_angular_velocity is not None
            else np.zeros(3)
        ),
        clubhead_mass=clubhead_mass,
        clubhead_loft=math.radians(params.dynamic_loft_deg),
        clubhead_moi=clubhead_moi,
        impact_offset=derived.impact_offset,
        clubhead_moi_tensor=clubhead_moi_tensor,
    )
