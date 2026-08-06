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

from .constants import DRIVER_MASS_KG, DRIVER_MOI_KG_M2
from .dplane import DPlaneAnalysis, analyze_dplane
from .types import PreImpactState

_MM_TO_M = 1e-3
_MAX_ANGLE_DEG = 89.0


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
