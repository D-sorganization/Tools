"""Physics-based gear-effect spin from off-center impact.

New in Tools (epic #4103 / issue #4106). Replaces the empirical
three-constant model from UpstreamDrift
``physics/impact_model/utils.py::compute_gear_effect_spin`` with a
first-principles derivation.

Mechanism
---------
An off-center normal impulse ``J n`` applied at CG-to-contact offset ``r``
exerts a torque impulse ``r x (-J n)`` on the clubhead, so the head picks
up a rotation recoil ``d_omega = I^-1 (r x (-J n))`` during contact
(``I`` scalar or full 3x3 tensor). Because the head CG sits a depth ``d``
BEHIND the face plane, the contact point is ``r = r_plane + d n`` and the
rotating face sweeps the contact point tangentially:

    v_surface = ramp * (d_omega x r)

(``ramp = 1/2``: the recoil builds roughly linearly from zero over the
contact, so the time-averaged surface velocity is half the final value).
The ball "gears" against this moving surface — friction drags the ball
surface with the face, producing spin opposite the head's recoil, via the
same Coulomb friction impulse capped at the rolling-without-slip limit
(``J_f = min(mu J, m_ball v_t 2/7)``, see
:data:`.models.SPHERE_ROLLING_CAP_FACTOR`) used for loft-driven spin.

Qualitative signatures reproduced (right-handed player, AffineDrift frame
x target / y up / z right, toe axis +z):

- Toe hit (+toe): head recoils about -y (toe twists open/back), face
  surface sweeps toe-ward under the ball, friction drags the ball toward
  +z at the back of the ball => +y ball spin = draw-side spin.
- High hit (+high): head recoils about +z (loft increases), face sweeps
  upward under the ball => -z spin component = reduced backspin.

Bulge/roll seam
---------------
Face curvature is club data, not impact physics, so it enters through the
optional ``face_normal_at_offset(toe_m, high_m) -> normal`` callable
(provided by the app's club package): the local normal at the offset
replaces the nominal face normal for the impulse direction, starting a
toe hit further right (open) so the gear-effect draw spin curves it back —
keeping swing_sim app-agnostic.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from shared.python.contracts import require

from .constants import (
    DRIVER_CG_DEPTH_M,
    GOLF_BALL_MASS_KG,
    GOLF_BALL_MOMENT_OF_INERTIA_KG_M2,
    GOLF_BALL_RADIUS_M,
)
from .models import SPHERE_ROLLING_CAP_FACTOR, _norm, offset_to_face_vector

FaceNormalAtOffset = Callable[[float, float], np.ndarray]
"""Callable seam for bulge/roll: ``(toe_m, high_m) -> local face normal``."""

_CONTACT_RAMP_FACTOR = 0.5
"""Time-average factor for the head recoil built up during contact."""


@dataclass(frozen=True)
class GearEffectResult:
    """Result of the physics-based gear-effect computation.

    Attributes:
        ball_spin_delta: Additional ball spin from gear effect [rad/s] (3,)
        head_angular_velocity_delta: Head rotation recoil acquired during
            contact [rad/s] (3,)
        contact_normal: Unit face normal actually used at the contact point
            (local bulge/roll normal when a curvature callable was
            supplied) (3,)
        tangential_surface_speed: Time-averaged tangential face-surface
            speed at the contact point [m/s]
    """

    ball_spin_delta: np.ndarray
    head_angular_velocity_delta: np.ndarray
    contact_normal: np.ndarray
    tangential_surface_speed: float


def resolve_contact_normal(
    impact_offset: np.ndarray,
    face_normal: np.ndarray,
    face_normal_at_offset: FaceNormalAtOffset | None = None,
) -> np.ndarray:
    """Unit contact normal: local bulge/roll normal if supplied, else nominal.

    Args:
        impact_offset: (2,) offset [m] [toe (+), high (+)]
        face_normal: Nominal face normal (3,)
        face_normal_at_offset: Optional club-package callable returning the
            local face normal at ``(toe_m, high_m)`` (bulge/roll curvature).

    Returns:
        Unit normal (3,) used for the impulse direction.
    """
    offset = np.asarray(impact_offset, dtype=float).reshape(-1)
    require(offset.size == 2, "impact_offset must have exactly 2 components")
    if face_normal_at_offset is not None:
        n = np.asarray(face_normal_at_offset(float(offset[0]), float(offset[1])))
    else:
        n = np.asarray(face_normal, dtype=float)
    n_mag = _norm(n)
    require(n_mag > 1e-10, "contact normal must be non-zero")
    unit_normal: np.ndarray = n / n_mag
    return unit_normal


def compute_gear_effect(
    impact_offset: np.ndarray,
    face_normal: np.ndarray,
    normal_impulse: float,
    clubhead_moi: float | np.ndarray,
    cg_depth_m: float = DRIVER_CG_DEPTH_M,
    friction_coefficient: float = 0.4,
    reference_up: np.ndarray | None = None,
    face_normal_at_offset: FaceNormalAtOffset | None = None,
) -> GearEffectResult:
    """Compute gear-effect spin from the head's rotation recoil.

    Args:
        impact_offset: (2,) offset from head CG in the face plane [m]
            [horizontal (+ = toe), vertical (+ = high)]
        face_normal: Nominal clubface normal (3,), need not be unit length
        normal_impulse: Normal impulse magnitude J from the base COR
            solve [N.s] (non-negative)
        clubhead_moi: Scalar clubhead MOI about the CG [kg.m^2], or a full
            3x3 MOI tensor in the same frame as the vectors
        cg_depth_m: CG distance behind the face plane [m] (front-to-back
            lever arm; zero depth means no tangential sweep and no gear
            spin from in-plane offsets)
        friction_coefficient: Ball-face Coulomb friction coefficient
        reference_up: Optional world-up hint for the face basis
        face_normal_at_offset: Optional bulge/roll callable (see module
            docstring)

    Returns:
        :class:`GearEffectResult` with the ball spin delta, head recoil,
        contact normal, and tangential surface speed.
    """
    require(normal_impulse >= 0.0, "normal_impulse must be non-negative")
    require(cg_depth_m >= 0.0, "cg_depth_m must be non-negative")
    require(friction_coefficient >= 0.0, "friction_coefficient non-negative")

    n = resolve_contact_normal(impact_offset, face_normal, face_normal_at_offset)
    offset = np.asarray(impact_offset, dtype=float).reshape(-1)

    # CG-to-contact vector: in-plane offset (built on the NOMINAL normal's
    # face basis so curvature does not skew the geometry) plus CG depth.
    nominal = np.asarray(face_normal, dtype=float)
    nominal = nominal / _norm(nominal)
    r_plane = offset_to_face_vector(offset, nominal, reference_up)
    r_contact = r_plane + cg_depth_m * nominal

    # Head rotation recoil from the torque impulse of the reaction -J n.
    torque_impulse = np.cross(r_contact, -normal_impulse * n)
    if isinstance(clubhead_moi, np.ndarray) and clubhead_moi.shape == (3, 3):
        d_omega = np.linalg.solve(np.asarray(clubhead_moi, dtype=float), torque_impulse)
    else:
        moi_scalar = float(np.asarray(clubhead_moi, dtype=float).reshape(-1)[0])
        require(moi_scalar > 0.0, "clubhead_moi must be positive")
        d_omega = torque_impulse / moi_scalar

    # Time-averaged tangential surface velocity of the face at the contact.
    v_surface = _CONTACT_RAMP_FACTOR * np.cross(d_omega, r_contact)
    v_tangent = v_surface - float(np.dot(v_surface, n)) * n
    v_t = _norm(v_tangent)

    if v_t <= 1e-9 or normal_impulse <= 0.0:
        return GearEffectResult(
            ball_spin_delta=np.zeros(3),
            head_angular_velocity_delta=d_omega,
            contact_normal=n,
            tangential_surface_speed=0.0,
        )

    # Friction drags the ball surface WITH the moving face (the ball gears
    # against the face), capped at the rolling-without-slip limit.
    t_dir = v_tangent / v_t
    j_friction = min(
        friction_coefficient * normal_impulse,
        GOLF_BALL_MASS_KG * v_t * SPHERE_ROLLING_CAP_FACTOR,
    )
    # Friction impulse J_f t_dir acts at the ball contact point -R n:
    # torque = (-R n) x (J_f t_dir) => spin axis = (t_dir x n).
    spin_axis = np.cross(t_dir, n)
    spin_magnitude = j_friction * GOLF_BALL_RADIUS_M / GOLF_BALL_MOMENT_OF_INERTIA_KG_M2

    return GearEffectResult(
        ball_spin_delta=spin_magnitude * spin_axis,
        head_angular_velocity_delta=d_omega,
        contact_normal=n,
        tangential_surface_speed=v_t,
    )
