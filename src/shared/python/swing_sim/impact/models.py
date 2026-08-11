"""Impact models (rigid-body COR impulse, spring-damper, finite-time).

Ported from UpstreamDrift ``src/shared/python/physics/impact_model/models.py``
(epic #4103 / issue #4106), rewritten self-contained against the vendored
:mod:`.constants` and the Tools shared :mod:`shared.python.contracts`.

Changes from the UpstreamDrift source:

- Full 3-D inertia treatment (opt-in): when ``PreImpactState`` carries a
  3x3 ``clubhead_moi_tensor``, the effective club mass for an off-center
  hit is computed from the exact rigid-body impulse denominator

      1 / m_eff = 1/m_club + (r x n)^T I^-1 (r x n)

  where ``r`` is the CG-to-contact vector and ``n`` the face normal. This
  is the club-side term of
  ``J = (1/m_ball + 1/m_club + n . ((I^-1 (r x n)) x r))^-1 (1+e) v_rel``
  (the triple product identity gives
  ``n . ((I^-1 (r x n)) x r) = (r x n)^T I^-1 (r x n)``).
  The scalar-MOI fallback ``1/m + |r|^2 / I`` is preserved and is exactly
  reproduced by a diagonal tensor ``I * eye(3)`` because ``r`` lies in the
  face plane (``r`` perpendicular to ``n`` implies ``|r x n| = |r|``).
  Any CG-depth component of ``r`` along ``n`` drops out of ``r x n`` and
  therefore never affects the normal-impulse effective mass.
- Friction-spin axis sign fix (found during the port): the source used
  ``n x t`` which spins a lofted strike toward topspin and contradicts
  its own pre-existing-spin slip reduction; the physical torque axis is
  ``t x n`` (see the inline derivation in ``_compute_friction_spin``).
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from shared.python.contracts import precondition, require, require_finite

from .constants import (
    GOLF_BALL_MASS_KG,
    GOLF_BALL_MOMENT_OF_INERTIA_KG_M2,
    GOLF_BALL_RADIUS_M,
)
from .types import ImpactModelType, ImpactParameters, PostImpactState, PreImpactState

# Rolling-without-slip tangential-impulse factor for a uniform solid sphere.
#
# Derivation (UpstreamDrift #7054). A tangential friction impulse ``J_f``
# applied at the ball surface (lever arm = radius R) changes the
# contact-point tangential velocity by both a linear and an angular
# contribution:
#   * CoM tangential velocity change:        dV     = J_f / m
#   * surface speed from spin-up:            R*dOmega = J_f*R^2 / I
# Rolling without slip is reached when the contact point stops sliding,
# i.e. when dV + R*dOmega equals the effective tangential approach speed:
#   J_f*(1/m + R^2/I) = v_t   =>   J_f = v_t / (1/m + R^2/I).
# For a uniform solid sphere I = (2/5) m R^2, so R^2/I = 5/(2m) and
#   J_f = v_t / (1/m + 5/(2m)) = m*v_t * (2/7).
# Ref: Cross, "Grip-slip behavior of a bouncing ball",
# Am. J. Phys. 70, 1093 (2002).
SPHERE_ROLLING_CAP_FACTOR = 2.0 / 7.0

FloatArray: TypeAlias = NDArray[np.float64]

_DEFAULT_REFERENCE_UP = np.array([0.0, 1.0, 0.0])
_FALLBACK_REFERENCE_UP = np.array([0.0, 0.0, 1.0])


def _norm(vector: FloatArray) -> float:
    """Euclidean norm of a small vector."""
    flat: FloatArray = np.asarray(vector, dtype=np.float64).reshape(-1)
    return 0.0 if flat.size == 0 else float(math.sqrt(float(np.dot(flat, flat))))


def face_basis(
    face_normal: FloatArray,
    reference_up: FloatArray | None = None,
) -> tuple[FloatArray, FloatArray]:
    """Return the (toe_axis, up_axis) face-plane basis for a face normal.

    ``up_axis`` is the projection of ``reference_up`` (default world up,
    AffineDrift +y) onto the face plane; ``toe_axis = n x up_axis`` so that
    for a target-facing normal ``n = +x`` in the AffineDrift frame
    (x target, y up, z right) the toe axis is ``+z`` — the toe of a
    right-handed player's club points right of the target line.

    Args:
        face_normal: Face normal vector (3,), need not be unit length.
        reference_up: World-up hint (3,); defaults to [0, 1, 0]. If the
            normal is (near-)parallel to it, [0, 0, 1] is used instead.

    Returns:
        Tuple ``(toe_axis, up_axis)`` of orthonormal vectors in the face
        plane (both perpendicular to the normal).
    """
    n = np.asarray(face_normal, dtype=float)
    n_mag = _norm(n)
    require(n_mag > 1e-10, "face_normal must be non-zero")
    n = n / n_mag
    up = (
        np.asarray(reference_up, dtype=float)
        if reference_up is not None
        else _DEFAULT_REFERENCE_UP
    )
    up_axis = up - float(np.dot(up, n)) * n
    if _norm(up_axis) < 1e-8:
        up_axis = _FALLBACK_REFERENCE_UP - float(np.dot(_FALLBACK_REFERENCE_UP, n)) * n
    up_axis = up_axis / _norm(up_axis)
    toe_axis = np.cross(n, up_axis)
    return toe_axis, up_axis


def offset_to_face_vector(
    impact_offset: FloatArray,
    face_normal: FloatArray,
    reference_up: FloatArray | None = None,
) -> FloatArray:
    """Lift a 2-D face offset [horizontal, vertical] to a 3-D vector.

    The result lies in the face plane:
    ``r = horizontal * toe_axis + vertical * up_axis``.

    Args:
        impact_offset: (2,) offset [m] [horizontal (+ toe), vertical (+ high)]
        face_normal: Face normal (3,)
        reference_up: Optional world-up hint for the face basis.

    Returns:
        (3,) CG-to-contact offset within the face plane [m].
    """
    offset = np.asarray(impact_offset, dtype=float).reshape(-1)
    require(offset.size == 2, "impact_offset must have exactly 2 components")
    toe_axis, up_axis = face_basis(face_normal, reference_up)
    lifted: FloatArray = offset[0] * toe_axis + offset[1] * up_axis
    return lifted


class ImpactModel(ABC):
    """Abstract base class for impact models."""

    @abstractmethod
    def solve(
        self,
        pre_state: PreImpactState,
        params: ImpactParameters,
    ) -> PostImpactState:
        """Solve the impact and return post-impact state."""
        ...


class RigidBodyImpactModel(ImpactModel):
    """Rigid body collision with coefficient of restitution.

    Uses instantaneous impulse-momentum equations with COR to compute
    post-impact velocities. Off-center hits reduce the effective club mass
    via the clubhead MOI (scalar or full 3x3 tensor, see module docstring).
    """

    def _compute_effective_club_mass(self, pre_state: PreImpactState) -> float:
        """Effective club mass along the contact normal.

        Scalar path: ``1 / (1/m + |r|^2 / I)`` with ``|r|`` the in-plane
        offset magnitude. Tensor path (opt-in via ``clubhead_moi_tensor``):
        ``1 / (1/m + (r x n)^T I^-1 (r x n))``.
        """
        if pre_state is None:
            raise ValueError("pre_state must be provided")
        m_club = float(pre_state.clubhead_mass)
        if pre_state.impact_offset is None:
            return m_club

        offset = np.asarray(pre_state.impact_offset, dtype=float).reshape(-1)
        r_offset = _norm(offset)
        if r_offset <= 1e-6:
            return m_club

        if pre_state.clubhead_moi_tensor is not None:
            tensor = np.asarray(pre_state.clubhead_moi_tensor, dtype=float)
            require(tensor.shape == (3, 3), "clubhead_moi_tensor must be 3x3")
            n = pre_state.clubhead_orientation / _norm(pre_state.clubhead_orientation)
            r_vec = offset_to_face_vector(offset, n)
            r_cross_n = np.cross(r_vec, n)
            angular_term = float(r_cross_n @ np.linalg.solve(tensor, r_cross_n))
            return 1.0 / (1.0 / m_club + angular_term)

        clubhead_moi = float(pre_state.clubhead_moi)
        if clubhead_moi > 0:
            return 1.0 / (1.0 / m_club + r_offset**2 / clubhead_moi)
        return m_club

    def _compute_impulse(
        self,
        v_rel: FloatArray,
        n: FloatArray,
        m_club_effective: float,
        cor: float,
    ) -> tuple[float, float]:
        """Normal impulse magnitude and approach speed along ``n``."""
        if v_rel is None:
            raise ValueError("v_rel must be provided")
        v_approach = float(np.dot(v_rel, n))
        m_eff = (GOLF_BALL_MASS_KG * m_club_effective) / (
            GOLF_BALL_MASS_KG + m_club_effective
        )
        j = (1 + cor) * m_eff * v_approach
        return j, v_approach

    def _compute_friction_spin(
        self,
        pre_state: PreImpactState,
        v_rel: FloatArray,
        v_approach: float,
        n: FloatArray,
        j: float,
        friction_coefficient: float,
    ) -> FloatArray:
        """Ball spin after the Coulomb friction impulse (2/7 rolling cap)."""
        if pre_state is None:
            raise ValueError("pre_state must be provided")
        base_spin: FloatArray = np.asarray(
            pre_state.ball_angular_velocity, dtype=np.float64
        )
        v_tangent = v_rel - v_approach * n
        tangent_mag = _norm(v_tangent)
        if tangent_mag <= 1e-6:
            return base_spin.copy()

        tangent_dir = v_tangent / tangent_mag
        # Spin axis: friction drags the ball's contact surface along the
        # face's relative tangential motion t; the impulse J_f*t acts at
        # the ball contact point -R*n, so torque = (-R n) x (J_f t) and the
        # axis is t x n. (Sign fix vs the UpstreamDrift source, which used
        # n x t and therefore spun a lofted strike toward TOPSPIN — also
        # inconsistent with its own pre-existing-spin slip reduction
        # below. For a lofted, target-bound strike in the AffineDrift
        # frame this axis is +z: backspin points right of the target.)
        spin_axis = np.cross(tangent_dir, n)
        # Rolling cap relative to contact-point speed (pre-existing spin
        # reduces sliding).
        omega_contact = float(np.dot(base_spin, spin_axis))
        v_t_eff = max(0.0, tangent_mag - omega_contact * GOLF_BALL_RADIUS_M)
        # Coulomb friction impulse, capped at the rolling-without-slip
        # impulse for a uniform solid sphere (J_f = m*v_t*2/7, see
        # SPHERE_ROLLING_CAP_FACTOR derivation above).
        j_friction = min(
            float(friction_coefficient * j),
            GOLF_BALL_MASS_KG * v_t_eff * SPHERE_ROLLING_CAP_FACTOR,
        )
        spin_magnitude = j_friction / (
            GOLF_BALL_MOMENT_OF_INERTIA_KG_M2 / GOLF_BALL_RADIUS_M
        )
        return base_spin + spin_magnitude * spin_axis

    def _compute_energy_transfer(
        self,
        pre_ball_velocity: FloatArray,
        post_ball_velocity: FloatArray,
    ) -> float:
        """Ball kinetic-energy change across the impact [J]."""
        if pre_ball_velocity is None:
            raise ValueError("pre_ball_velocity must be provided")
        ball_mass = float(GOLF_BALL_MASS_KG)
        ke_pre = 0.5 * ball_mass * float(np.dot(pre_ball_velocity, pre_ball_velocity))
        ke_post = (
            0.5 * ball_mass * float(np.dot(post_ball_velocity, post_ball_velocity))
        )
        return ke_post - ke_pre

    @precondition(
        lambda self, pre_state, params: pre_state.clubhead_mass > 0,
        "Clubhead mass must be positive",
    )
    @precondition(
        lambda self, pre_state, params: 0 <= params.cor <= 1,
        "Coefficient of restitution must be between 0 and 1",
    )
    @precondition(
        lambda self, pre_state, params: params.friction_coefficient >= 0,
        "Friction coefficient must be non-negative",
    )
    def solve(
        self,
        pre_state: PreImpactState,
        params: ImpactParameters,
    ) -> PostImpactState:
        """Solve impact using rigid body collision model with MOI."""
        require_finite(pre_state.clubhead_velocity, "clubhead_velocity")
        require_finite(pre_state.ball_velocity, "ball_velocity")
        require_finite(pre_state.clubhead_orientation, "clubhead_orientation")
        n_mag = _norm(pre_state.clubhead_orientation)
        require(n_mag > 1e-10, "clubhead_orientation must be non-zero")

        m_club_effective = self._compute_effective_club_mass(pre_state)
        n = pre_state.clubhead_orientation / n_mag
        v_rel = pre_state.clubhead_velocity - pre_state.ball_velocity

        j, v_approach = self._compute_impulse(v_rel, n, m_club_effective, params.cor)

        v_ball_post = pre_state.ball_velocity + (j / GOLF_BALL_MASS_KG) * n
        v_club_post = pre_state.clubhead_velocity - (j / pre_state.clubhead_mass) * n

        ball_spin = self._compute_friction_spin(
            pre_state, v_rel, v_approach, n, j, params.friction_coefficient
        )
        energy_transfer = self._compute_energy_transfer(
            pre_state.ball_velocity, v_ball_post
        )
        impact_loc = (
            np.asarray(pre_state.impact_offset, dtype=float).copy()
            if pre_state.impact_offset is not None
            else np.zeros(2)
        )

        return PostImpactState(
            ball_velocity=v_ball_post,
            ball_angular_velocity=ball_spin,
            clubhead_velocity=v_club_post,
            clubhead_angular_velocity=pre_state.clubhead_angular_velocity.copy(),
            contact_duration=0.0,
            energy_transfer=energy_transfer,
            impact_location=impact_loc,
        )


class SpringDamperImpactModel(ImpactModel):
    """Spring-damper (Kelvin-Voigt) compliant contact model.

    Uses semi-implicit integration of spring-damper contact to compute
    force and velocity evolution during impact.
    """

    @precondition(lambda self, dt=1e-7: dt > 0, "Time step must be positive")
    def __init__(self, dt: float = 1e-7) -> None:
        """Initialize spring-damper model.

        Args:
            dt: Integration time step [s]. Default: 0.1 us (1e-7 s).
        """
        if dt is None:
            raise ValueError("dt must be provided")
        self.dt = dt

    @precondition(
        lambda self, pre_state, params: pre_state.clubhead_mass > 0,
        "Clubhead mass must be positive",
    )
    @precondition(
        lambda self, pre_state, params: params.contact_stiffness > 0,
        "Contact stiffness must be positive",
    )
    def solve(
        self,
        pre_state: PreImpactState,
        params: ImpactParameters,
    ) -> PostImpactState:
        """Solve impact using spring-damper contact model."""
        if pre_state is None:
            raise ValueError("pre_state must be provided")
        m_ball = GOLF_BALL_MASS_KG
        m_club = pre_state.clubhead_mass
        n = pre_state.clubhead_orientation / _norm(pre_state.clubhead_orientation)

        # Tiny clearance avoids dt-dependent overshoot at contact onset.
        initial_gap = GOLF_BALL_RADIUS_M * 1e-4
        x_ball: np.ndarray = (GOLF_BALL_RADIUS_M + initial_gap) * n
        v_ball: np.ndarray = pre_state.ball_velocity.copy()
        x_club: np.ndarray = np.zeros(3)
        v_club: np.ndarray = pre_state.clubhead_velocity.copy()

        contact_time = 0.0
        max_time = 0.005  # 5 ms max contact time [s]
        max_steps = int(max_time / self.dt)
        max_force = 1e5  # [N] limit to prevent numerical blow-up

        for _ in range(max_steps):
            gap = float(np.dot(x_ball - x_club, n)) - GOLF_BALL_RADIUS_M

            if gap < 0:  # In contact (penetration)
                penetration = -gap
                v_rel_normal = float(np.dot(v_ball - v_club, n))
                f_spring = params.contact_stiffness * penetration
                f_damper = -params.contact_damping * v_rel_normal
                f_magnitude = max(0.0, min(f_spring + f_damper, max_force))
                f_contact = f_magnitude * n

                # Semi-implicit Euler: velocities first, then positions.
                v_ball = v_ball + (f_contact / m_ball) * self.dt
                v_club = v_club - (f_contact / m_club) * self.dt
                x_ball = x_ball + v_ball * self.dt
                x_club = x_club + v_club * self.dt
                contact_time += self.dt
            elif contact_time > 0:
                break  # Was in contact but now separated.
            else:
                # Pre-contact: advance positions only.
                x_ball = x_ball + v_ball * self.dt
                x_club = x_club + v_club * self.dt

        ke_pre = (
            0.5
            * m_ball
            * float(np.dot(pre_state.ball_velocity, pre_state.ball_velocity))
        )
        ke_post = 0.5 * m_ball * float(np.dot(v_ball, v_ball))

        return PostImpactState(
            ball_velocity=v_ball,
            ball_angular_velocity=pre_state.ball_angular_velocity.copy(),
            clubhead_velocity=v_club,
            clubhead_angular_velocity=pre_state.clubhead_angular_velocity.copy(),
            contact_duration=contact_time,
            energy_transfer=ke_post - ke_pre,
            impact_location=np.zeros(2),
        )


class FiniteTimeImpactModel(ImpactModel):
    """Finite-time impulse-momentum model.

    Uses the rigid-body result but reports the specified contact duration.
    """

    def solve(
        self,
        pre_state: PreImpactState,
        params: ImpactParameters,
    ) -> PostImpactState:
        """Solve impact using finite-time model."""
        if pre_state is None:
            raise ValueError("pre_state must be provided")
        result = RigidBodyImpactModel().solve(pre_state, params)
        return PostImpactState(
            ball_velocity=result.ball_velocity,
            ball_angular_velocity=result.ball_angular_velocity,
            clubhead_velocity=result.clubhead_velocity,
            clubhead_angular_velocity=result.clubhead_angular_velocity,
            contact_duration=params.contact_duration,
            energy_transfer=result.energy_transfer,
            impact_location=result.impact_location,
        )


def create_impact_model(model_type: ImpactModelType) -> ImpactModel:
    """Factory function to create an impact model instance."""
    if model_type == ImpactModelType.RIGID_BODY:
        return RigidBodyImpactModel()
    if model_type == ImpactModelType.SPRING_DAMPER:
        # Annotated local: the contract-decorated __init__ obscures the
        # constructor's return type from mypy.
        spring_model: ImpactModel = SpringDamperImpactModel()
        return spring_model
    if model_type == ImpactModelType.FINITE_TIME:
        return FiniteTimeImpactModel()
    raise ValueError(f"Unknown impact model type: {model_type}")
