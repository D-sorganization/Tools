"""Impact model tests: port-parity pins, COR, spin cap, MOI effective mass.

Includes the regression tests for both defects fixed relative to the
UpstreamDrift source (recon #4104): (a) off-center hits must see the MOI
effective-mass reduction; (b) opt-in full 3-D inertia tensor treatment.
"""

from __future__ import annotations

import numpy as np
import pytest

from shared.python.swing_sim.impact.constants import (
    GOLF_BALL_MASS_KG,
    GOLF_BALL_RADIUS_M,
)
from shared.python.swing_sim.impact.models import (
    SPHERE_ROLLING_CAP_FACTOR,
    FiniteTimeImpactModel,
    RigidBodyImpactModel,
    SpringDamperImpactModel,
    create_impact_model,
    face_basis,
    offset_to_face_vector,
)
from shared.python.swing_sim.impact.types import (
    ImpactModelType,
    ImpactParameters,
    PreImpactState,
)


def _pre_state(
    club_speed: float = 50.0,
    normal: np.ndarray | None = None,
    impact_offset: np.ndarray | None = None,
    clubhead_mass: float = 0.200,
    clubhead_moi: float = 4.5e-4,
    clubhead_moi_tensor: np.ndarray | None = None,
) -> PreImpactState:
    n = normal if normal is not None else np.array([1.0, 0.0, 0.0])
    return PreImpactState(
        clubhead_velocity=club_speed * np.array([1.0, 0.0, 0.0]),
        clubhead_angular_velocity=np.zeros(3),
        clubhead_orientation=n,
        ball_position=np.zeros(3),
        ball_velocity=np.zeros(3),
        ball_angular_velocity=np.zeros(3),
        clubhead_mass=clubhead_mass,
        clubhead_moi=clubhead_moi,
        impact_offset=impact_offset,
        clubhead_moi_tensor=clubhead_moi_tensor,
    )


@pytest.mark.unit
@pytest.mark.physics
class TestRigidBodyPins:
    """Port-parity pins against hand-computed impulse cases."""

    def test_center_strike_hand_computed_ball_speed(self) -> None:
        """j = (1+e) mu v; v_ball = j / m_ball, for a square center hit."""
        params = ImpactParameters(cor=0.83)
        m_club, v_club = 0.200, 50.0
        pre = _pre_state(club_speed=v_club, clubhead_mass=m_club)
        post = RigidBodyImpactModel().solve(pre, params)

        mu = (GOLF_BALL_MASS_KG * m_club) / (GOLF_BALL_MASS_KG + m_club)
        j = (1 + 0.83) * mu * v_club
        assert post.ball_velocity[0] == pytest.approx(j / GOLF_BALL_MASS_KG)
        assert post.ball_velocity[1] == pytest.approx(0.0)
        assert post.ball_velocity[2] == pytest.approx(0.0)
        # Club slows by j / m_club along the normal.
        assert post.clubhead_velocity[0] == pytest.approx(v_club - j / m_club)

    def test_center_strike_smash_factor_plausible(self) -> None:
        """Driver-like smash factor ~1.4-1.5 for a square center strike."""
        pre = _pre_state(club_speed=50.0)
        post = RigidBodyImpactModel().solve(pre, ImpactParameters(cor=0.83))
        smash = float(np.linalg.norm(post.ball_velocity)) / 50.0
        assert 1.35 < smash < 1.55

    def test_cor_monotonicity_and_bounds(self) -> None:
        """Higher COR launches the ball faster; e=0 vs e=1 pin."""
        speeds = []
        for cor in (0.0, 0.5, 0.83, 1.0):
            post = RigidBodyImpactModel().solve(_pre_state(), ImpactParameters(cor=cor))
            speeds.append(float(np.linalg.norm(post.ball_velocity)))
        assert speeds == sorted(speeds)
        # e=1 transfers exactly twice the impulse of e=0.
        assert speeds[-1] == pytest.approx(2.0 * speeds[0])

    def test_cor_out_of_range_rejected(self) -> None:
        with pytest.raises(Exception, match="restitution"):
            RigidBodyImpactModel().solve(_pre_state(), ImpactParameters(cor=1.5))


@pytest.mark.unit
@pytest.mark.physics
class TestFrictionSpin:
    def test_oblique_impact_spin_axis_and_cap(self) -> None:
        """Lofted (oblique) strike: backspin about +z, capped at 2/7 limit."""
        loft = np.radians(10.5)
        n = np.array([np.cos(loft), np.sin(loft), 0.0])
        pre = _pre_state(normal=n)
        params = ImpactParameters(cor=0.83, friction_coefficient=10.0)
        post = RigidBodyImpactModel().solve(pre, params)

        # AffineDrift frame: backspin axis is +z for a target-bound shot.
        assert post.ball_angular_velocity[2] > 0.0

        # With absurd friction the cap must bind: J_f = m v_t (2/7).
        v_rel = pre.clubhead_velocity
        v_t = float(np.linalg.norm(v_rel - np.dot(v_rel, n) * n))
        i_ball = (2.0 / 5.0) * GOLF_BALL_MASS_KG * GOLF_BALL_RADIUS_M**2
        cap_spin = (
            (GOLF_BALL_MASS_KG * v_t * SPHERE_ROLLING_CAP_FACTOR)
            * GOLF_BALL_RADIUS_M
            / i_ball
        )
        assert float(np.linalg.norm(post.ball_angular_velocity)) == pytest.approx(
            cap_spin, rel=1e-9
        )

    def test_square_strike_produces_no_spin(self) -> None:
        post = RigidBodyImpactModel().solve(_pre_state(), ImpactParameters())
        assert np.allclose(post.ball_angular_velocity, 0.0)

    def test_rolling_cap_factor_is_two_sevenths(self) -> None:
        assert SPHERE_ROLLING_CAP_FACTOR == pytest.approx(2.0 / 7.0)


@pytest.mark.unit
@pytest.mark.physics
class TestEffectiveMass:
    """Regression tests for defect (a) and the 3-D tensor opt-in (b)."""

    def test_off_center_ball_speed_below_center(self) -> None:
        """Regression (a): off-center hits must launch slower than center."""
        params = ImpactParameters(cor=0.83)
        model = RigidBodyImpactModel()
        center = model.solve(_pre_state(), params)
        toe = model.solve(_pre_state(impact_offset=np.array([0.02, 0.0])), params)
        v_center = float(np.linalg.norm(center.ball_velocity))
        v_toe = float(np.linalg.norm(toe.ball_velocity))
        assert v_toe < v_center

    def test_off_center_effective_mass_hand_computed(self) -> None:
        """m_eff = 1/(1/m + r^2/I) drives the reduced impulse."""
        params = ImpactParameters(cor=0.83)
        m_club, moi, r = 0.200, 4.5e-4, 0.02
        pre = _pre_state(
            impact_offset=np.array([r, 0.0]),
            clubhead_mass=m_club,
            clubhead_moi=moi,
        )
        post = RigidBodyImpactModel().solve(pre, params)
        m_eff_club = 1.0 / (1.0 / m_club + r**2 / moi)
        mu = (GOLF_BALL_MASS_KG * m_eff_club) / (GOLF_BALL_MASS_KG + m_eff_club)
        j = (1 + 0.83) * mu * 50.0
        assert post.ball_velocity[0] == pytest.approx(j / GOLF_BALL_MASS_KG)

    def test_diagonal_tensor_reproduces_scalar(self) -> None:
        """Regression (b): I*eye(3) tensor must match the scalar path."""
        params = ImpactParameters(cor=0.83)
        moi = 4.5e-4
        offset = np.array([0.015, 0.01])
        scalar_post = RigidBodyImpactModel().solve(
            _pre_state(impact_offset=offset, clubhead_moi=moi), params
        )
        tensor_post = RigidBodyImpactModel().solve(
            _pre_state(
                impact_offset=offset,
                clubhead_moi=0.0,  # must be ignored when tensor present
                clubhead_moi_tensor=moi * np.eye(3),
            ),
            params,
        )
        np.testing.assert_allclose(
            tensor_post.ball_velocity, scalar_post.ball_velocity, rtol=1e-12
        )

    def test_anisotropic_tensor_differs_from_scalar(self) -> None:
        """A non-spherical tensor must change the off-center result."""
        params = ImpactParameters(cor=0.83)
        moi = 4.5e-4
        offset = np.array([0.02, 0.0])
        scalar_post = RigidBodyImpactModel().solve(
            _pre_state(impact_offset=offset, clubhead_moi=moi), params
        )
        tensor = np.diag([moi, 5.0 * moi, moi])  # stiffer about the y axis
        tensor_post = RigidBodyImpactModel().solve(
            _pre_state(impact_offset=offset, clubhead_moi_tensor=tensor), params
        )
        # Toe offset twists about y: larger I_yy -> less mass loss -> faster.
        assert float(np.linalg.norm(tensor_post.ball_velocity)) > float(
            np.linalg.norm(scalar_post.ball_velocity)
        )

    def test_bad_tensor_shape_rejected(self) -> None:
        pre = _pre_state(
            impact_offset=np.array([0.02, 0.0]),
            clubhead_moi_tensor=np.eye(2),
        )
        with pytest.raises(Exception, match="3x3"):
            RigidBodyImpactModel().solve(pre, ImpactParameters())


@pytest.mark.unit
class TestFaceBasis:
    def test_target_facing_normal_gives_rh_toe_axis(self) -> None:
        """AffineDrift n=+x: toe axis is +z (right), up axis is +y."""
        toe, up = face_basis(np.array([1.0, 0.0, 0.0]))
        np.testing.assert_allclose(toe, [0.0, 0.0, 1.0], atol=1e-12)
        np.testing.assert_allclose(up, [0.0, 1.0, 0.0], atol=1e-12)

    def test_offset_lift_is_in_face_plane(self) -> None:
        loft = np.radians(30.0)
        n = np.array([np.cos(loft), np.sin(loft), 0.0])
        r = offset_to_face_vector(np.array([0.01, 0.02]), n)
        assert abs(float(np.dot(r, n))) < 1e-12
        assert float(np.linalg.norm(r)) == pytest.approx(
            np.hypot(0.01, 0.02), rel=1e-12
        )


@pytest.mark.unit
@pytest.mark.physics
class TestOtherModels:
    def test_spring_damper_transfers_momentum(self) -> None:
        model = SpringDamperImpactModel(dt=1e-6)
        post = model.solve(_pre_state(), ImpactParameters())
        assert post.ball_velocity[0] > 40.0
        assert post.contact_duration > 0.0
        assert post.clubhead_velocity[0] < 50.0

    def test_finite_time_reports_configured_duration(self) -> None:
        params = ImpactParameters(contact_duration=4.2e-4)
        post = FiniteTimeImpactModel().solve(_pre_state(), params)
        assert post.contact_duration == pytest.approx(4.2e-4)
        rigid = RigidBodyImpactModel().solve(_pre_state(), params)
        np.testing.assert_allclose(post.ball_velocity, rigid.ball_velocity)

    def test_factory_covers_all_types(self) -> None:
        assert isinstance(
            create_impact_model(ImpactModelType.RIGID_BODY), RigidBodyImpactModel
        )
        assert isinstance(
            create_impact_model(ImpactModelType.SPRING_DAMPER),
            SpringDamperImpactModel,
        )
        assert isinstance(
            create_impact_model(ImpactModelType.FINITE_TIME), FiniteTimeImpactModel
        )
