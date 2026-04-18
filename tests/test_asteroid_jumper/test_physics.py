"""TDD test suite for asteroid_jumper.physics.

Tests cover:
  - Vec2 arithmetic and geometry
  - RigidBody construction and invariants
  - Moment of inertia formulas
  - Impulse computation (Newton's third law, torque)
  - SpringLaunch impulse accumulation
  - SimState integration and momentum conservation
  - off_centre_ratio helper
"""

from __future__ import annotations

import math

import pytest

from asteroid_jumper.physics import (
    RigidBody,
    SimState,
    SpringLaunch,
    Vec2,
    apply_impulse,
    compute_jump_impulse,
    integrate_body,
    moment_of_inertia_disk,
    moment_of_inertia_ellipse,
    moment_of_inertia_rod,
    off_centre_ratio,
    step_simulation,
)

# ---------------------------------------------------------------------------
# Vec2 tests
# ---------------------------------------------------------------------------


class TestVec2:
    def test_default_is_origin(self) -> None:
        v = Vec2()
        assert v.x == 0.0 and v.y == 0.0

    def test_addition(self) -> None:
        a, b = Vec2(1, 2), Vec2(3, 4)
        result = a + b
        assert result == Vec2(4, 6)

    def test_subtraction(self) -> None:
        result = Vec2(5, 3) - Vec2(2, 1)
        assert result == Vec2(3, 2)

    def test_scalar_multiply(self) -> None:
        v = Vec2(2, 3) * 2.0
        assert v == Vec2(4, 6)

    def test_addition_rejects_non_vector(self) -> None:
        with pytest.raises(TypeError, match="Vec2 \\+ Vec2 required"):
            Vec2(1, 2) + object()

    def test_scalar_multiply_rejects_non_numeric(self) -> None:
        with pytest.raises(TypeError, match="Vec2 \\* scalar required"):
            Vec2(1, 2) * object()

    def test_rmul(self) -> None:
        v = 3.0 * Vec2(1, 2)
        assert v == Vec2(3, 6)

    def test_negation(self) -> None:
        assert -Vec2(1, -2) == Vec2(-1, 2)

    def test_dot_product(self) -> None:
        assert Vec2(1, 0).dot(Vec2(0, 1)) == pytest.approx(0.0)
        assert Vec2(1, 0).dot(Vec2(1, 0)) == pytest.approx(1.0)

    def test_cross_product(self) -> None:
        # i × j = k (positive z)
        assert Vec2(1, 0).cross(Vec2(0, 1)) == pytest.approx(1.0)
        # j × i = -k (negative z)
        assert Vec2(0, 1).cross(Vec2(1, 0)) == pytest.approx(-1.0)

    def test_length(self) -> None:
        assert Vec2(3, 4).length() == pytest.approx(5.0)

    def test_normalize(self) -> None:
        n = Vec2(3, 4).normalize()
        assert n.x == pytest.approx(0.6)
        assert n.y == pytest.approx(0.8)
        assert n.length() == pytest.approx(1.0)

    def test_normalize_zero_vector(self) -> None:
        n = Vec2(0, 0).normalize()
        assert n == Vec2(0, 0)

    def test_rotate_90(self) -> None:
        v = Vec2(1, 0).rotate(math.pi / 2)
        assert v.x == pytest.approx(0.0, abs=1e-9)
        assert v.y == pytest.approx(1.0)

    def test_perp(self) -> None:
        p = Vec2(1, 0).perp()
        assert p == Vec2(0, 1)


# ---------------------------------------------------------------------------
# RigidBody tests
# ---------------------------------------------------------------------------


class TestRigidBody:
    def test_construction(self) -> None:
        body = RigidBody(mass=10.0, moment_of_inertia=5.0)
        assert body.mass == 10.0
        assert body.moment_of_inertia == 5.0

    def test_negative_mass_raises(self) -> None:
        with pytest.raises(ValueError, match="mass must be positive"):
            RigidBody(mass=-1.0, moment_of_inertia=1.0)

    def test_zero_mass_raises(self) -> None:
        with pytest.raises(ValueError, match="mass must be positive"):
            RigidBody(mass=0.0, moment_of_inertia=1.0)

    def test_zero_moi_raises(self) -> None:
        with pytest.raises(ValueError, match="moment_of_inertia must be positive"):
            RigidBody(mass=1.0, moment_of_inertia=0.0)

    def test_speed_at_rest(self) -> None:
        body = RigidBody(mass=10.0, moment_of_inertia=5.0)
        assert body.speed == pytest.approx(0.0)

    def test_speed_with_velocity(self) -> None:
        body = RigidBody(mass=10.0, moment_of_inertia=5.0, vel=Vec2(3, 4))
        assert body.speed == pytest.approx(5.0)

    def test_kinetic_energy_translational(self) -> None:
        body = RigidBody(mass=2.0, moment_of_inertia=1.0, vel=Vec2(3, 0))
        # 0.5 * 2 * 9 = 9
        assert body.kinetic_energy_trans == pytest.approx(9.0)

    def test_kinetic_energy_rotational(self) -> None:
        body = RigidBody(mass=1.0, moment_of_inertia=4.0, angular_vel=2.0)
        # 0.5 * 4 * 4 = 8
        assert body.kinetic_energy_rot == pytest.approx(8.0)


# ---------------------------------------------------------------------------
# Moment of inertia tests
# ---------------------------------------------------------------------------


class TestMomentOfInertia:
    def test_disk(self) -> None:
        # I = 0.5 * m * r^2
        assert moment_of_inertia_disk(2.0, 3.0) == pytest.approx(9.0)

    def test_ellipse(self) -> None:
        # I = 0.25 * m * (a^2 + b^2)
        assert moment_of_inertia_ellipse(4.0, 3.0, 4.0) == pytest.approx(25.0)

    def test_rod(self) -> None:
        # I = m * L^2 / 12
        assert moment_of_inertia_rod(12.0, 1.0) == pytest.approx(1.0)

    def test_disk_zero_radius_raises(self) -> None:
        with pytest.raises(ValueError, match="radius must be positive"):
            moment_of_inertia_disk(1.0, 0.0)

    def test_ellipse_zero_axis_raises(self) -> None:
        with pytest.raises(ValueError, match="ellipse semi-axes must be positive"):
            moment_of_inertia_ellipse(1.0, 0.0, 1.0)

    def test_rod_zero_length_raises(self) -> None:
        with pytest.raises(ValueError, match="length must be positive"):
            moment_of_inertia_rod(1.0, 0.0)


# ---------------------------------------------------------------------------
# Impulse mechanics
# ---------------------------------------------------------------------------


class TestComputeJumpImpulse:
    def _make_colinear_config(self) -> dict:
        """Force pushes exactly through both COMs → no torque."""
        return {
            "force_magnitude": 100.0,
            "force_direction_rad": math.pi / 2,  # straight up
            "contact_point": Vec2(0, 5),  # directly above asteroid COM
            "asteroid_com": Vec2(0, 0),
            "jumper_com": Vec2(0, 6),
        }

    def test_colinear_gives_zero_asteroid_torque(self) -> None:
        cfg = self._make_colinear_config()
        _, ast_torque, _ = compute_jump_impulse(**cfg)
        assert abs(ast_torque) < 1e-9

    def test_colinear_gives_zero_jumper_torque(self) -> None:
        cfg = self._make_colinear_config()
        _, _, jmp_torque = compute_jump_impulse(**cfg)
        assert abs(jmp_torque) < 1e-9

    def test_impulse_direction(self) -> None:
        cfg = self._make_colinear_config()
        J, _, _ = compute_jump_impulse(**cfg)
        assert J.x == pytest.approx(0.0, abs=1e-9)
        assert J.y == pytest.approx(100.0)

    def test_offcentre_creates_torque(self) -> None:
        # Contact shifted 2 m to the right of the COM line
        J, ast_torque, jmp_torque = compute_jump_impulse(
            force_magnitude=100.0,
            force_direction_rad=math.pi / 2,
            contact_point=Vec2(2, 5),  # offset right
            asteroid_com=Vec2(0, 0),
            jumper_com=Vec2(0, 6),
        )
        assert abs(ast_torque) > 1.0
        assert abs(jmp_torque) > 1.0

    def test_opposite_torques_direction(self) -> None:
        """Asteroid and jumper spin in opposite directions for off-centre jump."""
        J, ast_torque, jmp_torque = compute_jump_impulse(
            force_magnitude=100.0,
            force_direction_rad=math.pi / 2,
            contact_point=Vec2(2, 5),
            asteroid_com=Vec2(0, 0),
            jumper_com=Vec2(0, 6),
        )
        # Asteroid torque is −r×J (reaction), jumper is +r×J
        assert math.copysign(1, ast_torque) != math.copysign(1, jmp_torque)

    def test_zero_force_magnitude(self) -> None:
        J, at, jt = compute_jump_impulse(0.0, 0.0, Vec2(1, 0), Vec2(0, 0), Vec2(2, 0))
        assert J.length() == pytest.approx(0.0)
        assert at == pytest.approx(0.0)
        assert jt == pytest.approx(0.0)

    def test_negative_force_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="force_magnitude must be non-negative"):
            compute_jump_impulse(-1.0, 0.0, Vec2(), Vec2(), Vec2(1, 0))


class TestApplyImpulse:
    def test_translational_change(self) -> None:
        body = RigidBody(mass=2.0, moment_of_inertia=1.0)
        apply_impulse(body, Vec2(4, 0), 0.0)
        assert body.vel.x == pytest.approx(2.0)
        assert body.angular_vel == pytest.approx(0.0)

    def test_rotational_change(self) -> None:
        body = RigidBody(mass=1.0, moment_of_inertia=2.0)
        apply_impulse(body, Vec2(0, 0), 6.0)
        assert body.angular_vel == pytest.approx(3.0)

    def test_combined_change(self) -> None:
        body = RigidBody(mass=2.0, moment_of_inertia=4.0)
        apply_impulse(body, Vec2(2, 0), 8.0)
        assert body.vel.x == pytest.approx(1.0)
        assert body.angular_vel == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# Spring launch
# ---------------------------------------------------------------------------


class TestSpringLaunch:
    def _make_spring(self) -> SpringLaunch:
        return SpringLaunch(
            total_impulse=200.0,
            force_direction_rad=math.pi / 2,
            contact_point=Vec2(0, 5),
            asteroid_com=Vec2(0, 0),
            jumper_com=Vec2(0, 6),
            duration=0.4,
        )

    def test_initial_not_complete(self) -> None:
        spring = self._make_spring()
        assert not spring.is_complete

    def test_completes_after_full_duration(self) -> None:
        spring = self._make_spring()
        # Step in one large chunk
        spring.step(0.4)
        assert spring.is_complete

    def test_total_impulse_conserved(self) -> None:
        """Sum of step impulses ≈ total_impulse."""
        spring = self._make_spring()
        dt = 0.01
        total_y = 0.0
        while not spring.is_complete:
            result = spring.step(dt)
            if result is not None:
                impulse, _, _ = result
                total_y += impulse.y
        # Should be close to 200 N·s
        assert total_y == pytest.approx(200.0, rel=0.01)

    def test_step_after_completion_returns_none(self) -> None:
        spring = self._make_spring()
        spring.step(0.5)  # complete it
        result = spring.step(0.1)
        assert result is None

    def test_invalid_duration_raises(self) -> None:
        with pytest.raises(ValueError, match="duration must be positive"):
            SpringLaunch(
                total_impulse=100.0,
                force_direction_rad=0.0,
                contact_point=Vec2(),
                asteroid_com=Vec2(),
                jumper_com=Vec2(0, 1),
                duration=0.0,  # invalid
            )

    def test_negative_total_impulse_raises(self) -> None:
        with pytest.raises(ValueError, match="total_impulse must be non-negative"):
            SpringLaunch(
                total_impulse=-1.0,
                force_direction_rad=0.0,
                contact_point=Vec2(),
                asteroid_com=Vec2(),
                jumper_com=Vec2(0, 1),
                duration=0.1,
            )

    def test_step_zero_dt_raises(self) -> None:
        spring = self._make_spring()
        with pytest.raises(ValueError, match="dt must be positive"):
            spring.step(0.0)

    @pytest.mark.parametrize("dt", [math.nan, math.inf, -math.inf])
    def test_step_non_finite_dt_raises(self, dt: float) -> None:
        spring = self._make_spring()
        with pytest.raises(ValueError, match="dt must be positive"):
            spring.step(dt)


# ---------------------------------------------------------------------------
# Integration
# ---------------------------------------------------------------------------


class TestIntegration:
    def test_position_updates(self) -> None:
        body = RigidBody(mass=1.0, moment_of_inertia=1.0, vel=Vec2(2, 3))
        integrate_body(body, 1.0)
        assert body.pos.x == pytest.approx(2.0)
        assert body.pos.y == pytest.approx(3.0)

    def test_angle_updates(self) -> None:
        body = RigidBody(mass=1.0, moment_of_inertia=1.0, angular_vel=math.pi)
        integrate_body(body, 1.0)
        assert body.angle == pytest.approx(math.pi)

    def test_zero_dt_raises(self) -> None:
        body = RigidBody(mass=1.0, moment_of_inertia=1.0)
        with pytest.raises(ValueError, match="dt must be positive"):
            integrate_body(body, 0.0)

    @pytest.mark.parametrize("dt", [math.nan, math.inf, -math.inf])
    def test_non_finite_dt_raises(self, dt: float) -> None:
        body = RigidBody(mass=1.0, moment_of_inertia=1.0)
        with pytest.raises(ValueError, match="dt must be positive"):
            integrate_body(body, dt)


# ---------------------------------------------------------------------------
# Newton's laws — momentum conservation
# ---------------------------------------------------------------------------


class TestMomentumConservation:
    def _system_at_rest(self) -> SimState:
        asteroid = RigidBody(mass=160.0, moment_of_inertia=100.0, pos=Vec2(0, 0))
        jumper = RigidBody(mass=80.0, moment_of_inertia=5.0, pos=Vec2(0, 6))
        return SimState(asteroid=asteroid, jumper=jumper)

    def test_initial_momentum_is_zero(self) -> None:
        state = self._system_at_rest()
        p = state.total_linear_momentum
        assert p.length() == pytest.approx(0.0)

    def test_sim_state_rejects_same_body(self) -> None:
        body = RigidBody(mass=1.0, moment_of_inertia=1.0)
        with pytest.raises(ValueError, match="asteroid and jumper must differ"):
            SimState(asteroid=body, jumper=body)

    def test_step_simulation_zero_dt_raises(self) -> None:
        state = self._system_at_rest()
        with pytest.raises(ValueError, match="dt must be positive"):
            step_simulation(state, 0.0)

    @pytest.mark.parametrize("dt", [math.nan, math.inf, -math.inf])
    def test_step_simulation_non_finite_dt_raises(self, dt: float) -> None:
        state = self._system_at_rest()
        with pytest.raises(ValueError, match="dt must be positive"):
            step_simulation(state, dt)

    def test_momentum_conserved_through_colinear_jump(self) -> None:
        state = self._system_at_rest()
        spring = SpringLaunch(
            total_impulse=500.0,
            force_direction_rad=math.pi / 2,
            contact_point=Vec2(0, 5),
            asteroid_com=state.asteroid.pos,
            jumper_com=state.jumper.pos,
            duration=0.4,
        )
        state.spring = spring
        state.phase = "jumping"
        # Run until spring completes
        for _ in range(100):
            step_simulation(state, 0.005)
        # Total momentum should remain zero (started at rest, no external forces)
        p = state.total_linear_momentum
        assert p.length() == pytest.approx(0.0, abs=1e-6)

    def test_momentum_conserved_through_offcentre_jump(self) -> None:
        state = self._system_at_rest()
        spring = SpringLaunch(
            total_impulse=500.0,
            force_direction_rad=math.pi / 2,
            contact_point=Vec2(3, 5),  # off-centre
            asteroid_com=state.asteroid.pos,
            jumper_com=state.jumper.pos,
            duration=0.4,
        )
        state.spring = spring
        state.phase = "jumping"
        for _ in range(100):
            step_simulation(state, 0.005)
        p = state.total_linear_momentum
        assert p.length() == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# off_centre_ratio
# ---------------------------------------------------------------------------


class TestOffCentreRatio:
    def test_through_both_coms_is_zero(self) -> None:
        # Contact on the line between asteroid_com and jumper_com → ratio ≈ 0
        ratio = off_centre_ratio(
            contact_point=Vec2(0, 3),
            asteroid_com=Vec2(0, 0),
            jumper_com=Vec2(0, 6),
        )
        assert ratio == pytest.approx(0.0, abs=1e-6)

    def test_maximally_offcentre(self) -> None:
        # COMs on y-axis; contact shifted horizontally → truly off-centre
        ratio = off_centre_ratio(
            contact_point=Vec2(3, 3),  # shifted sideways off the y-axis line
            asteroid_com=Vec2(0, 0),
            jumper_com=Vec2(0, 6),
        )
        # Should be > 0 since contact is not on the COM-to-COM line
        assert ratio > 0.0
        assert ratio <= 1.0

    def test_ratio_in_range(self) -> None:
        for shift in [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]:
            r = off_centre_ratio(
                contact_point=Vec2(shift, 3),
                asteroid_com=Vec2(0, 0),
                jumper_com=Vec2(0, 6),
            )
            assert 0.0 <= r <= 1.0
