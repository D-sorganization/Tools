"""TDD tests for asteroid_jumper.controller."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from asteroid_jumper.asteroid_shape import ShapeKind
from asteroid_jumper.controller import SimController


class TestSimController:
    def test_initial_state_is_ready(self) -> None:
        ctrl = SimController()
        assert ctrl.state.phase == "ready"

    def test_initial_speeds_are_zero(self) -> None:
        ctrl = SimController()
        assert ctrl.jumper_speed() == pytest.approx(0.0)
        assert ctrl.asteroid_speed() == pytest.approx(0.0)

    def test_configure_changes_mass(self) -> None:
        ctrl = SimController()
        ctrl.configure(asteroid_mass=300.0)
        assert ctrl.asteroid_mass == pytest.approx(300.0)

    def test_configure_changes_shape(self) -> None:
        ctrl = SimController()
        ctrl.configure(shape_kind=ShapeKind.CIRCLE)
        assert ctrl.asteroid_shape_kind == ShapeKind.CIRCLE

    def test_set_force_angle(self) -> None:
        ctrl = SimController()
        ctrl.set_force_angle(45.0)
        assert ctrl.force_angle_deg == pytest.approx(45.0)

    def test_set_impulse(self) -> None:
        ctrl = SimController()
        ctrl.set_impulse(800.0)
        assert ctrl.impulse_magnitude == pytest.approx(800.0)

    def test_negative_impulse_raises(self) -> None:
        ctrl = SimController()
        with pytest.raises((AssertionError, ValueError)):
            ctrl.set_impulse(-100.0)

    def test_start_jump_changes_phase(self) -> None:
        ctrl = SimController()
        ctrl.start_jump()
        assert ctrl.state.phase == "jumping"

    def test_double_jump_raises(self) -> None:
        ctrl = SimController()
        ctrl.start_jump()
        with pytest.raises((AssertionError, ValueError)):
            ctrl.start_jump()

    def test_reset_restores_ready_state(self) -> None:
        ctrl = SimController()
        ctrl.start_jump()
        # Tick until flight
        for _ in range(100):
            ctrl.tick(0.01)
        ctrl.reset()
        assert ctrl.state.phase == "ready"
        assert ctrl.jumper_speed() == pytest.approx(0.0)

    def test_tick_advances_time(self) -> None:
        ctrl = SimController()
        ctrl.start_jump()
        ctrl.tick(0.1)
        assert ctrl.state.time > 0.0

    def test_off_centre_fraction_range(self) -> None:
        ctrl = SimController()
        frac = ctrl.off_centre_fraction()
        assert 0.0 <= frac <= 1.0

    def test_colinear_jump_minimises_spin(self) -> None:
        """When force goes through both COMs, angular speed is minimal."""
        ctrl = SimController()
        # Position jumper directly above asteroid COM
        ctrl.set_force_angle(90.0)  # straight up
        ctrl.set_jump_direction(90.0)
        ctrl.state = ctrl._build_state()
        ctrl.start_jump()
        for _ in range(200):
            ctrl.tick(0.002)
        assert ctrl.jumper_angular_speed() < 0.5  # nearly no spin

    def test_offcentre_jump_creates_spin(self) -> None:
        """Off-centre jump creates measurable angular velocity on both bodies."""
        ctrl = SimController()
        ctrl.set_force_angle(0.0)  # right side of asteroid
        ctrl.set_jump_direction(90.0)  # jump straight up → off-centre torque
        ctrl.state = ctrl._build_state()
        ctrl.start_jump()
        for _ in range(200):
            ctrl.tick(0.002)
        # At least one body should spin
        total_spin = ctrl.jumper_angular_speed() + ctrl.asteroid_angular_speed()
        assert total_spin > 0.1

    def test_leg_phase_is_zero_at_rest(self) -> None:
        ctrl = SimController()
        assert ctrl.leg_phase() == pytest.approx(0.0)

    def test_leg_phase_advances_during_jump(self) -> None:
        ctrl = SimController()
        ctrl.start_jump()
        ctrl.tick(0.2)  # halfway through spring
        assert ctrl.leg_phase() > 0.0

    def test_all_shape_kinds_build_without_error(self) -> None:
        for kind in ShapeKind:
            ctrl = SimController(asteroid_shape_kind=kind)
            assert ctrl.shape is not None
            assert ctrl.state is not None
