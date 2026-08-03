"""Tests for the rate-of-closure impact model.

The model answers one question: when a clubhead translates *and* rotates,
how different is the velocity (path, attack angle, speed) of the actual
impact point from the velocity of the tracked reference point (COM or
geometric center)?

Frame convention under test (documented in model.py):
+Y toward target, +Z up, +X to a right-handed golfer's trail side (right
of the target line). Negative horizontal deviation = path moving LEFT.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from rate_of_closure.model import (
    ImpactScenario,
    solve,
    sweep,
)

pytestmark = pytest.mark.unit


def _scenario(**overrides: object) -> ImpactScenario:
    """A representative tour-driver scenario; fields overridable per test."""
    defaults: dict[str, object] = {
        "clubhead_speed_mph": 120.0,
        "omega_plane_dps": 2200.0,
        "omega_shaft_dps": 1700.0,
        "lie_angle_deg": 58.0,
        "com_to_face_mm": 35.0,
        "impact_offset_toe_mm": 0.0,
        "impact_offset_high_mm": 0.0,
        "contact_duration_us": 450.0,
    }
    defaults.update(overrides)
    return ImpactScenario(**defaults)  # type: ignore[arg-type]


class TestReferenceCase:
    """With zero rotation the impact point moves exactly with the COM."""

    def test_zero_rotation_means_zero_deviation(self) -> None:
        result = solve(_scenario(omega_plane_dps=0.0, omega_shaft_dps=0.0))
        assert result.path_deviation_deg == pytest.approx(0.0, abs=1e-12)
        assert result.aoa_deviation_deg == pytest.approx(0.0, abs=1e-12)
        assert result.speed_delta_mph == pytest.approx(0.0, abs=1e-12)

    def test_reference_speed_is_preserved(self) -> None:
        result = solve(_scenario())
        assert result.reference_speed_mph == pytest.approx(120.0)


class TestCommenterCase:
    """The forum case: 35 mm offset, 2000 deg/s about a vertical axis.

    omega * r = 34.907 rad/s * 0.035 m = 1.2217 m/s = 2.733 mph, which is a
    1.30 degree angular deviation at 120 mph — not the 1.2 mph the online
    calculator was read as (that was m/s).
    """

    def test_tangential_speed_matches_hand_calculation(self) -> None:
        # Pure vertical-axis closure: lie 90 puts the shaft axis vertical.
        result = solve(
            _scenario(omega_plane_dps=0.0, omega_shaft_dps=2000.0, lie_angle_deg=90.0)
        )
        assert result.tangential_speed_mph == pytest.approx(2.733, abs=0.005)

    def test_path_deviation_is_about_1p3_degrees(self) -> None:
        result = solve(
            _scenario(omega_plane_dps=0.0, omega_shaft_dps=2000.0, lie_angle_deg=90.0)
        )
        assert result.path_deviation_deg == pytest.approx(-1.30, abs=0.01)

    def test_speed_change_is_negligible_even_when_direction_is_not(self) -> None:
        """The perpendicular component redirects; it barely changes speed."""
        result = solve(
            _scenario(omega_plane_dps=0.0, omega_shaft_dps=2000.0, lie_angle_deg=90.0)
        )
        assert abs(result.speed_delta_mph) < 0.05
        assert abs(result.path_deviation_deg) > 1.0


class TestSignConventions:
    """Closing rotation must move the contact-point path LEFT and the
    in-plane swing rotation must add loft (shallow the delivery)."""

    def test_closure_shifts_path_left(self) -> None:
        result = solve(_scenario(omega_plane_dps=0.0))
        assert result.path_deviation_deg < 0.0

    def test_plane_rotation_alone_also_shifts_path_left(self) -> None:
        result = solve(_scenario(omega_shaft_dps=0.0))
        assert result.path_deviation_deg < 0.0

    def test_plane_rotation_shallows_delivery(self) -> None:
        result = solve(_scenario(omega_shaft_dps=0.0))
        assert result.aoa_deviation_deg > 0.0

    def test_combined_tour_case_matches_prototype(self) -> None:
        result = solve(_scenario())
        assert result.path_deviation_deg == pytest.approx(-1.70, abs=0.02)
        assert result.aoa_deviation_deg == pytest.approx(0.63, abs=0.02)

    def test_reversing_rotation_mirrors_the_deviation(self) -> None:
        left = solve(_scenario())
        right = solve(_scenario(omega_plane_dps=-2200.0, omega_shaft_dps=-1700.0))
        assert right.path_deviation_deg == pytest.approx(
            -left.path_deviation_deg, abs=1e-9
        )


class TestScaling:
    """The deviation angle scales like omega*r/v — small-angle linear."""

    def test_deviation_doubles_with_offset(self) -> None:
        base = solve(_scenario())
        double = solve(_scenario(com_to_face_mm=70.0))
        ratio = double.path_deviation_deg / base.path_deviation_deg
        assert ratio == pytest.approx(2.0, abs=0.02)

    def test_deviation_halves_with_speed(self) -> None:
        base = solve(_scenario())
        fast = solve(_scenario(clubhead_speed_mph=240.0))
        ratio = base.path_deviation_deg / fast.path_deviation_deg
        assert ratio == pytest.approx(2.0, abs=0.02)


class TestFaceRotationDuringContact:
    def test_closure_during_contact_is_about_a_degree(self) -> None:
        result = solve(_scenario())
        assert 0.8 < result.closure_during_contact_deg < 1.6

    def test_zero_contact_time_means_zero_rotation(self) -> None:
        result = solve(_scenario(contact_duration_us=0.0))
        assert result.closure_during_contact_deg == pytest.approx(0.0)
        assert result.loft_gain_during_contact_deg == pytest.approx(0.0)


class TestImpactLocation:
    """Off-center impact points get their own velocity — the launch-monitor
    geometric-center question."""

    def test_toe_impact_differs_from_center(self) -> None:
        center = solve(_scenario())
        toe = solve(_scenario(impact_offset_toe_mm=15.0))
        assert toe.path_deviation_deg != pytest.approx(
            center.path_deviation_deg, abs=1e-6
        )

    def test_high_face_impact_changes_delivered_speed(self) -> None:
        center = solve(_scenario())
        high = solve(_scenario(impact_offset_high_mm=10.0))
        assert high.speed_delta_mph != pytest.approx(center.speed_delta_mph, abs=1e-6)


class TestSweep:
    def test_sweep_is_vectorized_and_monotonic(self) -> None:
        omegas = np.linspace(0.0, 4000.0, 9)
        deviations = sweep(_scenario(omega_plane_dps=0.0), "omega_shaft_dps", omegas)
        assert deviations.shape == (9,)
        assert deviations[0] == pytest.approx(0.0, abs=1e-9)
        assert np.all(np.diff(deviations) < 0.0)  # more closure -> further left

    def test_sweep_matches_pointwise_solve(self) -> None:
        omegas = np.array([500.0, 1500.0, 3000.0])
        swept = sweep(_scenario(), "omega_shaft_dps", omegas)
        for value, expected in zip(omegas, swept, strict=True):
            single = solve(_scenario(omega_shaft_dps=float(value)))
            assert expected == pytest.approx(single.path_deviation_deg, abs=1e-9)


class TestGeometryOutputs:
    """The 3D view consumes these; they must be consistent unit vectors."""

    def test_frame_vectors_are_orthonormal(self) -> None:
        result = solve(_scenario())
        shaft = np.array(result.shaft_axis)
        normal = np.array(result.plane_normal)
        assert np.linalg.norm(shaft) == pytest.approx(1.0)
        assert np.linalg.norm(normal) == pytest.approx(1.0)
        assert float(shaft @ normal) == pytest.approx(0.0, abs=1e-12)

    def test_angular_velocity_vector_recombines(self) -> None:
        scenario = _scenario()
        result = solve(scenario)
        omega = np.array(result.omega_dps)
        magnitude = math.hypot(scenario.omega_plane_dps, scenario.omega_shaft_dps)
        # plane normal and shaft axis are orthogonal, so magnitudes add in
        # quadrature exactly.
        assert np.linalg.norm(omega) == pytest.approx(magnitude)
