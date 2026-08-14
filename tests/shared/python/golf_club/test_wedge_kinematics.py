"""Contract tests for impact-point wedge rigid-body kinematics."""

from __future__ import annotations

import math

import numpy as np
import pytest

from shared.python.golf_club import (
    WedgeKinematicState,
    analyze_wedge_kinematics,
    angle_of_attack_deg,
)


def _state(**overrides: object) -> WedgeKinematicState:
    values: dict[str, object] = {
        "frame_id": "target_ground",
        "reference_position_m": (0.0, 0.0, 0.0),
        "reference_velocity_mps": (10.0, -1.0, 0.0),
        "angular_velocity_rad_s": (0.0, 20.0, 0.0),
        "shaft_axis_point_m": (0.0, 0.0, 0.0),
        "shaft_axis_unit": (0.0, 1.0, 0.0),
        "contact_point_m": (0.02, 0.0, 0.0),
        "face_normal_unit": (1.0, 0.0, 0.0),
        "leading_edge_tangent_unit": (0.0, 0.0, 1.0),
        "ground_up_unit": (0.0, 1.0, 0.0),
        "arc_tangent_unit": (1.0, 0.0, 0.0),
        "arc_tangent_rate_per_s": (0.0, 0.0, 0.0),
    }
    values.update(overrides)
    return WedgeKinematicState(**values)  # type: ignore[arg-type]


def test_worked_example_attributes_shaft_twist_to_aoa() -> None:
    lie_rad = math.radians(64.0)
    lean_rad = math.radians(15.0)
    shaft_axis = (
        -math.sin(lean_rad),
        math.cos(lean_rad) * math.sin(lie_rad),
        -math.cos(lean_rad) * math.cos(lie_rad),
    )
    twist_rate = math.radians(1307.0)
    shaft_omega = tuple(twist_rate * value for value in shaft_axis)
    total_velocity = (13.207454, -2.328830, 0.0)
    shaft_velocity = np.cross(shaft_omega, (0.02, 0.0, 0.0))
    state = _state(
        reference_velocity_mps=tuple(np.asarray(total_velocity) - shaft_velocity),
        angular_velocity_rad_s=shaft_omega,
        shaft_axis_unit=shaft_axis,
        face_normal_unit=(math.cos(lie_rad), 0.0, math.sin(lie_rad)),
        leading_edge_tangent_unit=(-math.sin(lie_rad), 0.0, math.cos(lie_rad)),
    )

    result = analyze_wedge_kinematics(state)

    assert result.total_aoa_deg == pytest.approx(-10.0, abs=2e-5)
    assert result.without_shaft_aoa_deg == pytest.approx(-9.18118, abs=2e-5)
    assert result.shaft_counterfactual_aoa_delta_deg == pytest.approx(
        -0.81882, abs=3e-5
    )
    assert result.shaft_rotation_velocity_mps == pytest.approx(
        (0.0, -0.193183, -0.396084), abs=2e-6
    )


def test_decomposition_sums_exactly_at_contact() -> None:
    state = _state(
        reference_position_m=(0.4, -0.2, 0.1),
        reference_velocity_mps=(9.0, -1.2, 0.6),
        angular_velocity_rad_s=(3.0, 20.0, -4.0),
        shaft_axis_point_m=(0.1, 0.0, -0.1),
        contact_point_m=(0.03, -0.01, 0.02),
    )

    result = analyze_wedge_kinematics(state)
    reconstructed = np.sum(
        np.asarray(
            (
                result.shaft_axis_velocity_mps,
                result.shaft_rotation_velocity_mps,
                result.non_shaft_rotation_velocity_mps,
            )
        ),
        axis=0,
    )

    assert reconstructed == pytest.approx(result.contact_velocity_mps, abs=1e-12)


def test_shaft_contribution_is_zero_on_physical_axis() -> None:
    result = analyze_wedge_kinematics(_state(contact_point_m=(0.0, 0.15, 0.0)))

    assert result.shaft_rotation_velocity_mps == pytest.approx((0.0, 0.0, 0.0))
    assert result.shaft_counterfactual_aoa_delta_deg == pytest.approx(0.0)


def test_shaft_vertical_velocity_matches_lie_offset_formula() -> None:
    lie_rad = math.radians(64.0)
    rate = 17.0
    offset = 0.028
    axis = (0.0, math.sin(lie_rad), -math.cos(lie_rad))
    result = analyze_wedge_kinematics(
        _state(
            angular_velocity_rad_s=tuple(rate * value for value in axis),
            shaft_axis_unit=axis,
            contact_point_m=(offset, 0.0, 0.0),
        )
    )

    expected_vertical = -rate * offset * math.cos(lie_rad)
    assert result.shaft_rotation_velocity_mps[1] == pytest.approx(expected_vertical)


def test_reference_point_shift_preserves_all_contact_metrics() -> None:
    original = _state(angular_velocity_rad_s=(3.0, 20.0, -4.0))
    shift = np.asarray((0.3, -0.1, 0.2))
    shifted_velocity = np.asarray(original.reference_velocity_mps) + np.cross(
        original.angular_velocity_rad_s, shift
    )
    shifted = _state(
        reference_position_m=tuple(np.asarray(original.reference_position_m) + shift),
        reference_velocity_mps=tuple(shifted_velocity),
        angular_velocity_rad_s=original.angular_velocity_rad_s,
    )

    first = analyze_wedge_kinematics(original)
    second = analyze_wedge_kinematics(shifted)

    assert second.contact_velocity_mps == pytest.approx(first.contact_velocity_mps)
    assert second.shaft_axis_velocity_mps == pytest.approx(
        first.shaft_axis_velocity_mps
    )
    assert second.shaft_counterfactual_aoa_delta_deg == pytest.approx(
        first.shaft_counterfactual_aoa_delta_deg
    )


def test_shapley_attribution_closes_nonlinear_aoa_difference() -> None:
    result = analyze_wedge_kinematics(_state(angular_velocity_rad_s=(6.0, 20.0, -5.0)))

    attributed = result.shaft_shapley_aoa_deg + result.non_shaft_shapley_aoa_deg
    total_change = result.total_aoa_deg - result.shaft_axis_translation_aoa_deg
    assert attributed == pytest.approx(total_change, abs=1e-12)


def test_orientation_rates_include_ground_and_arc_references() -> None:
    result = analyze_wedge_kinematics(
        _state(
            angular_velocity_rad_s=(2.0, 3.0, 4.0),
            arc_tangent_rate_per_s=(0.0, 0.0, 0.5),
        )
    )

    assert result.face_normal_rate_per_s == pytest.approx((0.0, 4.0, -3.0))
    assert result.face_normal_3d_rate_rad_s == pytest.approx(5.0)
    assert result.leading_edge_ground_heading_rate_rad_s == pytest.approx(3.0)
    assert result.arc_ground_heading_rate_rad_s == pytest.approx(-0.5)
    assert result.leading_edge_relative_arc_heading_rate_rad_s == pytest.approx(3.5)


def test_instantaneous_screw_axis_reports_contact_clearance() -> None:
    result = analyze_wedge_kinematics(
        _state(
            reference_velocity_mps=(0.0, 0.0, 0.0),
            angular_velocity_rad_s=(0.0, 2.0, 0.0),
            contact_point_m=(0.03, 0.0, 0.0),
        )
    )

    assert result.screw_axis is not None
    assert result.screw_axis.point_nearest_origin_m == pytest.approx((0.0, 0.0, 0.0))
    assert result.screw_axis.pitch_m_per_rad == pytest.approx(0.0)
    assert result.screw_axis.contact_distance_m == pytest.approx(0.03)


def test_aoa_is_undefined_without_horizontal_speed() -> None:
    assert angle_of_attack_deg((0.0, -3.0, 0.0), (0.0, 1.0, 0.0)) is None


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("shaft_axis_unit", (0.0, 2.0, 0.0), "shaft_axis_unit must be unit length"),
        (
            "leading_edge_tangent_unit",
            (1.0, 0.0, 0.0),
            "face normal and leading edge tangent must be orthogonal",
        ),
        (
            "arc_tangent_rate_per_s",
            (0.1, 0.0, 0.0),
            "arc tangent rate must be orthogonal to arc tangent",
        ),
    ),
)
def test_state_rejects_invalid_frame_geometry(
    field: str, value: object, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _state(**{field: value})
