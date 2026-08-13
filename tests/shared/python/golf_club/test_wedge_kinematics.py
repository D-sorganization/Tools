"""Contract tests for impact-point wedge rigid-body kinematics."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from shared.python.golf_club import (
    SASHO_FACE_CENTER_ROTATION_METHOD_ID,
    WedgeKinematicState,
    analyze_wedge_kinematics,
    angle_of_attack_deg,
    sasho_face_center_rotation_aoa,
)

_SASHO_GOLDEN = (
    Path(__file__).resolve().parents[4]
    / "src/rate_of_closure/web/src/model/__fixtures__"
    / "sasho_face_center_rotation_golden_v1.json"
)


def _state(**overrides: object) -> WedgeKinematicState:
    values: dict[str, object] = {
        "frame_id": "target_ground",
        "reference_position_m": (0.0, 0.0, 0.0),
        "reference_velocity_mps": (10.0, -1.0, 0.0),
        "angular_velocity_rad_s": (0.0, 20.0, 0.0),
        "shaft_axis_point_m": (0.0, 0.0, 0.0),
        "shaft_axis_unit": (0.0, 1.0, 0.0),
        "face_center_point_m": (0.02, 0.0, 0.0),
        "contact_point_m": (0.02, 0.0, 0.0),
        "face_normal_unit": (1.0, 0.0, 0.0),
        "leading_edge_tangent_unit": (0.0, 0.0, 1.0),
        "ground_up_unit": (0.0, 1.0, 0.0),
        "arc_tangent_unit": (1.0, 0.0, 0.0),
        "arc_tangent_rate_per_s": (0.0, 0.0, 0.0),
    }
    values.update(overrides)
    return WedgeKinematicState(**values)  # type: ignore[arg-type]


def _worked_example_state(rate_dps: float = 1307.0) -> WedgeKinematicState:
    lie_rad = math.radians(64.0)
    lean_rad = math.radians(15.0)
    shaft_axis = (
        -math.sin(lean_rad),
        math.cos(lean_rad) * math.sin(lie_rad),
        -math.cos(lean_rad) * math.cos(lie_rad),
    )
    reference_twist_rate = math.radians(1307.0)
    reference_shaft_omega = tuple(reference_twist_rate * value for value in shaft_axis)
    total_velocity = (13.207454, -2.328830, 0.0)
    reference_shaft_velocity = np.cross(reference_shaft_omega, (0.02, 0.0, 0.0))
    shaft_omega = tuple(math.radians(rate_dps) * value for value in shaft_axis)
    return _state(
        reference_velocity_mps=tuple(
            np.asarray(total_velocity) - reference_shaft_velocity
        ),
        angular_velocity_rad_s=shaft_omega,
        shaft_axis_unit=shaft_axis,
        face_normal_unit=(math.cos(lie_rad), 0.0, math.sin(lie_rad)),
        leading_edge_tangent_unit=(-math.sin(lie_rad), 0.0, math.cos(lie_rad)),
    )


def test_worked_example_attributes_shaft_twist_to_aoa() -> None:
    result = analyze_wedge_kinematics(_worked_example_state())

    assert result.total_aoa_deg == pytest.approx(-10.0, abs=2e-5)
    assert result.without_shaft_aoa_deg == pytest.approx(-9.18118, abs=2e-5)
    assert result.shaft_counterfactual_aoa_delta_deg == pytest.approx(
        -0.81882, abs=3e-5
    )
    assert result.shaft_rotation_velocity_mps == pytest.approx(
        (0.0, -0.193183, -0.396084), abs=2e-6
    )
    assert result.shaft_axis_velocity_mps == pytest.approx(
        (13.207454, -2.135647, 0.396084), abs=2e-6
    )
    assert np.linalg.norm(result.contact_velocity_mps) == pytest.approx(
        30.0 * 0.44704, abs=2e-6
    )
    assert result.shaft_vertical_velocity_share == pytest.approx(0.08295277, abs=1e-8)


def test_sasho_face_center_rotation_matches_cross_runtime_golden() -> None:
    fixture = json.loads(_SASHO_GOLDEN.read_text(encoding="utf-8"))

    result = sasho_face_center_rotation_aoa(
        angular_velocity_rad_s=fixture["angular_velocity_rad_s"],
        shaft_axis_point_m=fixture["shaft_axis_point_m"],
        shaft_axis_unit=fixture["shaft_axis_unit"],
        face_center_point_m=fixture["face_center_point_m"],
        ground_up_unit=fixture["ground_up_unit"],
    )

    expected = fixture["expected"]
    assert result.method_id == expected["method_id"]
    assert result.method_id == SASHO_FACE_CENTER_ROTATION_METHOD_ID
    assert result.nearest_shaft_point_m == pytest.approx(
        expected["nearest_shaft_point_m"]
    )
    assert result.lever_arm_m == pytest.approx(expected["lever_arm_m"])
    assert result.velocity_mps == pytest.approx(expected["velocity_mps"])
    assert result.aoa_deg == pytest.approx(expected["aoa_deg"])


def test_sasho_face_center_rotation_is_invariant_to_shaft_line_datum() -> None:
    first = sasho_face_center_rotation_aoa(
        angular_velocity_rad_s=(2.0, 10.0, -4.0),
        shaft_axis_point_m=(0.0, 0.0, 0.0),
        shaft_axis_unit=(0.0, 1.0, 0.0),
        face_center_point_m=(0.02, 0.1, 0.03),
        ground_up_unit=(0.0, 1.0, 0.0),
    )
    shifted = sasho_face_center_rotation_aoa(
        angular_velocity_rad_s=(2.0, 10.0, -4.0),
        shaft_axis_point_m=(0.0, 0.4, 0.0),
        shaft_axis_unit=(0.0, -1.0, 0.0),
        face_center_point_m=(0.02, 0.1, 0.03),
        ground_up_unit=(0.0, 1.0, 0.0),
    )

    assert shifted.nearest_shaft_point_m == pytest.approx(first.nearest_shaft_point_m)
    assert shifted.velocity_mps == pytest.approx(first.velocity_mps)
    assert shifted.aoa_deg == pytest.approx(first.aoa_deg)


@pytest.mark.parametrize(
    ("omega", "expected_velocity", "expected_aoa"),
    (
        ((2.0, 10.0, -4.0), (0.3, -0.14, -0.2), -21.220700223593433),
        ((-2.0, -10.0, 4.0), (-0.3, 0.14, 0.2), 21.220700223593433),
        ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), None),
    ),
)
def test_sasho_rotation_sign_and_zero_availability(
    omega: tuple[float, float, float],
    expected_velocity: tuple[float, float, float],
    expected_aoa: float | None,
) -> None:
    result = sasho_face_center_rotation_aoa(
        angular_velocity_rad_s=omega,
        shaft_axis_point_m=(0.0, 0.0, 0.0),
        shaft_axis_unit=(0.0, 1.0, 0.0),
        face_center_point_m=(0.02, 0.1, 0.03),
        ground_up_unit=(0.0, 1.0, 0.0),
    )

    assert result.velocity_mps == pytest.approx(expected_velocity)
    if expected_aoa is None:
        assert result.aoa_deg is None
    else:
        assert result.aoa_deg == pytest.approx(expected_aoa)


def test_sasho_rotation_is_rigid_frame_equivariant() -> None:
    rotation = np.array(((0.0, 0.0, 1.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)))
    arguments = {
        "angular_velocity_rad_s": np.array((2.0, 10.0, -4.0)),
        "shaft_axis_point_m": np.array((0.0, 0.0, 0.0)),
        "shaft_axis_unit": np.array((0.0, 1.0, 0.0)),
        "face_center_point_m": np.array((0.02, 0.1, 0.03)),
        "ground_up_unit": np.array((0.0, 1.0, 0.0)),
    }
    original = sasho_face_center_rotation_aoa(**arguments)
    transformed = sasho_face_center_rotation_aoa(
        **{name: rotation @ value for name, value in arguments.items()}
    )

    assert transformed.nearest_shaft_point_m == pytest.approx(
        rotation @ original.nearest_shaft_point_m
    )
    assert transformed.lever_arm_m == pytest.approx(rotation @ original.lever_arm_m)
    assert transformed.velocity_mps == pytest.approx(rotation @ original.velocity_mps)
    assert transformed.aoa_deg == pytest.approx(original.aoa_deg)


def test_sasho_rotation_is_unavailable_when_face_center_is_on_shaft() -> None:
    result = sasho_face_center_rotation_aoa(
        angular_velocity_rad_s=(2.0, 10.0, -4.0),
        shaft_axis_point_m=(0.0, 0.0, 0.0),
        shaft_axis_unit=(0.0, 1.0, 0.0),
        face_center_point_m=(0.0, 0.1, 0.0),
        ground_up_unit=(0.0, 1.0, 0.0),
    )

    assert result.lever_arm_m == pytest.approx((0.0, 0.0, 0.0))
    assert result.velocity_mps == pytest.approx((0.0, 0.0, 0.0))
    assert result.aoa_deg is None


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        (
            "angular_velocity_rad_s",
            (math.nan, 10.0, -4.0),
            "must contain only finite values",
        ),
        ("shaft_axis_unit", (0.0, 0.0, 0.0), "must be unit length"),
    ),
)
def test_sasho_face_center_rotation_rejects_invalid_geometry(
    field: str, value: object, message: str
) -> None:
    arguments: dict[str, object] = {
        "angular_velocity_rad_s": (2.0, 10.0, -4.0),
        "shaft_axis_point_m": (0.0, 0.0, 0.0),
        "shaft_axis_unit": (0.0, 1.0, 0.0),
        "face_center_point_m": (0.02, 0.1, 0.03),
        "ground_up_unit": (0.0, 1.0, 0.0),
    }
    arguments[field] = value

    with pytest.raises(ValueError, match=message):
        sasho_face_center_rotation_aoa(**arguments)


@pytest.mark.parametrize(
    ("rate_dps", "total_aoa_deg", "shaft_delta_deg", "vertical_share"),
    (
        (0.0, -9.18117341, 0.0, 0.0),
        (652.0, -9.59110580, -0.40993239, 0.04317608),
        (1003.0, -9.81060610, -0.62943269, 0.06491089),
        (1307.0, -9.99999795, -0.81882454, 0.08295277),
        (1611.0, -10.18869313, -1.00751972, 0.10031162),
        (2432.0, -10.69458379, -1.51341038, 0.14406769),
    ),
)
def test_worked_example_rotation_sensitivity_is_pinned(
    rate_dps: float,
    total_aoa_deg: float,
    shaft_delta_deg: float,
    vertical_share: float,
) -> None:
    result = analyze_wedge_kinematics(_worked_example_state(rate_dps))

    assert result.total_aoa_deg == pytest.approx(total_aoa_deg, abs=1e-8)
    assert result.shaft_counterfactual_aoa_delta_deg == pytest.approx(
        shaft_delta_deg, abs=1e-8
    )
    assert result.shaft_vertical_velocity_share == pytest.approx(
        vertical_share, abs=1e-8
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
