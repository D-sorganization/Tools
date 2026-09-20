"""Contract and regression tests for frame-explicit wedge delivery metrics."""

from __future__ import annotations

import math

import numpy as np
import pytest

from shared.python.golf_club import (
    WedgePreset,
    wedge_preset,
)
from shared.python.golf_club._wedge_delivery_metrics import (
    LinearVelocityWaterfall,
    WedgeDeliveryMetrics,
    compute_wedge_delivery_metrics,
    delivered_bounce_deg,
    linear_velocity_waterfall,
    path_projected_metrics,
)

pytestmark = [pytest.mark.unit, pytest.mark.contract]


SetupTuple = tuple[object, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]


def _canonical_setup() -> SetupTuple:
    parameters = wedge_preset(WedgePreset.MID_BOUNCE)
    pose: np.ndarray = np.eye(4, dtype=float)
    v_ref = np.array([15.0, -2.0, 0.0])
    lie_rad = math.radians(parameters.lie_deg)
    shaft_unit = np.array([0.0, math.sin(lie_rad), -math.cos(lie_rad)])
    shaft_unit /= np.linalg.norm(shaft_unit)
    omega = 25.0 * shaft_unit
    twist = np.concatenate([omega, v_ref])
    ground_normal = np.array([0.0, 1.0, 0.0])
    target_unit = np.array([1.0, 0.0, 0.0])
    return parameters, pose, twist, ground_normal, target_unit, shaft_unit


def test_linear_velocity_waterfall_exact_identity() -> None:
    """Waterfall terms must satisfy v_contact = v_axis + v_shaft + v_other."""
    (
        _params,
        _pose,
        twist,
        ground_normal,
        target_unit,
        shaft_unit,
    ) = _canonical_setup()
    twist[:3] += np.array([10.0, 0.0, 5.0])
    contact_pt = np.array([0.02, 0.01, -0.005])

    waterfall = linear_velocity_waterfall(
        reference_velocity_mps=twist[3:],
        angular_velocity_rad_s=twist[:3],
        shaft_axis_point_m=np.zeros(3),
        shaft_axis_unit=shaft_unit,
        contact_point_m=contact_pt,
        ground_up_unit=ground_normal,
        target_unit=target_unit,
    )

    assert isinstance(waterfall, LinearVelocityWaterfall)
    reconstructed_x = (
        waterfall.base_axis.downrange_mps
        + waterfall.shaft_rotation.downrange_mps
        + waterfall.other_rotation.downrange_mps
    )
    reconstructed_y = (
        waterfall.base_axis.vertical_mps
        + waterfall.shaft_rotation.vertical_mps
        + waterfall.other_rotation.vertical_mps
    )
    reconstructed_z = (
        waterfall.base_axis.lateral_mps
        + waterfall.shaft_rotation.lateral_mps
        + waterfall.other_rotation.lateral_mps
    )

    tc = waterfall.total_contact
    assert math.isclose(reconstructed_x, tc.downrange_mps, abs_tol=1e-12)
    assert math.isclose(reconstructed_y, tc.vertical_mps, abs_tol=1e-12)
    assert math.isclose(reconstructed_z, tc.lateral_mps, abs_tol=1e-12)

    computed_speed = math.hypot(
        tc.downrange_mps,
        tc.vertical_mps,
        tc.lateral_mps,
    )
    assert math.isclose(tc.total_speed_mps, computed_speed, abs_tol=1e-12)


def test_linear_velocity_waterfall_zero_shaft_rate() -> None:
    """When rotation is zero, shaft velocity is 0 and AoA delta is 0."""
    (
        _params,
        _pose,
        twist,
        ground_normal,
        target_unit,
        shaft_unit,
    ) = _canonical_setup()
    twist[:3] = 0.0

    waterfall = linear_velocity_waterfall(
        reference_velocity_mps=twist[3:],
        angular_velocity_rad_s=twist[:3],
        shaft_axis_point_m=np.zeros(3),
        shaft_axis_unit=shaft_unit,
        contact_point_m=np.array([0.02, 0.0, 0.0]),
        ground_up_unit=ground_normal,
        target_unit=target_unit,
    )

    assert waterfall.shaft_rotation.total_speed_mps == pytest.approx(0.0, abs=1e-12)
    assert waterfall.other_rotation.total_speed_mps == pytest.approx(0.0, abs=1e-12)
    assert waterfall.shaft_counterfactual_aoa_delta_deg == pytest.approx(0.0, abs=1e-12)
    assert waterfall.total_aoa_deg == pytest.approx(waterfall.without_shaft_aoa_deg)


def test_counterfactual_aoa_deltas_non_additive_euler() -> None:
    """Counterfactual angle deltas are nonlinear; Shapley values sum to total delta."""
    (
        _params,
        _pose,
        twist,
        ground_normal,
        target_unit,
        shaft_unit,
    ) = _canonical_setup()
    twist[:3] += np.array([8.0, 2.0, 4.0])

    waterfall = linear_velocity_waterfall(
        reference_velocity_mps=twist[3:],
        angular_velocity_rad_s=twist[:3],
        shaft_axis_point_m=np.zeros(3),
        shaft_axis_unit=shaft_unit,
        contact_point_m=np.array([0.02, 0.01, 0.0]),
        ground_up_unit=ground_normal,
        target_unit=target_unit,
    )

    assert waterfall.total_aoa_deg is not None
    assert waterfall.without_shaft_aoa_deg is not None
    assert waterfall.shaft_counterfactual_aoa_delta_deg is not None
    assert waterfall.shaft_shapley_aoa_deg is not None
    assert waterfall.other_shapley_aoa_deg is not None

    base_aoa = math.degrees(
        math.atan2(waterfall.base_axis.vertical_mps, waterfall.base_axis.downrange_mps)
    )
    total_delta = waterfall.total_aoa_deg - base_aoa
    shapley_sum = waterfall.shaft_shapley_aoa_deg + waterfall.other_shapley_aoa_deg
    assert math.isclose(shapley_sum, total_delta, abs_tol=1e-9)


def test_wedge_delivery_metrics_comprehensive() -> None:
    """Full delivery metrics calculation delivers all required cards."""
    parameters = wedge_preset(WedgePreset.MID_BOUNCE)
    pose: np.ndarray = np.eye(4, dtype=float)
    v_ref = np.array([16.0, -2.5, 0.5])
    lie_rad = math.radians(parameters.lie_deg)
    shaft_unit = np.array([0.0, math.sin(lie_rad), -math.cos(lie_rad)])
    shaft_unit /= np.linalg.norm(shaft_unit)
    omega = 20.0 * shaft_unit + np.array([2.0, 1.0, 0.0])
    twist = np.concatenate([omega, v_ref])
    ground_normal = np.array([0.0, 1.0, 0.0])

    metrics = compute_wedge_delivery_metrics(
        parameters=parameters,
        pose=pose,
        twist=twist,
        ground_normal=ground_normal,
        shaft_axis_unit=shaft_unit,
    )

    assert isinstance(metrics, WedgeDeliveryMetrics)
    assert math.isclose(
        metrics.le_total_speed_mps,
        math.hypot(
            metrics.le_downrange_rate_mps,
            metrics.le_vertical_rate_mps,
            metrics.le_lateral_rate_mps,
        ),
        abs_tol=1e-12,
    )
    assert metrics.le_3d_angular_rate_dps > 0.0
    assert 0.0 < metrics.dynamic_loft_deg < 90.0
    assert 0.0 < metrics.dynamic_lie_deg < 90.0
    assert metrics.dynamic_face_angle_deg is not None

    del_b = delivered_bounce_deg(parameters, pose, ground_normal)
    assert math.isclose(metrics.delivered_bounce_deg, del_b, abs_tol=1e-9)
    eff_b, _ref_aoa, margin = path_projected_metrics(
        parameters, pose, twist, ground_normal
    )
    assert math.isclose(
        metrics.path_projected_effective_bounce_deg or 0.0,
        eff_b or 0.0,
        abs_tol=1e-9,
    )
    assert math.isclose(
        metrics.bounce_utilization_margin_deg or 0.0,
        margin or 0.0,
        abs_tol=1e-9,
    )

    assert metrics.low_point_world_m is not None
    assert len(metrics.low_point_world_m) == 3

    explainers = metrics.explainers()
    required_keys = {
        "total_aoa",
        "without_shaft_aoa",
        "shaft_counterfactual_aoa_delta",
        "le_vertical_rate",
        "le_downrange_rate",
        "le_lateral_rate",
        "dynamic_loft",
        "dynamic_lie",
        "dynamic_face_angle",
        "delivered_bounce",
        "low_point",
        "waterfall_axis",
        "waterfall_shaft",
        "waterfall_other",
    }
    present_keys = {exp.key for exp in explainers}
    assert required_keys.issubset(present_keys)

    for exp in explainers:
        assert exp.equation != ""
        assert exp.frame != ""
        assert exp.units != ""
        assert exp.assumptions != ""
        assert exp.availability in ("available", "undefined")
