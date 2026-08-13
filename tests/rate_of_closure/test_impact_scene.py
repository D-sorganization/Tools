"""UI-independent impact-scene contracts shared by presentation layers."""

from __future__ import annotations

import json

import numpy as np
import pytest

from rate_of_closure.club import get_club
from rate_of_closure.model import ImpactScenario
from rate_of_closure.simulation import (
    SimulationConfig,
    impact_scene_for_run,
    run_simulation,
)

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


def _scene():
    run = run_simulation(
        SimulationConfig(
            scenario=ImpactScenario(
                clubhead_speed_mph=30.0,
                lie_angle_deg=64.0,
                omega_plane_dps=0.0,
                omega_shaft_dps=1307.0,
                com_to_face_mm=20.0,
            ),
            club=get_club("Pitching Wedge"),
            impact_time_s=0.03025,
        )
    )
    return impact_scene_for_run(run)


def test_scene_exposes_frame_explicit_geometry_and_velocity_identity() -> None:
    scene = _scene()
    vectors = {vector.key: vector for vector in scene.vectors}

    assert scene.event_label == "Impact"
    assert scene.frame_id == "app_frame:x_target,y_up,z_right"
    assert {"total", "axis_translation", "shaft_rotation", "other_rotation"} <= set(
        vectors
    )
    reconstructed = (
        np.asarray(vectors["axis_translation"].vector)
        + np.asarray(vectors["shaft_rotation"].vector)
        + np.asarray(vectors["other_rotation"].vector)
    )
    np.testing.assert_allclose(reconstructed, vectors["total"].vector, atol=1e-12)
    assert all(vector.units == "m/s" for vector in scene.vectors)
    assert vectors["sasho_face_center_rotation"].origin_m == pytest.approx(
        scene.face_center_point_m
    )
    expected_face_center_velocity = np.asarray(scene.face_center_velocity_mps)
    np.testing.assert_allclose(
        scene.face_center_dplane.travel_direction_unit,
        expected_face_center_velocity / np.linalg.norm(expected_face_center_velocity),
    )


def test_scene_metrics_are_self_describing_and_strict_json_safe() -> None:
    scene = _scene()
    metrics = {metric.key: metric for metric in scene.metrics}

    assert metrics["total_aoa"].equation
    assert metrics["shaft_counterfactual_aoa"].assumptions
    assert metrics["shaft_rotation_rate"].units == "deg/s"
    payload = scene.to_json_dict()
    assert json.dumps(payload, allow_nan=False)
    assert payload["format"] == "rate-of-closure.impact-scene/v3"
    assert payload["face_center_dplane"]["status"] == "defined"
    assert payload["sasho_face_center_rotation"]["method_id"] == (
        "sasho_nearest_shaft_face_center_rotation_only_aoa_v1"
    )
    assert payload["angular_velocity_rad_s"] == scene.angular_velocity_rad_s
    assert metrics["spin_loft_3d"].units == "deg"
    assert metrics["spin_loft_planar"].equation
    assert metrics["spin_loft_residual"].assumptions
    sasho = metrics["sasho_nearest_shaft_face_center_rotation_only_aoa_v1"]
    assert sasho.label == "Sasho Face-Center Rotation-Only AoA"
    assert "complete angular velocity" in sasho.assumptions


def test_scene_contains_engineering_orientation_and_screw_geometry() -> None:
    scene = _scene()

    assert np.linalg.norm(scene.face_normal_unit) == pytest.approx(1.0)
    assert np.dot(scene.face_normal_unit, scene.leading_edge_unit) == pytest.approx(
        0.0, abs=1e-12
    )
    assert scene.screw_axis is not None
    assert scene.screw_axis.contact_distance_m >= 0.0
    assert len(scene.spin_loft_sector_unit) == 25
    normal = np.asarray(scene.face_center_dplane.dplane_normal_unit)
    assert all(
        np.dot(normal, np.asarray(direction)) == pytest.approx(0.0, abs=1e-12)
        for direction in scene.spin_loft_sector_unit
    )
