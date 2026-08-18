from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from shared.python.golf_club import (
    GroundPlane,
    TurfCalibrationStatus,
    TurfContactStatus,
    TurfPreset,
    WedgeContactFeature,
    WedgePreset,
    evaluate_wedge_turf_wrench,
    turf_profile_preset,
    wedge_preset,
)


def _pose(*, height_m: float = 0.0, pitch_deg: float = 0.0) -> np.ndarray:
    angle = np.deg2rad(pitch_deg)
    cosine = np.cos(angle)
    sine = np.sin(angle)
    pose = np.eye(4)
    pose[:3, :3] = ((cosine, -sine, 0.0), (sine, cosine, 0.0), (0.0, 0.0, 1.0))
    pose[1, 3] = height_m
    return pose


def test_distributed_wrench_reports_named_active_patches_and_passive_power() -> None:
    parameters = wedge_preset(WedgePreset.MID_BOUNCE)
    profile = turf_profile_preset(TurfPreset.FIRM_FAIRWAY)

    result = evaluate_wedge_turf_wrench(
        parameters,
        profile,
        _pose(height_m=-0.003),
        (0.0, 0.0, 0.0, 1.0, -0.5, 0.0),
        GroundPlane(),
    )

    assert result.status is TurfContactStatus.ACTIVE
    assert result.active_patches
    assert result.force_world_n[1] > 0.0
    assert result.dissipated_power_w >= 0.0
    assert result.maximum_penetration_m > 0.0
    assert not result.supports_turf_rankings


def test_calibration_state_is_the_only_turf_ranking_gate() -> None:
    profile = replace(
        turf_profile_preset(TurfPreset.SOFT_TURF),
        calibration_status=TurfCalibrationStatus.CALIBRATED,
    )
    result = evaluate_wedge_turf_wrench(
        wedge_preset(WedgePreset.MID_BOUNCE),
        profile,
        _pose(height_m=-0.003),
        (0.0, 0.0, 0.0, 0.0, -0.5, 0.0),
        GroundPlane(),
    )

    assert result.supports_turf_rankings


def test_pitch_changes_the_first_active_sole_region() -> None:
    parameters = wedge_preset(WedgePreset.MID_BOUNCE)
    profile = turf_profile_preset(TurfPreset.FIRM_FAIRWAY)
    leading = evaluate_wedge_turf_wrench(
        parameters,
        profile,
        _pose(height_m=-0.001, pitch_deg=0.0),
        (0.0, 0.0, 0.0, 0.0, -0.1, 0.0),
        GroundPlane(),
    )
    trailing = evaluate_wedge_turf_wrench(
        parameters,
        profile,
        _pose(height_m=-0.001, pitch_deg=30.0),
        (0.0, 0.0, 0.0, 0.0, -0.1, 0.0),
        GroundPlane(),
    )

    leading_features = {patch.feature for patch in leading.active_patches}
    trailing_features = {patch.feature for patch in trailing.active_patches}
    assert WedgeContactFeature.LEADING_EDGE_CENTER in leading_features
    assert any(
        feature.value.startswith("trailing_sole") for feature in trailing_features
    )


def test_sloped_ground_changes_contact_patch_selection() -> None:
    parameters = wedge_preset(WedgePreset.MID_BOUNCE)
    profile = turf_profile_preset(TurfPreset.FIRM_FAIRWAY)
    flat = evaluate_wedge_turf_wrench(
        parameters,
        profile,
        _pose(height_m=-0.001),
        (0.0, 0.0, 0.0, 0.0, -0.1, 0.0),
        GroundPlane(),
    )
    slope_normal = np.asarray((0.0, 1.0, 0.25))
    slope_normal /= np.linalg.norm(slope_normal)
    sloped = evaluate_wedge_turf_wrench(
        parameters,
        profile,
        _pose(height_m=-0.001),
        (0.0, 0.0, 0.0, 0.0, -0.1, 0.0),
        GroundPlane(normal_unit=tuple(slope_normal)),
    )

    assert {patch.feature for patch in flat.active_patches} != {
        patch.feature for patch in sloped.active_patches
    }


def test_invalid_pose_and_twist_are_rejected() -> None:
    parameters = wedge_preset(WedgePreset.MID_BOUNCE)
    profile = turf_profile_preset(TurfPreset.FIRM_FAIRWAY)
    with pytest.raises(ValueError, match="4x4"):
        evaluate_wedge_turf_wrench(
            parameters, profile, np.eye(3), np.zeros(6), GroundPlane()
        )
    with pytest.raises(ValueError, match="6-vector"):
        evaluate_wedge_turf_wrench(
            parameters, profile, np.eye(4), np.zeros(3), GroundPlane()
        )
