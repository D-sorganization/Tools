"""TDD contract tests for markerless-mocap intrinsic calibration."""

from __future__ import annotations

import math

import pytest
from sidekick.lab.mocap.calibration import (
    CalibrationDegeneracyKind,
    CalibrationObservation,
    CalibrationPatternKind,
    CalibrationQuality,
    CalibrationTarget,
    DistortionCoefficients,
    DistortionModel,
    FisheyeIntrinsics,
    IntrinsicCalibrationResult,
    PinholeIntrinsics,
    ReprojectionResidual,
    check_coverage_and_degeneracy,
    evaluate_intrinsic_quality,
)
from sidekick.lab.mocap.devices import CameraIdentity


def test_distortion_coefficients_contracts() -> None:
    # Valid Brown-Conrady coefficients (k1, k2, p1, p2, k3)
    dist = DistortionCoefficients(
        model=DistortionModel.BROWN_CONRADY,
        coefficients=(0.1, -0.05, 0.001, -0.002, 0.01),
    )
    assert dist.model == DistortionModel.BROWN_CONRADY
    assert len(dist.coefficients) == 5
    assert dist.coefficients[0] == 0.1

    # Invalid coefficient length for Brown-Conrady (expects 4, 5, or 8)
    with pytest.raises(
        ValueError, match="Brown-Conrady distortion requires 4, 5, or 8 coefficients"
    ):
        DistortionCoefficients(
            model=DistortionModel.BROWN_CONRADY,
            coefficients=(0.1, 0.2),
        )

    # Kannala-Brandt expects 4 coefficients
    with pytest.raises(
        ValueError, match="Kannala-Brandt fisheye distortion requires 4 coefficients"
    ):
        DistortionCoefficients(
            model=DistortionModel.KANNALA_BRANDT,
            coefficients=(0.1, 0.2, 0.3),
        )


def test_pinhole_intrinsics_contracts_and_projection() -> None:
    dist = DistortionCoefficients(
        model=DistortionModel.NONE,
        coefficients=(),
    )
    pinhole = PinholeIntrinsics(
        fx=1000.0,
        fy=1000.0,
        cx=960.0,
        cy=540.0,
        resolution_px=(1920, 1080),
        distortion=dist,
    )
    assert pinhole.fx == 1000.0
    assert pinhole.resolution_px == (1920, 1080)

    # Reject non-positive focal length
    with pytest.raises(ValueError, match="focal length must be positive"):
        PinholeIntrinsics(
            fx=-100.0,
            fy=1000.0,
            cx=960.0,
            cy=540.0,
            resolution_px=(1920, 1080),
            distortion=dist,
        )

    # Reject principal point out of bounds
    with pytest.raises(ValueError, match="principal point must be within image bounds"):
        PinholeIntrinsics(
            fx=1000.0,
            fy=1000.0,
            cx=-10.0,
            cy=540.0,
            resolution_px=(1920, 1080),
            distortion=dist,
        )

    # Project nominal 3D point in front of camera (Z > 0)
    pt_3d = (0.0, 0.0, 2.0)
    uv = pinhole.project_point(pt_3d)
    assert math.isclose(uv[0], 960.0, abs_tol=1e-5)
    assert math.isclose(uv[1], 540.0, abs_tol=1e-5)

    # Unproject ray direction
    ray = pinhole.unproject_point((960.0, 540.0))
    assert math.isclose(ray[0], 0.0, abs_tol=1e-5)
    assert math.isclose(ray[1], 0.0, abs_tol=1e-5)
    assert math.isclose(ray[2], 1.0, abs_tol=1e-5)

    # Reject point on or behind camera plane (Z <= 0)
    with pytest.raises(
        ValueError, match="cannot project point with non-positive depth"
    ):
        pinhole.project_point((1.0, 1.0, 0.0))


def test_fisheye_intrinsics_contracts_and_projection() -> None:
    dist = DistortionCoefficients(
        model=DistortionModel.KANNALA_BRANDT,
        coefficients=(0.0, 0.0, 0.0, 0.0),
    )
    fisheye = FisheyeIntrinsics(
        fx=600.0,
        fy=600.0,
        cx=640.0,
        cy=480.0,
        resolution_px=(1280, 960),
        distortion=dist,
    )
    assert fisheye.fx == 600.0

    # Project optical center
    uv = fisheye.project_point((0.0, 0.0, 1.0))
    assert math.isclose(uv[0], 640.0, abs_tol=1e-5)
    assert math.isclose(uv[1], 480.0, abs_tol=1e-5)

    # Unproject optical center
    ray = fisheye.unproject_point((640.0, 480.0))
    assert math.isclose(ray[0], 0.0, abs_tol=1e-5)
    assert math.isclose(ray[1], 0.0, abs_tol=1e-5)
    assert math.isclose(ray[2], 1.0, abs_tol=1e-5)


def test_calibration_target_and_observation_provenance() -> None:
    target = CalibrationTarget(
        pattern_kind=CalibrationPatternKind.CHESSBOARD,
        grid_size=(9, 6),
        feature_spacing_m=0.025,
    )
    assert target.pattern_kind == CalibrationPatternKind.CHESSBOARD
    assert target.total_points == 54

    with pytest.raises(ValueError, match="grid_size dimensions must be at least 2"):
        CalibrationTarget(
            pattern_kind=CalibrationPatternKind.CHESSBOARD,
            grid_size=(1, 5),
            feature_spacing_m=0.025,
        )

    # Calibration observation
    obj_pts = tuple((float(x), float(y), 0.0) for x in range(3) for y in range(2))
    img_pts = tuple(
        (float(x * 100), float(y * 100)) for x in range(3) for y in range(2)
    )

    obs = CalibrationObservation(
        camera_key="synthetic:cam-01",
        frame_sequence=1,
        timestamp_ns=100_000_000,
        target_object_points=obj_pts,
        detected_pixel_points=img_pts,
        detection_confidence=0.98,
    )
    assert obs.camera_key == "synthetic:cam-01"
    assert len(obs.target_object_points) == 6
    assert len(obs.detected_pixel_points) == 6

    # Mismatched object and detected point counts
    with pytest.raises(
        ValueError,
        match="target_object_points and detected_pixel_points count must match",
    ):
        CalibrationObservation(
            camera_key="synthetic:cam-01",
            frame_sequence=1,
            timestamp_ns=100_000_000,
            target_object_points=obj_pts,
            detected_pixel_points=img_pts[:4],
            detection_confidence=0.98,
        )


def test_reprojection_residual_and_quality_floors() -> None:
    res = ReprojectionResidual(
        observation_index=0,
        point_index=1,
        residual_uv_px=(0.3, -0.4),
        norm_px=0.5,
    )
    assert res.observation_index == 0
    assert math.isclose(res.norm_px, 0.5, abs_tol=1e-6)

    # Inconsistent norm
    with pytest.raises(
        ValueError, match="norm_px does not match residual_uv_px Euclidean norm"
    ):
        ReprojectionResidual(
            observation_index=0,
            point_index=1,
            residual_uv_px=(0.3, -0.4),
            norm_px=1.0,
        )

    # Qualification evaluation
    # Good quality: mean residual < 1.0 px, views >= 5, sensor coverage >= 0.4
    quality, degeneracies = evaluate_intrinsic_quality(
        mean_residual_px=0.45,
        views_count=10,
        sensor_coverage_ratio=0.6,
        floor_threshold_px=1.0,
    )
    assert quality == CalibrationQuality.QUALIFIED
    assert len(degeneracies) == 0

    # Degraded quality: residual borderline or views low
    quality_deg, degen_deg = evaluate_intrinsic_quality(
        mean_residual_px=1.2,
        views_count=4,
        sensor_coverage_ratio=0.35,
        floor_threshold_px=1.0,
    )
    assert quality_deg in {CalibrationQuality.DEGRADED, CalibrationQuality.UNQUALIFIED}
    assert CalibrationDegeneracyKind.INSUFFICIENT_VIEWS in degen_deg

    # Unqualified: high residual > 2.0 * floor_threshold_px
    quality_unq, degen_unq = evaluate_intrinsic_quality(
        mean_residual_px=2.5,
        views_count=12,
        sensor_coverage_ratio=0.5,
        floor_threshold_px=1.0,
    )
    assert quality_unq == CalibrationQuality.UNQUALIFIED
    assert CalibrationDegeneracyKind.HIGH_REPROJECTION_ERROR in degen_unq


def test_coverage_and_degeneracy_detection() -> None:
    # Corners concentrated in single quadrant (0 to 300, 0 to 300 on 1920x1080)
    points = [(100.0, 100.0), (200.0, 100.0), (100.0, 200.0), (200.0, 200.0)]
    coverage, degeneracies = check_coverage_and_degeneracy(
        points=points,
        resolution_px=(1920, 1080),
        min_views=5,
        views_count=2,
    )
    assert coverage < 0.2
    assert CalibrationDegeneracyKind.POOR_SENSOR_COVERAGE in degeneracies
    assert CalibrationDegeneracyKind.INSUFFICIENT_VIEWS in degeneracies


def test_intrinsic_calibration_result_binding_to_camera_identity() -> None:
    cam = CameraIdentity(
        provider_id="synthetic",
        device_id="cam-calibration-01",
        transport="memory",
    )
    dist = DistortionCoefficients(
        model=DistortionModel.NONE,
        coefficients=(),
    )
    intrinsics = PinholeIntrinsics(
        fx=800.0,
        fy=800.0,
        cx=320.0,
        cy=240.0,
        resolution_px=(640, 480),
        distortion=dist,
    )

    cov = (
        (0.01, 0.0, 0.0, 0.0),
        (0.0, 0.01, 0.0, 0.0),
        (0.0, 0.0, 0.005, 0.0),
        (0.0, 0.0, 0.0, 0.005),
    )

    result = IntrinsicCalibrationResult(
        camera_key=cam.stable_key,
        calibrated_at_utc="2026-09-07T14:00:00Z",
        intrinsics=intrinsics,
        mean_reprojection_residual_px=0.32,
        max_reprojection_residual_px=0.88,
        parameter_covariance=cov,
        quality=CalibrationQuality.QUALIFIED,
        degeneracies=(),
    )
    assert result.camera_key == "synthetic:cam-calibration-01"
    assert result.quality == CalibrationQuality.QUALIFIED
    assert result.is_qualified
