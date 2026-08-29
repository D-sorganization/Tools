from __future__ import annotations

import math

import pytest
from sidekick.lab.mocap import (
    Availability,
    CameraCapabilities,
    CameraIdentity,
    ClockDomain,
    ClockKind,
    CoordinateFrame,
    FeatureSupport,
    FrameStamp,
    Landmark3D,
    NumericRange,
    PixelObservation,
    RigidTransform,
    ShutterKind,
    SkeletonDefinition,
    SupportLevel,
)


def test_camera_identity_requires_stable_provider_and_device_ids() -> None:
    with pytest.raises(ValueError, match="provider_id"):
        CameraIdentity(provider_id="", device_id="camera-01", transport="GenTL")
    with pytest.raises(ValueError, match="device_id"):
        CameraIdentity(provider_id="gentl", device_id=" ", transport="GenTL")


def test_camera_identity_does_not_invent_absent_serial_number() -> None:
    identity = CameraIdentity(
        provider_id="gentl",
        device_id="mac-00-01",
        transport="GigE Vision",
        vendor="Example",
        model="GS-200",
    )
    assert identity.serial_number is None
    assert identity.stable_key == "gentl:mac-00-01"


def test_numeric_range_is_finite_ordered_and_unit_bearing() -> None:
    assert NumericRange(18.0, 10_000_000.0, "us").contains(100.0)
    with pytest.raises(ValueError, match="finite"):
        NumericRange(0.0, math.inf, "Hz")
    with pytest.raises(ValueError, match="minimum"):
        NumericRange(2.0, 1.0, "Hz")
    with pytest.raises(ValueError, match="unit"):
        NumericRange(0.0, 1.0, "")


def test_feature_support_requires_reason_for_degraded_or_unsupported() -> None:
    assert FeatureSupport(SupportLevel.SUPPORTED).reason is None
    with pytest.raises(ValueError, match="reason"):
        FeatureSupport(SupportLevel.DEGRADED)
    with pytest.raises(ValueError, match="reason"):
        FeatureSupport(SupportLevel.UNSUPPORTED, reason=" ")


def test_camera_capabilities_are_explicit_and_nonempty() -> None:
    capabilities = CameraCapabilities(
        resolutions_px=((1632, 1248),),
        frame_rates_hz=(120.0, 225.0),
        pixel_formats=("BayerRG8",),
        shutter=ShutterKind.GLOBAL,
        hardware_trigger=FeatureSupport(SupportLevel.SUPPORTED),
        device_timestamps=FeatureSupport(SupportLevel.SUPPORTED),
        exposure_us=NumericRange(18.0, 10_000_000.0, "us"),
    )
    assert capabilities.supports_mode((1632, 1248), 225.0, "BayerRG8")
    assert not capabilities.supports_mode((1920, 1080), 225.0, "BayerRG8")
    with pytest.raises(ValueError, match="resolutions_px"):
        CameraCapabilities(
            resolutions_px=(),
            frame_rates_hz=(120.0,),
            pixel_formats=("Mono8",),
            shutter=ShutterKind.GLOBAL,
            hardware_trigger=FeatureSupport(SupportLevel.SUPPORTED),
            device_timestamps=FeatureSupport(SupportLevel.SUPPORTED),
        )


def test_clock_domain_and_frame_stamp_preserve_clock_evidence() -> None:
    clock = ClockDomain(
        clock_id="camera-01-hardware",
        kind=ClockKind.DEVICE_HARDWARE,
        tick_period_seconds=1e-9,
        monotonic=True,
    )
    stamp = FrameStamp(
        source_id="camera-01",
        stream_id="video",
        sequence_number=5,
        clock_id=clock.clock_id,
        capture_timestamp_ns=10_000,
        host_monotonic_ns=12_000,
        timing_uncertainty_ns=250,
        exposure_start_ns=9_000,
        exposure_end_ns=9_500,
    )
    assert stamp.clock_id == clock.clock_id
    assert stamp.exposure_duration_ns == 500
    with pytest.raises(ValueError, match="exposure"):
        FrameStamp(
            source_id="camera-01",
            stream_id="video",
            sequence_number=5,
            clock_id=clock.clock_id,
            capture_timestamp_ns=10_000,
            host_monotonic_ns=12_000,
            timing_uncertainty_ns=250,
            exposure_start_ns=10_000,
            exposure_end_ns=9_000,
        )


def test_coordinate_frame_names_axes_and_uses_si() -> None:
    frame = CoordinateFrame.affinedrift_world_v1()
    assert frame.handedness == "right-handed"
    assert (frame.x_axis, frame.y_axis, frame.z_axis) == (
        "toward-target",
        "up",
        "right",
    )
    assert frame.length_unit == "m"


def test_rigid_transform_explicitly_maps_source_to_target() -> None:
    transform = RigidTransform(
        target_frame_id="world-v1",
        source_frame_id="camera-01-optical",
        rotation_wxyz=(1.0, 0.0, 0.0, 0.0),
        translation_m=(1.0, 2.0, 3.0),
    )
    assert transform.transform_name == "T_world-v1_from_camera-01-optical"
    with pytest.raises(ValueError, match="unit quaternion"):
        RigidTransform(
            target_frame_id="world-v1",
            source_frame_id="camera-01-optical",
            rotation_wxyz=(2.0, 0.0, 0.0, 0.0),
            translation_m=(0.0, 0.0, 0.0),
        )


def test_skeleton_rejects_duplicate_keypoint_ids() -> None:
    with pytest.raises(ValueError, match="unique"):
        SkeletonDefinition(
            skeleton_id="test",
            version="1.0.0",
            keypoint_ids=("left-hip", "left-hip"),
        )


def test_pixel_observation_contracts_confidence_and_covariance() -> None:
    observation = PixelObservation(
        observation_id="obs-1",
        camera_id="camera-01",
        frame_sequence=5,
        timestamp_ns=10_000,
        skeleton_id="human-17",
        keypoint_id="left-hip",
        uv_px=(640.5, 480.5),
        confidence=0.9,
        covariance_px2=(1.0, 0.0, 0.0, 1.0),
        availability=Availability.OBSERVED,
    )
    assert observation.confidence == 0.9
    with pytest.raises(ValueError, match="confidence"):
        PixelObservation(
            observation_id="obs-2",
            camera_id="camera-01",
            frame_sequence=5,
            timestamp_ns=10_000,
            skeleton_id="human-17",
            keypoint_id="left-hip",
            uv_px=(640.5, 480.5),
            confidence=1.1,
            covariance_px2=(1.0, 0.0, 0.0, 1.0),
            availability=Availability.OBSERVED,
        )


def test_triangulated_landmark_requires_two_unique_contributing_cameras() -> None:
    with pytest.raises(ValueError, match="two unique cameras"):
        Landmark3D(
            landmark_id="landmark-1",
            world_frame_id="world-v1",
            skeleton_id="human-17",
            keypoint_id="left-hip",
            timestamp_ns=10_000,
            xyz_m=(0.0, 1.0, 0.0),
            covariance_m2=(1e-6, 0.0, 0.0, 0.0, 1e-6, 0.0, 0.0, 0.0, 1e-6),
            contributing_camera_ids=("camera-01",),
            rejected_camera_ids=(),
            method_id="weighted-triangulation-v1",
            availability=Availability.DERIVED,
        )


def test_model_conditioned_single_view_depth_remains_explicit() -> None:
    point = Landmark3D(
        landmark_id="landmark-1",
        world_frame_id="world-v1",
        skeleton_id="human-17",
        keypoint_id="left-hip",
        timestamp_ns=10_000,
        xyz_m=(0.0, 1.0, 0.0),
        covariance_m2=(1e-3, 0.0, 0.0, 0.0, 1e-3, 0.0, 0.0, 0.0, 1e-3),
        contributing_camera_ids=("camera-01",),
        rejected_camera_ids=(),
        method_id="monocular-body-model-v1",
        availability=Availability.MODEL_CONDITIONED,
    )
    assert point.availability is Availability.MODEL_CONDITIONED
