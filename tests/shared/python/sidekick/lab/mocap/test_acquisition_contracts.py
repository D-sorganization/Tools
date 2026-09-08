"""TDD test suite for markerless-mocap camera acquisition protocol (TOOLS-M2 #4713)."""

from __future__ import annotations

import time

import pytest
from sidekick.lab.mocap import (
    CameraCapabilities,
    CameraIdentity,
    FeatureSupport,
    NumericRange,
    ShutterKind,
    SupportLevel,
)
from sidekick.lab.mocap.acquisition import (
    AcquisitionError,
    CaptureGroup,
    DropPolicy,
    FramePacket,
    FrameSource,
    PrerecordedFrameSource,
    QueueFullError,
    SourceState,
    SyntheticFrameSource,
)


def _sample_identity(device_id: str = "cam-01") -> CameraIdentity:
    return CameraIdentity(
        provider_id="synthetic",
        device_id=device_id,
        transport="memory",
        vendor="SyntheticLabs",
        model="VirtualSensor",
    )


def _sample_capabilities() -> CameraCapabilities:
    return CameraCapabilities(
        resolutions_px=((640, 480), (1280, 720)),
        frame_rates_hz=(30.0, 60.0),
        pixel_formats=("RGB8", "GRAY8"),
        shutter=ShutterKind.GLOBAL,
        hardware_trigger=FeatureSupport(SupportLevel.UNSUPPORTED, "Virtual camera"),
        device_timestamps=FeatureSupport(SupportLevel.SUPPORTED),
        exposure_us=NumericRange(100.0, 33000.0, "us"),
    )


def test_frame_packet_contracts() -> None:
    packet = FramePacket(
        source_id="cam-01",
        sequence_number=1,
        timestamp_ns=1_000_000,
        host_monotonic_ns=1_000_000,
        image_bytes=b"\x00" * (640 * 480),
        pixel_format="GRAY8",
        resolution_px=(640, 480),
    )
    assert packet.source_id == "cam-01"
    assert packet.sequence_number == 1
    assert len(packet.image_bytes) == 640 * 480

    with pytest.raises(ValueError, match="sequence_number"):
        FramePacket(
            source_id="cam-01",
            sequence_number=-1,
            timestamp_ns=100,
            host_monotonic_ns=100,
            image_bytes=b"0",
            pixel_format="GRAY8",
            resolution_px=(1, 1),
        )


def test_synthetic_frame_source_lifecycle() -> None:
    ident = _sample_identity("cam-01")
    caps = _sample_capabilities()
    source = SyntheticFrameSource(
        source_id="cam-01",
        identity=ident,
        capabilities=caps,
        resolution_px=(640, 480),
        frame_rate_hz=30.0,
        pixel_format="GRAY8",
    )

    assert source.state == SourceState.UNINITIALIZED
    assert isinstance(source, FrameSource)

    source.initialize()
    assert source.state == SourceState.INITIALIZED

    source.start_capture()
    assert source.state == SourceState.CAPTURING

    packet = source.read_frame(timeout_seconds=1.0)
    assert packet.source_id == "cam-01"
    assert packet.resolution_px == (640, 480)
    assert packet.sequence_number == 1

    packet2 = source.read_frame(timeout_seconds=1.0)
    assert packet2.sequence_number == 2
    assert packet2.timestamp_ns > packet.timestamp_ns

    source.stop_capture()
    assert source.state == SourceState.STOPPED

    source.close()
    assert source.state == SourceState.CLOSED


def test_synthetic_frame_source_invalid_mode() -> None:
    ident = _sample_identity("cam-01")
    caps = _sample_capabilities()
    with pytest.raises(ValueError, match="unsupported mode"):
        SyntheticFrameSource(
            source_id="cam-01",
            identity=ident,
            capabilities=caps,
            resolution_px=(1920, 1080),
            frame_rate_hz=30.0,
            pixel_format="GRAY8",
        )


def test_capture_group_multi_camera_synchronization() -> None:
    s1 = SyntheticFrameSource(
        source_id="cam-01",
        identity=_sample_identity("cam-01"),
        capabilities=_sample_capabilities(),
        resolution_px=(640, 480),
        frame_rate_hz=30.0,
        pixel_format="GRAY8",
    )
    s2 = SyntheticFrameSource(
        source_id="cam-02",
        identity=_sample_identity("cam-02"),
        capabilities=_sample_capabilities(),
        resolution_px=(640, 480),
        frame_rate_hz=30.0,
        pixel_format="GRAY8",
    )

    group = CaptureGroup(
        sources=(s1, s2), max_queue_size=10, drop_policy=DropPolicy.DROP_OLDEST
    )
    group.initialize()
    group.start_capture()

    frames = group.read_frames(timeout_seconds=1.0)
    assert "cam-01" in frames
    assert "cam-02" in frames
    assert frames["cam-01"].sequence_number >= 1
    assert frames["cam-02"].sequence_number >= 1

    group.stop_capture()
    group.close()


def test_prerecorded_frame_source_replay() -> None:
    ident = _sample_identity("cam-rec")
    caps = _sample_capabilities()
    packets = [
        FramePacket(
            source_id="cam-rec",
            sequence_number=seq,
            timestamp_ns=seq * 33_333_333,
            host_monotonic_ns=seq * 33_333_333,
            image_bytes=b"\xaa" * 100,
            pixel_format="GRAY8",
            resolution_px=(640, 480),
        )
        for seq in range(1, 4)
    ]
    source = PrerecordedFrameSource(
        source_id="cam-rec",
        identity=ident,
        capabilities=caps,
        frames=packets,
        loop=False,
    )
    source.initialize()
    source.start_capture()

    r1 = source.read_frame()
    assert r1.sequence_number == 1
    r2 = source.read_frame()
    assert r2.sequence_number == 2
    r3 = source.read_frame()
    assert r3.sequence_number == 3

    with pytest.raises(AcquisitionError, match="end of prerecorded stream"):
        source.read_frame()

    source.close()


def test_capture_group_backpressure_and_drop_policy() -> None:
    s1 = SyntheticFrameSource(
        source_id="cam-01",
        identity=_sample_identity("cam-01"),
        capabilities=_sample_capabilities(),
        resolution_px=(640, 480),
        frame_rate_hz=60.0,
        pixel_format="GRAY8",
    )
    group_fail_closed = CaptureGroup(
        sources=(s1,),
        max_queue_size=2,
        drop_policy=DropPolicy.FAIL_CLOSED,
    )
    group_fail_closed.initialize()
    group_fail_closed.start_capture()

    time.sleep(0.05)
    with pytest.raises(QueueFullError):
        for _ in range(10):
            packet = s1.read_frame(timeout_seconds=0.1)
            group_fail_closed.push_frame_direct("cam-01", packet)

    group_fail_closed.close()
