"""TDD contract tests for synchronization monitoring and skew estimation."""

from __future__ import annotations

import pytest
from sidekick.lab.mocap.acquisition import FramePacket
from sidekick.lab.mocap.sync import (
    ClockSkewEstimate,
    SyncAnomalyType,
    SyncMonitor,
    SyncQuality,
)


def _make_packet(source_id: str, seq: int, ts_ns: int, host_ns: int) -> FramePacket:
    return FramePacket(
        source_id=source_id,
        sequence_number=seq,
        timestamp_ns=ts_ns,
        host_monotonic_ns=host_ns,
        image_bytes=b"\x00" * 16,
        pixel_format="GRAY8",
        resolution_px=(4, 4),
    )


def test_clock_skew_estimate_contracts() -> None:
    est = ClockSkewEstimate(
        source_a="cam-01",
        source_b="cam-02",
        offset_ns=1500,
        drift_rate_ppm=2.5,
        max_jitter_ns=300,
        sample_count=10,
    )
    assert est.source_a == "cam-01"
    assert est.source_b == "cam-02"
    assert est.offset_ns == 1500
    assert est.drift_rate_ppm == 2.5
    assert est.max_jitter_ns == 300
    assert est.sample_count == 10

    with pytest.raises(
        ValueError, match="cannot estimate skew between a source and itself"
    ):
        ClockSkewEstimate(
            source_a="cam-01",
            source_b="cam-01",
            offset_ns=0,
            drift_rate_ppm=0.0,
            max_jitter_ns=0,
            sample_count=1,
        )

    with pytest.raises(ValueError, match="sample_count must be positive"):
        ClockSkewEstimate(
            source_a="cam-01",
            source_b="cam-02",
            offset_ns=0,
            drift_rate_ppm=0.0,
            max_jitter_ns=0,
            sample_count=0,
        )


def test_sync_monitor_tracks_sequence_and_detects_anomalies() -> None:
    monitor = SyncMonitor(max_allowable_skew_ns=10_000_000)  # 10ms

    # Normal frame sequence on cam-01
    p1 = _make_packet("cam-01", 1, 100_000_000, 100_000_000)
    anomalies1 = monitor.record_frame(p1)
    assert len(anomalies1) == 0

    # Normal next frame
    p2 = _make_packet("cam-01", 2, 133_333_333, 133_333_333)
    anomalies2 = monitor.record_frame(p2)
    assert len(anomalies2) == 0

    # Dropped frame: seq jumps from 2 to 5 (missing 3, 4)
    p5 = _make_packet("cam-01", 5, 233_333_333, 233_333_333)
    anomalies5 = monitor.record_frame(p5)
    assert len(anomalies5) == 1
    assert anomalies5[0].anomaly_type == SyncAnomalyType.DROPPED
    assert anomalies5[0].expected_seq == 3
    assert anomalies5[0].actual_seq == 5
    assert anomalies5[0].dropped_count == 2

    # Duplicate frame: seq 5 again
    dup = _make_packet("cam-01", 5, 233_333_333, 233_333_333)
    anomalies_dup = monitor.record_frame(dup)
    assert len(anomalies_dup) == 1
    assert anomalies_dup[0].anomaly_type == SyncAnomalyType.DUPLICATE

    # Out of order: seq 4 arrives after seq 5
    p4 = _make_packet("cam-01", 4, 200_000_000, 200_000_000)
    anomalies_ooo = monitor.record_frame(p4)
    assert len(anomalies_ooo) == 1
    assert anomalies_ooo[0].anomaly_type == SyncAnomalyType.OUT_OF_ORDER


def test_sync_monitor_evaluates_inter_camera_skew_and_quality() -> None:
    monitor = SyncMonitor(max_allowable_skew_ns=2_000_000)  # 2ms threshold

    # Tightly synchronized packets from two cameras
    for i in range(1, 11):
        ts = i * 33_333_333
        p_a = _make_packet("cam-01", i, ts, ts)
        p_b = _make_packet(
            "cam-02", i, ts + 50_000, ts + 50_000
        )  # 50 microseconds offset
        monitor.record_frame(p_a)
        monitor.record_frame(p_b)

    skew = monitor.estimate_skew("cam-01", "cam-02")
    assert skew is not None
    assert abs(skew.offset_ns) < 100_000
    quality = monitor.evaluate_quality("cam-01", "cam-02")
    assert quality in {
        SyncQuality.HARDWARE_LOCKED,
        SyncQuality.SOFTWARE_MONOTONIC_ALIGNED,
    }

    # Skew that exceeds threshold fails closed to UNALIGNED_SKEW_EXCEEDED
    p_a_late = _make_packet("cam-01", 11, 400_000_000, 400_000_000)
    p_b_late = _make_packet(
        "cam-02", 11, 405_000_000, 405_000_000
    )  # 5ms offset > 2ms threshold
    monitor.record_frame(p_a_late)
    monitor.record_frame(p_b_late)

    quality_exceeded = monitor.evaluate_quality("cam-01", "cam-02")
    assert quality_exceeded == SyncQuality.UNALIGNED_SKEW_EXCEEDED
