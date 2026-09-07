"""TDD contract tests for crash-safe recording and manifest integrity."""

from __future__ import annotations

from pathlib import Path

import pytest
from sidekick.lab.mocap.acquisition import FramePacket
from sidekick.lab.mocap.devices import CameraIdentity
from sidekick.lab.mocap.enums import ClockKind, SessionState
from sidekick.lab.mocap.geometry import CoordinateFrame
from sidekick.lab.mocap.recording import (
    FrameIndexEntry,
    RecordingReader,
    RecordingWriter,
)
from sidekick.lab.mocap.session import (
    MethodDescriptor,
    MocapSessionManifest,
    RecordingPolicy,
)
from sidekick.lab.mocap.timebase import ClockDomain


def _manifest(
    consent: bool = True, no_store: bool = False, raw_retained: bool = True
) -> MocapSessionManifest:
    return MocapSessionManifest(
        session_id="session-record-001",
        created_at_utc="2026-09-07T12:00:00Z",
        state=SessionState.RECORDING,
        world_frame=CoordinateFrame.affinedrift_world_v1(),
        cameras=(
            CameraIdentity(
                provider_id="synthetic",
                device_id="cam-01",
                transport="memory",
                vendor="D-sorganization",
                model="rec-cam",
                serial_number="REC-01",
            ),
        ),
        clocks=(
            ClockDomain(
                clock_id="cam-01-clock",
                kind=ClockKind.DEVICE_HARDWARE,
                tick_period_seconds=1e-9,
                monotonic=True,
            ),
        ),
        methods=(
            MethodDescriptor(
                method_id="recording-test",
                version="1.0.0",
                implementation="sidekick.lab.mocap.recording",
                license_spdx="MIT",
            ),
        ),
        recording_policy=RecordingPolicy(
            consent_recorded=consent,
            raw_video_retained=raw_retained,
            retention_days=7 if raw_retained else 0,
            no_store=no_store,
        ),
        calibration_ids=("calib-001",),
        warnings=(),
    )


def _make_packet(source_id: str, seq: int, ts_ns: int) -> FramePacket:
    return FramePacket(
        source_id=source_id,
        sequence_number=seq,
        timestamp_ns=ts_ns,
        host_monotonic_ns=ts_ns,
        image_bytes=b"FRAME_DATA_" + str(seq).encode("ascii"),
        pixel_format="GRAY8",
        resolution_px=(16, 16),
    )


def test_frame_index_entry_contracts() -> None:
    entry = FrameIndexEntry(
        stream_id="cam-01",
        sequence_number=1,
        timestamp_ns=100_000_000,
        byte_offset=0,
        payload_bytes=1024,
        checksum_crc32="1a2b3c4d",
    )
    assert entry.stream_id == "cam-01"
    assert entry.sequence_number == 1
    assert entry.timestamp_ns == 100_000_000
    assert entry.byte_offset == 0
    assert entry.payload_bytes == 1024
    assert entry.checksum_crc32 == "1a2b3c4d"

    with pytest.raises(
        ValueError, match="checksum_crc32 must be 8 hexadecimal characters"
    ):
        FrameIndexEntry(
            stream_id="cam-01",
            sequence_number=1,
            timestamp_ns=100_000_000,
            byte_offset=0,
            payload_bytes=1024,
            checksum_crc32="invalid",
        )


def test_recording_writer_enforces_policy_and_crash_safe_manifest(
    tmp_path: Path,
) -> None:
    session_dir = tmp_path / "test_session"
    manifest = _manifest(consent=True, no_store=False, raw_retained=True)

    writer = RecordingWriter(session_dir=session_dir, manifest=manifest)
    writer.initialize()

    # Verify atomic manifest was written
    manifest_path = session_dir / "session_manifest.json"
    assert manifest_path.is_file()

    # Write frames
    for i in range(1, 6):
        packet = _make_packet("cam-01", i, i * 33_333_333)
        writer.write_frame(packet)

    writer.finalize()

    # Ensure index exists and manifest transitioned to FINALIZED
    index_path = session_dir / "index.jsonl"
    assert index_path.is_file()

    reader = RecordingReader(session_dir=session_dir)
    final_manifest = reader.read_manifest()
    assert final_manifest.state == SessionState.FINALIZED

    # Read frames back and verify count and checksums
    frames = list(reader.read_frames())
    assert len(frames) == 5
    assert frames[0].sequence_number == 1
    assert frames[4].sequence_number == 5

    report = reader.verify_integrity()
    assert report.is_valid
    assert report.total_frames == 5
    assert report.corrupt_frames == 0
    assert report.missing_frames == 0


def test_recording_writer_enforces_no_store_policy(tmp_path: Path) -> None:
    session_dir = tmp_path / "test_nostore_session"
    manifest = _manifest(consent=True, no_store=True, raw_retained=False)

    writer = RecordingWriter(session_dir=session_dir, manifest=manifest)
    writer.initialize()

    packet = _make_packet("cam-01", 1, 33_333_333)
    # Writing a packet under no_store must reject raw image retention
    with pytest.raises(
        ValueError, match="no_store policy forbids writing raw image bytes"
    ):
        writer.write_frame(packet)


def test_recording_reader_detects_corrupt_chunks(tmp_path: Path) -> None:
    session_dir = tmp_path / "corrupt_session"
    manifest = _manifest(consent=True, no_store=False, raw_retained=True)

    writer = RecordingWriter(session_dir=session_dir, manifest=manifest)
    writer.initialize()
    for i in range(1, 4):
        writer.write_frame(_make_packet("cam-01", i, i * 33_333_333))
    writer.finalize()

    # Corrupt the payload file
    payload_file = session_dir / "streams" / "cam-01.bin"
    assert payload_file.is_file()
    with open(payload_file, "r+b") as f:
        f.seek(5)
        f.write(b"CORRUPT")

    reader = RecordingReader(session_dir=session_dir)
    report = reader.verify_integrity()
    assert not report.is_valid
    assert report.corrupt_frames > 0
