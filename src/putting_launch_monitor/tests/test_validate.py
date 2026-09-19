"""Tests for the accuracy validation harness and stats tracking."""

from __future__ import annotations

import csv
import io
from pathlib import Path

import pytest

from putting_launch_monitor.calibration import Calibration
from putting_launch_monitor.monitor import LogSink, PuttingMonitor
from putting_launch_monitor.tests.synthetic import (
    MAT_L_MM,
    MAT_W_MM,
    SyntheticCamera,
    putt_positions,
    render_frame,
)
from putting_launch_monitor.validate import (
    CSV_HEADER,
    RunningStats,
    ValidationHarness,
    ValidationRecord,
)
from shared.python.contracts import StateError
from shared.python.sidekick.lab.mocap.acquisition import FramePacket, FrameSource
from shared.python.sidekick.lab.mocap.devices import (
    CameraCapabilities,
    CameraIdentity,
    FeatureSupport,
)
from shared.python.sidekick.lab.mocap.enums import ShutterKind, SupportLevel

pytest.importorskip("cv2")
pytestmark = pytest.mark.unit


def _mock_calibration(cam: SyntheticCamera) -> Calibration:
    corners = tuple((float(x), float(y)) for x, y in cam.mat_corners_px())
    return Calibration(
        camera_instance_id="synthetic:validate_test",
        mat_corners_px=corners,
        mat_width_mm=MAT_W_MM,
        mat_length_mm=MAT_L_MM,
        capture_width=cam.width,
        capture_height=cam.height,
        fps=60,
    )


class _SyntheticStream(FrameSource):
    def __init__(
        self,
        cam: SyntheticCamera,
        positions: list[tuple[float, float] | None],
        fps: float = 60.0,
    ) -> None:
        self.cam = cam
        self.positions = positions
        self.fps = fps
        self.i = 0

    source_id = "synthetic:validate"
    identity = CameraIdentity(
        provider_id="synthetic", device_id="validate", transport="memory"
    )
    capabilities = CameraCapabilities(
        resolutions_px=((960, 600),),
        frame_rates_hz=(60.0,),
        pixel_formats=("bgr24",),
        shutter=ShutterKind.GLOBAL,
        hardware_trigger=FeatureSupport(SupportLevel.UNSUPPORTED, "synthetic"),
        device_timestamps=FeatureSupport(SupportLevel.SUPPORTED),
    )

    @property
    def state(self) -> object:
        return "capturing"

    def initialize(self) -> None: ...
    def start_capture(self) -> None: ...
    def stop_capture(self) -> None: ...
    def close(self) -> None: ...

    def read_frame(self, timeout_seconds: float = 5.0) -> FramePacket:
        if self.i >= len(self.positions):
            raise StateError("end of stream")
        pos = self.positions[self.i]
        frame = render_frame(self.cam, pos, noise=2.0, seed=self.i)
        pkt = FramePacket(
            source_id=self.source_id,
            sequence_number=self.i,
            timestamp_ns=int(self.i * 1e9 / self.fps),
            host_monotonic_ns=0,
            image_bytes=frame.tobytes(),
            pixel_format="bgr24",
            resolution_px=(self.cam.width, self.cam.height),
        )
        self.i += 1
        return pkt


def _make_putt_positions(
    speed_mps: float, hla_deg: float
) -> list[tuple[float, float] | None]:
    start_mm = (610.0, 250.0)
    rest = [start_mm] * 18
    _, world = putt_positions(start_mm, speed_mps, hla_deg, fps=60.0, seconds=0.8)
    roll = [(float(p[0]), float(p[1])) for p in world]
    return rest + roll + [None] * 10


def test_validation_record_and_errors() -> None:
    rec = ValidationRecord(
        timestamp_iso="2026-09-19T20:00:00Z",
        speed_mph=4.0,
        hla_deg=1.5,
        points=15,
        r2=0.995,
        span_mm=300.0,
        ref_speed_mph=3.8,
        ref_hla_deg=1.0,
        accepted=True,
    )
    assert rec.speed_err_pct is not None
    assert abs(rec.speed_err_pct - (4.0 - 3.8) / 3.8 * 100.0) < 1e-4
    assert rec.hla_err_deg is not None
    assert abs(rec.hla_err_deg - 0.5) < 1e-4

    row = rec.to_csv_row()
    assert len(row) == len(CSV_HEADER)
    assert row[0] == "2026-09-19T20:00:00Z"
    assert row[1] == "4.00"
    assert row[2] == "+1.50"
    assert row[10] == "true"


def test_running_stats_computations() -> None:
    stats = RunningStats()
    assert stats.summary().startswith("Putts: 0")
    assert stats.mean_speed_err_pct is None

    r1 = ValidationRecord(
        timestamp_iso="t1",
        speed_mph=4.0,
        hla_deg=0.5,
        points=12,
        r2=0.99,
        span_mm=250.0,
        ref_speed_mph=4.0,
        ref_hla_deg=0.0,
        accepted=True,
    )
    r2 = ValidationRecord(
        timestamp_iso="t2",
        speed_mph=2.1,
        hla_deg=-0.4,
        points=14,
        r2=0.99,
        span_mm=280.0,
        ref_speed_mph=2.0,
        ref_hla_deg=0.0,
        accepted=True,
    )
    r_rej = ValidationRecord(
        timestamp_iso="t3",
        speed_mph=1.0,
        hla_deg=0.0,
        points=3,
        r2=0.8,
        span_mm=50.0,
        ref_speed_mph=None,
        ref_hla_deg=None,
        accepted=False,
        reason="twitch",
    )

    stats.update(r1)
    stats.update(r2)
    stats.update(r_rej)

    assert stats.total_putts == 3
    assert stats.accepted_putts == 2
    assert stats.rejected_putts == 1

    # r1 err: 0%, r2 err: (2.1 - 2.0)/2.0 = 5.0% -> mean 2.5%, MAE 2.5%
    assert stats.mean_speed_err_pct is not None
    assert abs(stats.mean_speed_err_pct - 2.5) < 1e-4
    # r1 HLA: +0.5, r2 HLA: -0.4 -> mean +0.05, MAE 0.45
    assert stats.mean_hla_err_deg is not None
    assert abs(stats.mean_hla_err_deg - 0.05) < 1e-4
    assert stats.mae_hla_err_deg is not None
    assert abs(stats.mae_hla_err_deg - 0.45) < 1e-4

    assert not stats.meets_acceptance(speed_tol_pct=3.0, hla_tol_deg=1.0)
    assert stats.meets_acceptance(speed_tol_pct=6.0, hla_tol_deg=1.0)


def test_validation_harness_interactive_session(tmp_path: Path) -> None:
    csv_path = tmp_path / "session.csv"
    cam = SyntheticCamera()
    cal = _mock_calibration(cam)
    positions = _make_putt_positions(speed_mps=1.8, hla_deg=-1.0)
    stream = _SyntheticStream(cam, positions)
    sink = LogSink()
    monitor = PuttingMonitor(cal, stream, sink)

    simulated_stdin = io.StringIO("4.0\n-1.0\noperator test 1\n")
    simulated_stdout = io.StringIO()

    harness = ValidationHarness(
        monitor,
        csv_path=csv_path,
        input_stream=simulated_stdin,
        output_stream=simulated_stdout,
        interactive=True,
    )
    frames = harness.run()
    assert frames > 0
    assert harness.stats.accepted_putts == 1
    assert len(harness.records) == 1

    # Verify CSV file contents
    assert csv_path.is_file()
    with open(csv_path, encoding="utf-8") as f:
        reader = list(csv.reader(f))
    assert len(reader) == 2  # Header + 1 record
    assert reader[0] == CSV_HEADER
    row = reader[1]
    assert row[6] == "4.00"  # Ref speed
    assert row[7] == "-1.00"  # Ref HLA
    assert row[10] == "true"  # Accepted
    assert row[12] == "operator test 1"  # Note


def test_validation_harness_fixed_reference_and_max_putts(tmp_path: Path) -> None:
    csv_path = tmp_path / "fixed_session.csv"
    cam = SyntheticCamera()
    cal = _mock_calibration(cam)
    # Two putts in the stream
    p1 = _make_putt_positions(speed_mps=1.5, hla_deg=0.0)
    p2 = _make_putt_positions(speed_mps=2.0, hla_deg=1.0)
    stream = _SyntheticStream(cam, p1 + p2)
    sink = LogSink()
    monitor = PuttingMonitor(cal, stream, sink)

    harness = ValidationHarness(
        monitor,
        csv_path=csv_path,
        output_stream=io.StringIO(),
        fixed_ref_speed=3.36,
        fixed_ref_hla=0.0,
        interactive=False,
        max_putts=1,
    )
    harness.run()
    # Stopped after 1 putt because max_putts=1
    assert harness.stats.accepted_putts == 1


def test_cli_validate_replay_video(tmp_path: Path) -> None:
    import argparse

    import cv2

    from putting_launch_monitor.cli import cmd_validate

    cam = SyntheticCamera()
    cal = _mock_calibration(cam)
    cal_path = tmp_path / "cal.json"
    cal.save(cal_path)

    video_path = tmp_path / "test_putt.avi"
    positions = _make_putt_positions(speed_mps=1.8, hla_deg=-1.0)
    fourcc = int(cv2.VideoWriter.fourcc(*"MJPG"))
    writer = cv2.VideoWriter(str(video_path), fourcc, 60.0, (cam.width, cam.height))
    for i, pos in enumerate(positions):
        writer.write(render_frame(cam, pos, noise=2.0, seed=i))
    writer.release()

    csv_out = tmp_path / "cli_val.csv"
    args = argparse.Namespace(
        calibration=cal_path,
        video=video_path,
        camera=None,
        fps=60.0,
        width=960,
        out=csv_out,
        ref_speed=4.0,
        ref_hla=-1.0,
        max_putts=1,
        non_interactive=True,
        gspro=False,
    )
    code = cmd_validate(args)
    assert code == 0
    assert csv_out.is_file()
    with open(csv_out, encoding="utf-8") as f:
        rows = list(csv.reader(f))
    assert len(rows) == 2
    assert rows[1][10] == "true"
