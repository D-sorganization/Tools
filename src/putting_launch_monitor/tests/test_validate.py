"""Unit tests for the putting monitor validation harness."""

from __future__ import annotations

import argparse
import io
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from putting_launch_monitor.calibration import Calibration
from putting_launch_monitor.geometry import Launch
from putting_launch_monitor.monitor import FrameEvent, PuttingMonitor
from putting_launch_monitor.track import Phase, Putt
from putting_launch_monitor.validate import (
    CSV_HEADER,
    RunningStats,
    ValidationHarness,
    ValidationRecord,
    add_validate_parser,
    cmd_validate,
)

from .synthetic import MAT_L_MM, MAT_W_MM, SyntheticCamera
from .test_pipeline import RenderedPuttSource

pytest.importorskip("cv2")
pytestmark = pytest.mark.unit

CAMERA_ID = "USB\\VID_32E4&PID_5234&MI_00\\9&2A7EE39F&0&0000"


def _make_calibration(cam: SyntheticCamera) -> Calibration:
    corners = tuple((float(x), float(y)) for x, y in cam.mat_corners_px())
    return Calibration(
        camera_instance_id=CAMERA_ID,
        mat_corners_px=corners,
        mat_width_mm=MAT_W_MM,
        mat_length_mm=MAT_L_MM,
        capture_width=cam.width,
        capture_height=cam.height,
    )


def test_validation_record_contracts_and_csv_row() -> None:
    rec = ValidationRecord(
        timestamp="2026-09-20T06:00:00Z",
        speed_mph=2.54,
        hla_deg=-0.45,
        points=12,
        r2=0.998,
        span_mm=295.4,
        ref_speed_mph=2.50,
        ref_hla_deg=-0.50,
        note="ramp roll 1",
    )
    row = rec.to_csv_row()
    assert len(row) == len(CSV_HEADER)
    assert row[0] == "2026-09-20T06:00:00Z"
    assert row[1] == "2.540"
    assert row[2] == "-0.45"
    assert row[3] == "12"
    assert row[4] == "0.9980"
    assert row[5] == "295.4"
    assert row[6] == "2.500"
    assert row[7] == "-0.50"
    assert row[8] == "ramp roll 1"

    with pytest.raises(Exception, match="speed_mph"):
        ValidationRecord("now", float("nan"), 0.0, 5, 0.99, 300.0)

    with pytest.raises(ValueError, match="points >= 0"):
        ValidationRecord("now", 2.0, 0.0, -1, 0.99, 300.0)


def test_running_stats_computations_and_tolerances() -> None:
    stats = RunningStats()
    assert stats.count == 0
    assert "no reference speed" in stats.format_summary()
    assert stats.is_within_tolerance(3.0, 1.0)

    # Add putt 1: ref 2.50 mph, measured 2.55 mph (+2.0% error);
    # ref 0.0 deg, measured +0.4 deg (+0.4 deg error)
    stats.add(
        ValidationRecord(
            "t1",
            speed_mph=2.55,
            hla_deg=0.40,
            points=10,
            r2=0.99,
            span_mm=250.0,
            ref_speed_mph=2.50,
            ref_hla_deg=0.0,
        )
    )
    assert stats.count == 1
    assert stats.speed_pct_errors[0] == pytest.approx(2.0)
    assert stats.hla_deg_errors[0] == pytest.approx(0.40)
    assert stats.is_within_tolerance(3.0, 1.0)

    # Add putt 2: ref 2.50 mph, measured 2.45 mph (-2.0% error);
    # ref 0.0 deg, measured -0.2 deg (-0.2 deg error)
    stats.add(
        ValidationRecord(
            "t2",
            speed_mph=2.45,
            hla_deg=-0.20,
            points=10,
            r2=0.99,
            span_mm=250.0,
            ref_speed_mph=2.50,
            ref_hla_deg=0.0,
        )
    )
    assert stats.count == 2
    # Mean speed error = 0%, std = 2.0%
    summary = stats.format_summary()
    assert "mean=+0.00%" in summary
    assert "std=2.00%" in summary
    assert "mean=+0.10°" in summary
    assert stats.is_within_tolerance(3.0, 1.0)

    # Add putt 3 that violates speed tolerance (+4.0%)
    stats.add(
        ValidationRecord(
            "t3",
            speed_mph=2.60,
            hla_deg=0.0,
            points=10,
            r2=0.99,
            span_mm=250.0,
            ref_speed_mph=2.50,
            ref_hla_deg=0.0,
        )
    )
    assert not stats.is_within_tolerance(max_speed_pct=3.0)


def test_validation_harness_interactive_input_and_csv() -> None:
    buf = io.StringIO()
    inputs = iter(["2.5 -0.5 straight roll", "skip", "q"])

    harness = ValidationHarness(
        buf,
        default_ref_speed_mph=2.0,
        default_ref_hla_deg=0.0,
        auto=False,
        input_fn=lambda prompt: next(inputs),
    )

    dummy_launch = Launch(
        speed_mps=1.1176,
        direction=np.array([0.0, 1.0]),
        points=10,
        r2=0.995,
        span_mm=280.0,
    )
    putt = Putt(
        launch=dummy_launch,
        hla_deg=-0.48,
        start_mm=(610.0, 250.0),
        accepted=True,
    )
    event1 = FrameEvent(
        sequence=1,
        timestamp_ns=0,
        seen=None,
        phase=Phase.WAITING,
        putt=putt,
        outcome="accepted",
    )

    harness.on_frame_event(event1)
    assert harness.stats.count == 1
    assert harness.stats.records[0].ref_speed_mph == 2.5
    assert harness.stats.records[0].ref_hla_deg == -0.5
    assert harness.stats.records[0].note == "straight roll"

    # Second putt: skipped
    event2 = FrameEvent(
        sequence=2,
        timestamp_ns=1000,
        seen=None,
        phase=Phase.WAITING,
        putt=putt,
        outcome="accepted",
    )
    harness.on_frame_event(event2)
    assert harness.stats.count == 1  # unchanged because of skip

    # Third putt: quits
    with pytest.raises(StopIteration, match="terminated by operator"):
        harness.on_frame_event(event2)

    csv_content = buf.getvalue().splitlines()
    assert csv_content[0].startswith("timestamp,speed_mph,hla_deg")
    assert len(csv_content) == 2  # header + 1 record


def test_validation_harness_rejects_and_counts() -> None:
    buf = io.StringIO()
    harness = ValidationHarness(buf, auto=True)

    dummy_launch = Launch(
        speed_mps=0.1,
        direction=np.array([0.0, 1.0]),
        points=3,
        r2=0.85,
        span_mm=50.0,
    )
    rejected_putt = Putt(
        launch=dummy_launch,
        hla_deg=0.0,
        start_mm=(610.0, 250.0),
        accepted=False,
        reason="low_points",
    )
    event = FrameEvent(
        sequence=1,
        timestamp_ns=0,
        seen=None,
        phase=Phase.WAITING,
        putt=rejected_putt,
        outcome="rejected",
    )

    harness.on_frame_event(event)
    assert harness.stats.count == 0
    assert harness.rejected_count == 1


def test_validation_harness_with_rendered_putt_auto(tmp_path: Path) -> None:
    cam = SyntheticCamera()
    cal = _make_calibration(cam)
    # 2.5 mph ~ 1.1176 m/s, HLA = +0.5 deg
    ref_speed_mph = 2.50
    ref_hla_deg = 0.50
    speed_mps = ref_speed_mph / 2.2369362920544

    source = RenderedPuttSource(cam, speed=speed_mps, hla=ref_hla_deg)
    csv_file = tmp_path / "val.csv"

    records: list[ValidationRecord] = []
    with csv_file.open("w", newline="", encoding="utf-8") as f:
        harness = ValidationHarness(
            f,
            default_ref_speed_mph=ref_speed_mph,
            default_ref_hla_deg=ref_hla_deg,
            auto=True,
            on_record=records.append,
        )
        monitor = PuttingMonitor(cal, source)
        monitor.add_observer(harness.on_frame_event)
        frames = monitor.run()

    assert frames > 20
    assert len(records) == 1
    rec = records[0]
    # Check tolerance criteria: speed +/-3%, HLA +/-1 deg
    speed_err_pct = abs(100.0 * (rec.speed_mph - ref_speed_mph) / ref_speed_mph)
    hla_err_deg = abs(rec.hla_deg - ref_hla_deg)
    assert speed_err_pct < 3.0, f"Speed error {speed_err_pct:.2f}% exceeds 3%"
    assert hla_err_deg < 1.0, f"HLA error {hla_err_deg:.2f}° exceeds 1°"
    assert harness.stats.is_within_tolerance(3.0, 1.0)


def test_add_validate_parser() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command")
    add_validate_parser(sub)
    args = parser.parse_args(["validate", "--ref-speed", "3.0", "--auto"])
    assert args.command == "validate"
    assert args.ref_speed == 3.0
    assert args.auto is True


def test_cmd_validate_execution(tmp_path: Path, monkeypatch: Any) -> None:
    cam = SyntheticCamera()
    cal = _make_calibration(cam)
    cal_path = tmp_path / "cal.json"
    cal.save(cal_path)

    csv_path = tmp_path / "test_out.csv"

    # Run cmd_validate using a scripted stop
    args = argparse.Namespace(
        calibration=cal_path,
        video=None,
        camera=CAMERA_ID,
        fps=60,
        width=960,
        csv=csv_path,
        ref_speed=2.5,
        ref_hla=0.0,
        max_putts=1,
        max_frames=1,
        auto=True,
    )

    class DummyFramePacket:
        pixel_format = "bgr24"
        resolution_px = (960, 600)
        sequence_number = 0
        timestamp_ns = 0
        image_bytes = np.zeros((600, 960, 3), dtype=np.uint8).tobytes()

    class MockSource:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.i = 0

        def initialize(self) -> None: ...
        def start_capture(self) -> None: ...
        def read_frame(self) -> DummyFramePacket:
            self.i += 1
            if self.i > 1:
                from shared.python.contracts import StateError

                raise StateError("end")
            return DummyFramePacket()

        def close(self) -> None: ...

    monkeypatch.setattr(
        "shared.python.camera.FfmpegDirectShowSource",
        MockSource,
    )

    code = cmd_validate(args)
    assert code == 0
    assert csv_path.exists()
    assert "timestamp,speed_mph" in csv_path.read_text(encoding="utf-8")
