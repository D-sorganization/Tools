"""Putting launch monitor accuracy validation harness.

Runs the putting monitor on a live camera or recorded video, records each
accepted putt's metrics alongside operator reference values into a CSV, and
computes running statistics (mean, spread, standard deviation) against target
tolerances (+/-3% speed, +/-1 deg HLA at 1-4 mph).
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TextIO, cast

from shared.python.contracts import require, require_finite

from .calibration import Calibration, default_calibration_path
from .monitor import FrameEvent, LogSink, PuttingMonitor
from .track import Putt

logger = logging.getLogger("putting_launch_monitor.validate")

CSV_HEADER: Sequence[str] = (
    "timestamp",
    "speed_mph",
    "hla_deg",
    "points",
    "r2",
    "span_mm",
    "ref_speed_mph",
    "ref_hla_deg",
    "note",
)


@dataclass(frozen=True)
class ValidationRecord:
    """A single validation observation pairing measured and reference metrics."""

    timestamp: str
    speed_mph: float
    hla_deg: float
    points: int
    r2: float
    span_mm: float
    ref_speed_mph: float | None = None
    ref_hla_deg: float | None = None
    note: str = ""

    def __post_init__(self) -> None:
        require_finite(self.speed_mph, "speed_mph")
        require_finite(self.hla_deg, "hla_deg")
        require_finite(self.r2, "r2")
        require_finite(self.span_mm, "span_mm")
        require(self.points >= 0, "points >= 0", self.points)

    def to_csv_row(self) -> list[str]:
        """Format as a list of strings matching CSV_HEADER."""
        ref_s = f"{self.ref_speed_mph:.3f}" if self.ref_speed_mph is not None else ""
        ref_h = f"{self.ref_hla_deg:+.2f}" if self.ref_hla_deg is not None else ""
        return [
            self.timestamp,
            f"{self.speed_mph:.3f}",
            f"{self.hla_deg:+.2f}",
            str(self.points),
            f"{self.r2:.4f}",
            f"{self.span_mm:.1f}",
            ref_s,
            ref_h,
            self.note,
        ]


@dataclass
class RunningStats:
    """Accumulates errors and calculates running statistics."""

    speed_pct_errors: list[float] = field(default_factory=list)
    hla_deg_errors: list[float] = field(default_factory=list)
    records: list[ValidationRecord] = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.records)

    def add(self, record: ValidationRecord) -> None:
        """Incorporate a new record into running stats."""
        self.records.append(record)
        if record.ref_speed_mph is not None and record.ref_speed_mph > 0:
            err_mph = record.speed_mph - record.ref_speed_mph
            pct = 100.0 * err_mph / record.ref_speed_mph
            self.speed_pct_errors.append(pct)
        if record.ref_hla_deg is not None:
            self.hla_deg_errors.append(record.hla_deg - record.ref_hla_deg)

    @staticmethod
    def _mean_std_min_max(values: list[float]) -> tuple[float, float, float, float]:
        if not values:
            return 0.0, 0.0, 0.0, 0.0
        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / len(values)
        std = math.sqrt(variance)
        return mean, std, min(values), max(values)

    def format_summary(self) -> str:
        """Return formatted multi-line summary of running statistics."""
        lines = [f"=== Validation Running Statistics (N={self.count}) ==="]
        if self.speed_pct_errors:
            m, s, lo, hi = self._mean_std_min_max(self.speed_pct_errors)
            lines.append(
                f"Speed Error (%): mean={m:+.2f}%  std={s:.2f}%  "
                f"spread=[{lo:+.2f}%, {hi:+.2f}%]  (tol: +/-3.0%)"
            )
        else:
            lines.append("Speed Error (%): no reference speed provided")

        if self.hla_deg_errors:
            m, s, lo, hi = self._mean_std_min_max(self.hla_deg_errors)
            lines.append(
                f"HLA Error (deg): mean={m:+.2f}°  std={s:.2f}°  "
                f"spread=[{lo:+.2f}°, {hi:+.2f}°]  (tol: +/-1.0°)"
            )
        else:
            lines.append("HLA Error (deg): no reference HLA provided")
        return "\n".join(lines)

    def is_within_tolerance(
        self, max_speed_pct: float = 3.0, max_hla_deg: float = 1.0
    ) -> bool:
        """Check if all recorded observations with reference values meet tolerance."""
        for err in self.speed_pct_errors:
            if abs(err) > max_speed_pct:
                return False
        for err in self.hla_deg_errors:
            if abs(err) > max_hla_deg:
                return False
        return True


class ValidationHarness:
    """Observer and CSV recorder for putting accuracy validation."""

    def __init__(
        self,
        csv_file: TextIO,
        *,
        default_ref_speed_mph: float | None = None,
        default_ref_hla_deg: float | None = None,
        auto: bool = False,
        input_fn: Callable[[str], str] | None = None,
        on_record: Callable[[ValidationRecord], None] | None = None,
    ) -> None:
        self.csv_file = csv_file
        self.ref_speed = default_ref_speed_mph
        self.ref_hla = default_ref_hla_deg
        self.auto = auto
        self.input_fn = input_fn or input
        self.on_record = on_record
        self.stats = RunningStats()
        self.csv_writer = csv.writer(csv_file)
        self.rejected_count = 0
        self._ensure_header()

    def _ensure_header(self) -> None:
        if self.csv_file.tell() == 0:
            self.csv_writer.writerow(CSV_HEADER)
            self.csv_file.flush()

    def on_frame_event(self, event: FrameEvent) -> None:
        """Observer callback invoked by PuttingMonitor on each frame event."""
        if event.putt is None:
            return
        putt: Putt = event.putt
        if not putt.accepted:
            self.rejected_count += 1
            logger.warning(
                "Putt #%d REJECTED: %s (speed=%.2f mph, HLA=%+.2f deg)",
                self.rejected_count,
                putt.reason,
                putt.speed_mph,
                putt.hla_deg,
            )
            return

        ref_s = self.ref_speed
        ref_h = self.ref_hla
        note = ""

        if not self.auto:
            prompt = (
                f"\n[PUTT DETECTED] {putt.speed_mph:.2f} mph, "
                f"HLA {putt.hla_deg:+.2f}° "
                f"({putt.launch.points} pts, r2={putt.launch.r2:.3f})\n"
                f"Enter reference [speed_mph [hla_deg] [note]] "
                f"(Enter to use [{ref_s} mph, {ref_h}°], "
                f"'skip' to discard, 'q' to stop): "
            )
            try:
                line = self.input_fn(prompt).strip()
            except (EOFError, KeyboardInterrupt):
                line = "q"

            if line.lower() in ("q", "quit"):
                raise StopIteration("validation terminated by operator")
            if line.lower() == "skip":
                logger.info("Discarded putt without recording")
                return

            if line:
                tokens = line.split(maxsplit=2)
                if len(tokens) >= 1:
                    ref_s = float(tokens[0])
                if len(tokens) >= 2:
                    ref_h = float(tokens[1])
                if len(tokens) >= 3:
                    note = tokens[2]

        record = ValidationRecord(
            timestamp=datetime.now(UTC).isoformat(),
            speed_mph=putt.speed_mph,
            hla_deg=putt.hla_deg,
            points=putt.launch.points,
            r2=putt.launch.r2,
            span_mm=putt.launch.span_mm,
            ref_speed_mph=ref_s,
            ref_hla_deg=ref_h,
            note=note,
        )
        self.stats.add(record)
        self.csv_writer.writerow(record.to_csv_row())
        self.csv_file.flush()

        logger.info(
            "Recorded putt #%d: %.2f mph (ref=%s), HLA %+.2f° (ref=%s)",
            self.stats.count,
            record.speed_mph,
            f"{ref_s:.2f}" if ref_s is not None else "None",
            record.hla_deg,
            f"{ref_h:+.2f}" if ref_h is not None else "None",
        )
        logger.info("\n%s\n", self.stats.format_summary())

        if self.on_record is not None:
            self.on_record(record)


def add_validate_parser(sub: Any) -> argparse.ArgumentParser:
    """Add the 'validate' subcommand parser."""
    val = cast(
        argparse.ArgumentParser,
        sub.add_parser(
            "validate", help="accuracy validation harness against known reference putts"
        ),
    )
    val.add_argument(
        "--calibration",
        type=Path,
        default=None,
        help="path to calibration JSON (default: per-user calibration)",
    )
    val.add_argument("--camera", default=None, help="PnP instance id of camera")
    val.add_argument(
        "--video", type=Path, default=None, help="video file for replay validation"
    )
    val.add_argument("--fps", type=float, default=None, help="override camera rate")
    val.add_argument(
        "--width", type=int, default=960, help="decode width (default: 960)"
    )
    val.add_argument(
        "--csv",
        type=Path,
        default=Path("putting_validation.csv"),
        help="output CSV file path",
    )
    val.add_argument(
        "--ref-speed", type=float, default=None, help="default reference speed (mph)"
    )
    val.add_argument(
        "--ref-hla", type=float, default=None, help="default reference HLA (deg)"
    )
    val.add_argument(
        "--max-putts", type=int, default=None, help="stop after N recorded putts"
    )
    val.add_argument(
        "--max-frames", type=int, default=None, help="stop after N processed frames"
    )
    val.add_argument(
        "--auto",
        action="store_true",
        help="non-interactive: record putts using default reference values",
    )
    return val


def cmd_validate(
    args: argparse.Namespace, *, input_fn: Callable[[str], str] | None = None
) -> int:
    """Execute the validate subcommand."""
    from shared.python.camera import (
        CaptureMode,
        FfmpegDirectShowSource,
        VideoFileSource,
    )

    from .cli import scaled_calibration

    cal_path = args.calibration or default_calibration_path()
    cal = Calibration.load(cal_path)

    if args.video is not None:
        source = VideoFileSource(args.video, fps=args.fps)
        active_cal = cal
    else:
        cam_id = args.camera or cal.camera_instance_id
        fps = int(args.fps) if args.fps is not None else cal.fps
        mode = CaptureMode(width=cal.capture_width, height=cal.capture_height, fps=fps)
        source = FfmpegDirectShowSource(cam_id, mode, width=args.width)
        active_cal = scaled_calibration(cal, args.width)

    csv_path: Path = args.csv
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    with csv_path.open("a+", newline="", encoding="utf-8") as f:
        harness = ValidationHarness(
            f,
            default_ref_speed_mph=args.ref_speed,
            default_ref_hla_deg=args.ref_hla,
            auto=args.auto,
            input_fn=input_fn,
        )

        monitor = PuttingMonitor(active_cal, source, LogSink())

        def _on_event(event: FrameEvent) -> None:
            try:
                harness.on_frame_event(event)
            except StopIteration:
                monitor.stop()
            if args.max_putts and harness.stats.count >= args.max_putts:
                monitor.stop()

        monitor.add_observer(_on_event)
        logger.info(
            "Validation harness active. Writing observations to %s", csv_path.resolve()
        )
        monitor.run(max_frames=args.max_frames)

    logger.info("Validation finished. %d putts recorded.", harness.stats.count)
    logger.info("\n%s\n", harness.stats.format_summary())
    return 0
