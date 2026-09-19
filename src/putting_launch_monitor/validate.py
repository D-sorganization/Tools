"""Accuracy validation harness for the putting launch monitor.

Records each detected putt alongside operator-entered or fixed reference values,
appends results to a CSV file, and maintains running error statistics.
Used on the capture rig or against recorded benchmark videos to qualify speed
and HLA accuracy against acceptance gates.
"""

from __future__ import annotations

import csv
import logging
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TextIO

from shared.python.contracts import require

from .monitor import FrameEvent, PuttingMonitor
from .track import Putt

logger = logging.getLogger("putting_launch_monitor.validate")

CSV_HEADER = [
    "timestamp",
    "speed_mph",
    "hla_deg",
    "points",
    "r2",
    "span_mm",
    "ref_speed_mph",
    "ref_hla_deg",
    "speed_err_pct",
    "hla_err_deg",
    "accepted",
    "reason",
    "note",
]


@dataclass(frozen=True)
class ValidationRecord:
    """One validation row with measured metrics and optional reference values."""

    timestamp_iso: str
    speed_mph: float
    hla_deg: float
    points: int
    r2: float
    span_mm: float
    ref_speed_mph: float | None
    ref_hla_deg: float | None
    accepted: bool
    reason: str = ""
    note: str = ""

    def __post_init__(self) -> None:
        require(bool(self.timestamp_iso), "timestamp cannot be empty")
        require(self.points >= 0, "points must be non-negative", self.points)
        require(self.span_mm >= 0, "span must be non-negative", self.span_mm)

    @property
    def speed_err_pct(self) -> float | None:
        """Percentage error: (measured - reference) / reference * 100."""
        if self.ref_speed_mph is None or self.ref_speed_mph <= 0:
            return None
        return float((self.speed_mph - self.ref_speed_mph) / self.ref_speed_mph * 100.0)

    @property
    def hla_err_deg(self) -> float | None:
        """Signed angle error in degrees: measured - reference."""
        if self.ref_hla_deg is None:
            return None
        return float(self.hla_deg - self.ref_hla_deg)

    def to_csv_row(self) -> list[str]:
        speed_err = (
            f"{self.speed_err_pct:+.2f}" if self.speed_err_pct is not None else ""
        )
        hla_err = f"{self.hla_err_deg:+.2f}" if self.hla_err_deg is not None else ""
        ref_speed = (
            f"{self.ref_speed_mph:.2f}" if self.ref_speed_mph is not None else ""
        )
        ref_hla = f"{self.ref_hla_deg:+.2f}" if self.ref_hla_deg is not None else ""
        return [
            self.timestamp_iso,
            f"{self.speed_mph:.2f}",
            f"{self.hla_deg:+.2f}",
            str(self.points),
            f"{self.r2:.3f}",
            f"{self.span_mm:.1f}",
            ref_speed,
            ref_hla,
            speed_err,
            hla_err,
            "true" if self.accepted else "false",
            self.reason,
            self.note,
        ]


@dataclass
class RunningStats:
    """Tracks running statistics over a validation session."""

    total_putts: int = 0
    accepted_putts: int = 0
    rejected_putts: int = 0
    speed_errors_pct: list[float] = field(default_factory=list)
    hla_errors_deg: list[float] = field(default_factory=list)

    def update(self, record: ValidationRecord) -> None:
        self.total_putts += 1
        if record.accepted:
            self.accepted_putts += 1
            if record.speed_err_pct is not None:
                self.speed_errors_pct.append(record.speed_err_pct)
            if record.hla_err_deg is not None:
                self.hla_errors_deg.append(record.hla_err_deg)
        else:
            self.rejected_putts += 1

    @property
    def mean_speed_err_pct(self) -> float | None:
        if not self.speed_errors_pct:
            return None
        return float(sum(self.speed_errors_pct) / len(self.speed_errors_pct))

    @property
    def mae_speed_err_pct(self) -> float | None:
        if not self.speed_errors_pct:
            return None
        total = sum(abs(e) for e in self.speed_errors_pct)
        return float(total / len(self.speed_errors_pct))

    @property
    def mean_hla_err_deg(self) -> float | None:
        if not self.hla_errors_deg:
            return None
        return float(sum(self.hla_errors_deg) / len(self.hla_errors_deg))

    @property
    def mae_hla_err_deg(self) -> float | None:
        if not self.hla_errors_deg:
            return None
        total = sum(abs(e) for e in self.hla_errors_deg)
        return float(total / len(self.hla_errors_deg))

    def summary(self) -> str:
        base = (
            f"Putts: {self.total_putts} ({self.accepted_putts} accepted, "
            f"{self.rejected_putts} rejected)"
        )
        if not self.speed_errors_pct and not self.hla_errors_deg:
            return base
        speed_s = (
            f"Speed MAE: {self.mae_speed_err_pct:.2f}% "
            f"(mean {self.mean_speed_err_pct:+.2f}%)"
            if self.mae_speed_err_pct is not None
            else "Speed: no ref"
        )
        hla_s = (
            f"HLA MAE: {self.mae_hla_err_deg:.2f}° (mean {self.mean_hla_err_deg:+.2f}°)"
            if self.mae_hla_err_deg is not None
            else "HLA: no ref"
        )
        return f"{base} | {speed_s} | {hla_s}"

    def meets_acceptance(
        self,
        speed_tol_pct: float = 3.0,
        hla_tol_deg: float = 1.0,
    ) -> bool:
        """True if all recorded reference errors fall within acceptance gates."""
        require(speed_tol_pct > 0, "speed tolerance must be positive")
        require(hla_tol_deg > 0, "HLA tolerance must be positive")
        if not self.speed_errors_pct or not self.hla_errors_deg:
            return False
        speed_ok = all(abs(e) <= speed_tol_pct for e in self.speed_errors_pct)
        hla_ok = all(abs(e) <= hla_tol_deg for e in self.hla_errors_deg)
        return speed_ok and hla_ok


class ValidationHarness:
    """Manages an interactive or automated accuracy validation session."""

    def __init__(
        self,
        monitor: PuttingMonitor,
        csv_path: Path,
        *,
        input_stream: TextIO | None = None,
        output_stream: TextIO | None = None,
        fixed_ref_speed: float | None = None,
        fixed_ref_hla: float | None = None,
        interactive: bool = True,
        max_putts: int | None = None,
    ) -> None:
        require(
            fixed_ref_speed is None or fixed_ref_speed > 0,
            "fixed_ref_speed must be positive",
        )
        require(
            max_putts is None or max_putts > 0,
            "max_putts must be positive",
        )
        self.monitor = monitor
        self.csv_path = csv_path
        self.input_stream = input_stream or sys.stdin
        self.output_stream = output_stream or sys.stdout
        self.fixed_ref_speed = fixed_ref_speed
        self.fixed_ref_hla = fixed_ref_hla
        self.interactive = interactive
        self.max_putts = max_putts
        self.stats = RunningStats()
        self.records: list[ValidationRecord] = []

        self._ensure_csv_header()
        self.monitor.add_observer(self._on_frame_event)

    def _ensure_csv_header(self) -> None:
        if not self.csv_path.exists() or self.csv_path.stat().st_size == 0:
            self.csv_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(CSV_HEADER)

    def _append_record(self, record: ValidationRecord) -> None:
        self.records.append(record)
        self.stats.update(record)
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(record.to_csv_row())

    def _on_frame_event(self, event: FrameEvent) -> None:
        if event.putt is None:
            return
        putt: Putt = event.putt
        now_iso = datetime.now(UTC).isoformat()

        if not putt.accepted:
            logger.info("PUTT REJECTED: %s", putt.reason)
            record = ValidationRecord(
                timestamp_iso=now_iso,
                speed_mph=putt.speed_mph,
                hla_deg=putt.hla_deg,
                points=putt.launch.points,
                r2=putt.launch.r2,
                span_mm=putt.launch.span_mm,
                ref_speed_mph=None,
                ref_hla_deg=None,
                accepted=False,
                reason=putt.reason,
            )
            self._append_record(record)
            self._write(f"[REJECTED] {putt.reason}\n")
            return

        ref_speed = self.fixed_ref_speed
        ref_hla = self.fixed_ref_hla
        note = ""

        if self.interactive and (ref_speed is None or ref_hla is None):
            self._write(
                f"\n>>> PUTT ACCEPTED: {putt.speed_mph:.2f} mph, "
                f"HLA {putt.hla_deg:+.2f}° "
                f"({putt.launch.points} pts, r²={putt.launch.r2:.3f})\n"
            )
            if ref_speed is None:
                self._write("    Enter reference speed (mph) [Enter to skip]: ")
                line = self.input_stream.readline().strip()
                if line:
                    try:
                        ref_speed = float(line)
                    except ValueError:
                        self._write("    Invalid float; skipping speed ref.\n")
            if ref_hla is None:
                self._write("    Enter reference HLA (deg) [Enter to skip]: ")
                line = self.input_stream.readline().strip()
                if line:
                    try:
                        ref_hla = float(line)
                    except ValueError:
                        self._write("    Invalid float; skipping HLA ref.\n")
            self._write("    Note/tag [Enter for none]: ")
            note = self.input_stream.readline().strip()

        record = ValidationRecord(
            timestamp_iso=now_iso,
            speed_mph=putt.speed_mph,
            hla_deg=putt.hla_deg,
            points=putt.launch.points,
            r2=putt.launch.r2,
            span_mm=putt.launch.span_mm,
            ref_speed_mph=ref_speed,
            ref_hla_deg=ref_hla,
            accepted=True,
            reason="",
            note=note,
        )
        self._append_record(record)

        summary = self.stats.summary()
        self._write(f"--> Record #{len(self.records)} saved. {summary}\n")

        if self.max_putts is not None and self.stats.accepted_putts >= self.max_putts:
            logger.info(
                "Reached maximum requested putts (%d); stopping.", self.max_putts
            )
            self.monitor.stop()

    def _write(self, msg: str) -> None:
        self.output_stream.write(msg)
        self.output_stream.flush()

    def run(self) -> int:
        """Executes the monitor and returns the total frames processed."""
        logger.info("Validation harness started; logging to %s", self.csv_path)
        frames = self.monitor.run()
        logger.info("Validation harness ended: %s", self.stats.summary())
        return frames
