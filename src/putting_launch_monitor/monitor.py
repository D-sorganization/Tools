"""The running monitor: frames in, putts out, GSPro told.

:class:`PuttingMonitor` wires a :class:`FrameSource` to the detector and
tracker from a :class:`Calibration`, and hands accepted putts to a
:class:`Sink`. The GSPro sink only forwards while GSPro reports a putter
in hand (or when the operator overrides that), so a stray roll during a
full-swing hole never reaches the simulator. Observers see every event —
frame, phase change, putt, rejection — for a GUI or a log; the monitor
itself never draws or prints.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from shared.python.contracts import StateError, require
from shared.python.sidekick.lab.mocap.acquisition import FramePacket, FrameSource

from .calibration import Calibration
from .detect import BallDetector, BallObservation, Frame
from .gspro import GsproClient, PuttShot, Reply
from .track import Phase, Putt, PuttTracker

logger = logging.getLogger(__name__)


class Sink(Protocol):
    """Where accepted putts go."""

    def deliver(self, putt: Putt) -> str: ...


@dataclass
class LogSink:
    """Records putts; the default when no simulator is attached."""

    putts: list[Putt] = field(default_factory=list)

    def deliver(self, putt: Putt) -> str:
        self.putts.append(putt)
        return f"logged {putt.speed_mph:.2f} mph, HLA {putt.hla_deg:+.1f}"


@dataclass
class GsproSink:
    """Forwards to GSPro, but only in putting mode unless ``always`` is set."""

    client: GsproClient
    always: bool = False
    sent: int = 0
    held: int = 0

    def deliver(self, putt: Putt) -> str:
        if not self.always and not self.client.putting_mode:
            self.held += 1
            club = self.client.player.club if self.client.player else "unknown"
            return f"held: GSPro club is {club}, not a putter"
        reply: Reply = self.client.send_shot(PuttShot(putt.speed_mph, putt.hla_deg))
        if reply.ok:
            self.sent += 1
            return f"GSPro accepted shot #{self.client.shot_number} ({reply.code})"
        return f"GSPro refused shot: {reply.code} {reply.message}"


@dataclass(frozen=True)
class FrameEvent:
    """What happened on one frame, for observers.

    ``candidates`` are every plausible ball the detector saw (best first;
    ``seen`` is the first) and ``frame`` is the decoded BGR image the
    detector ran on, shared without a copy so a display can draw over it.
    Both default empty so older observers and hand-built events still work.
    """

    sequence: int
    timestamp_ns: int
    seen: BallObservation | None
    phase: Phase
    putt: Putt | None
    outcome: str
    candidates: tuple[BallObservation, ...] = ()
    frame: Frame | None = None


Observer = Callable[[FrameEvent], None]


def frame_to_bgr(packet: FramePacket) -> Frame:
    """The packet's pixels as an ``(h, w, 3)`` BGR array (no copy)."""
    require(
        packet.pixel_format == "bgr24", "monitor expects bgr24", packet.pixel_format
    )
    w, h = packet.resolution_px
    pixels = np.frombuffer(packet.image_bytes, dtype=np.uint8)
    return np.asarray(pixels.reshape(h, w, 3), dtype=np.uint8)


class PuttingMonitor:
    """Runs the pipeline over a source until told to stop or the source ends."""

    def __init__(
        self,
        calibration: Calibration,
        source: FrameSource,
        sink: Sink | None = None,
        *,
        detector: BallDetector | None = None,
    ) -> None:
        self.calibration = calibration
        self.source = source
        self.sink: Sink = sink or LogSink()
        self.detector: BallDetector = detector or calibration.detector()
        self.tracker = PuttTracker(
            calibration.ground_plane(), calibration.tracker_settings()
        )
        self._observers: list[Observer] = []
        self._stop = False
        self.frames = 0
        self.putts: list[Putt] = []

    def add_observer(self, observer: Observer) -> None:
        self._observers.append(observer)

    def stop(self) -> None:
        self._stop = True

    def step(self, packet: FramePacket) -> FrameEvent:
        """Process one frame. Postcondition: ``frames`` advanced by one."""
        frame = frame_to_bgr(packet)
        candidates = self.detector.detect_all(frame)
        seen = candidates[0] if candidates else None
        putt = self.tracker.update(packet.timestamp_ns, candidates)
        outcome = ""
        if putt is not None:
            self.putts.append(putt)
            outcome = self._handle(putt)
        self.frames += 1
        event = FrameEvent(
            packet.sequence_number,
            packet.timestamp_ns,
            seen,
            self.tracker.phase,
            putt,
            outcome,
            tuple(candidates),
            frame,
        )
        for observer in self._observers:
            observer(event)
        return event

    def run(self, max_frames: int | None = None) -> int:
        """Pump frames until stop, source end or ``max_frames``; returns frames seen."""
        require(max_frames is None or max_frames > 0, "max_frames", max_frames)
        self.source.initialize()
        self.source.start_capture()
        try:
            while not self._stop and (max_frames is None or self.frames < max_frames):
                try:
                    packet = self.source.read_frame()
                except StateError as exc:
                    logger.info("source ended: %s", exc)
                    break
                self.step(packet)
        finally:
            self.source.close()
        return self.frames

    def _handle(self, putt: Putt) -> str:
        if not putt.accepted:
            logger.info("putt rejected: %s", putt.reason)
            return f"rejected: {putt.reason}"
        outcome = self.sink.deliver(putt)
        logger.info(
            "putt %.2f mph HLA %+.1f -> %s", putt.speed_mph, putt.hla_deg, outcome
        )
        return outcome
