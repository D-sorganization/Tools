"""Running the monitor off the GUI thread, and getting its events back on it.

:class:`MonitorWorker` runs :meth:`PuttingMonitor.run` on a ``QThread``.
The monitor calls observers on that thread; :class:`EventBridge` is the
observer that carries :class:`FrameEvent` objects to the GUI thread over a
queued signal, coalescing frames so a 60 fps camera never outruns the
painter — a putt event is never dropped, a plain frame may be superseded
by the next one. Nothing here inspects pixels.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable

from PyQt6.QtCore import QObject, QThread, pyqtSignal

from putting_launch_monitor.calibration import Calibration
from putting_launch_monitor.cli import scaled_calibration
from putting_launch_monitor.detect import Frame
from putting_launch_monitor.monitor import FrameEvent, PuttingMonitor, Sink
from shared.python.camera import CaptureMode, FfmpegDirectShowSource
from shared.python.camera.ffmpeg_source import DEFAULT_MODE
from shared.python.contracts import StateError, require
from shared.python.sidekick.lab.mocap.acquisition import FrameSource

logger = logging.getLogger(__name__)

SNAPSHOT_SETTLE_FRAMES = 10  # let the camera's exposure settle, as the CLI does
DEFAULT_DECODE_WIDTH = 960


class EventBridge(QObject):
    """An observer that re-emits events on the GUI thread, coalescing frames.

    Call :meth:`release` from the slot once an event has been consumed;
    until then further frame-only events are dropped. Events carrying a
    putt always go through.
    """

    arrived = pyqtSignal(object)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._lock = threading.Lock()
        self._pending = False
        self.dropped = 0

    def __call__(self, event: FrameEvent) -> None:
        with self._lock:
            if self._pending and event.putt is None:
                self.dropped += 1
                return
            self._pending = True
        self.arrived.emit(event)

    def release(self) -> None:
        with self._lock:
            self._pending = False


class MonitorWorker(QThread):
    """Owns one :class:`PuttingMonitor` and pumps it until stopped."""

    failed = pyqtSignal(str)
    frames_done = pyqtSignal(int)

    def __init__(self, monitor: PuttingMonitor, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.monitor = monitor

    def run(self) -> None:  # runs on the worker thread
        try:
            frames = self.monitor.run()
        except (OSError, StateError, ValueError, RuntimeError) as exc:
            logger.exception("monitor stopped on error")
            self.failed.emit(str(exc))
            return
        self.frames_done.emit(frames)

    def stop(self, timeout_ms: int = 10_000) -> bool:
        """Ask the monitor to stop and wait for the thread; True when it ended."""
        self.monitor.stop()
        return bool(self.wait(timeout_ms))


SourceFactory = Callable[[Calibration, int], FrameSource]


def camera_source(calibration: Calibration, decode_width: int) -> FrameSource:
    """The live DirectShow source for a calibration, decoded at ``decode_width``."""
    require(decode_width > 0, "decode width", decode_width)
    mode = CaptureMode(
        width=calibration.capture_width,
        height=calibration.capture_height,
        fps=calibration.fps,
    )
    return FfmpegDirectShowSource(
        calibration.camera_instance_id, mode, width=decode_width
    )


def build_monitor(
    calibration: Calibration,
    decode_width: int,
    sink: Sink | None,
    observer: EventBridge,
    source_factory: SourceFactory = camera_source,
) -> PuttingMonitor:
    """A monitor over the camera at the decode width, observed by ``observer``.

    The calibration is re-expressed at the decode width (as the CLI does) so
    the search region and radius bounds match the frames the detector sees.
    """
    scaled = scaled_calibration(calibration, decode_width)
    monitor = PuttingMonitor(scaled, source_factory(calibration, decode_width), sink)
    monitor.add_observer(observer)
    return monitor


def take_snapshot(
    camera_instance_id: str,
    mode: CaptureMode = DEFAULT_MODE,
    *,
    source: FrameSource | None = None,
) -> Frame:
    """One settled full-resolution frame from the camera, as BGR.

    The camera is single-open: the caller must have stopped the monitor.
    ``source`` is injectable for tests.
    """
    from putting_launch_monitor.monitor import frame_to_bgr

    require(bool(camera_instance_id.strip()), "camera instance id")
    src = source or FfmpegDirectShowSource(camera_instance_id, mode)
    src.initialize()
    src.start_capture()
    try:
        packet = src.read_frame()
        for _ in range(SNAPSHOT_SETTLE_FRAMES - 1):
            packet = src.read_frame()
    finally:
        src.close()
    return frame_to_bgr(packet).copy()
