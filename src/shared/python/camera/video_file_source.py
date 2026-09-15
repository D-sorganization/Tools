"""A recorded video as a :class:`FrameSource`.

The same contract a live camera satisfies, so a tracking pipeline is
exercised on a file — a regression corpus of recorded putts, say — exactly
as it runs on hardware. Timestamps follow the file's frame rate; ``fps``
overrides it for files whose container lies (MJPEG captures often do).
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from shared.python.contracts import StateError, require
from shared.python.sidekick.lab.mocap.acquisition import (
    FramePacket,
    FrameSource,
    SourceState,
)
from shared.python.sidekick.lab.mocap.devices import (
    CameraCapabilities,
    CameraIdentity,
    FeatureSupport,
)
from shared.python.sidekick.lab.mocap.enums import ShutterKind, SupportLevel

PIXEL_FORMAT = "bgr24"
NS_PER_S = 1_000_000_000


class VideoFileSource(FrameSource):
    """Frames from a video file, in order, through OpenCV.

    Precondition at construction: the file exists. ``cv2`` is imported on
    :meth:`initialize`, so the module loads without OpenCV installed.
    """

    def __init__(self, path: Path | str, *, fps: float | None = None) -> None:
        self._path = Path(path)
        require(self._path.is_file(), "video file must exist", str(self._path))
        require(fps is None or fps > 0, "fps override must be positive", fps)
        self._fps_override = fps
        self._cap: Any = None
        self._cv2: Any = None
        self._state = SourceState.UNINITIALIZED
        self._sequence = 0
        self._shape: tuple[int, int] = (1, 1)
        self._fps: float = fps or 0.0
        self._frame_count = 0

    @property
    def source_id(self) -> str:
        return f"file:{self._path.name}"

    @property
    def identity(self) -> CameraIdentity:
        return CameraIdentity(
            provider_id="video-file", device_id=str(self._path), transport="file"
        )

    @property
    def capabilities(self) -> CameraCapabilities:
        return CameraCapabilities(
            resolutions_px=(self._shape,),
            frame_rates_hz=(float(self._fps or 1.0),),
            pixel_formats=(PIXEL_FORMAT,),
            shutter=ShutterKind.GLOBAL,
            hardware_trigger=FeatureSupport(SupportLevel.UNSUPPORTED, "recorded file"),
            device_timestamps=FeatureSupport(
                SupportLevel.DEGRADED, "timestamps derive from frame index x rate"
            ),
        )

    @property
    def state(self) -> SourceState:
        return self._state

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def frame_count(self) -> int:
        """Frames the container reports (0 when unknown)."""
        return self._frame_count

    def initialize(self) -> None:
        import cv2

        self._cv2 = cv2
        cap = cv2.VideoCapture(str(self._path))
        if not cap.isOpened():
            raise StateError(f"cannot open video: {self._path}")
        self._cap = cap
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._shape = (max(w, 1), max(h, 1))
        reported = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        self._fps = self._fps_override or (reported if reported > 0 else 30.0)
        self._frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self._state = SourceState.INITIALIZED

    def start_capture(self) -> None:
        require(self._cap is not None, "initialize() first")
        self._state = SourceState.CAPTURING

    def read_frame(self, timeout_seconds: float = 5.0) -> FramePacket:
        """The next frame; ``StateError`` at end of file."""
        if self._cap is None or self._state is not SourceState.CAPTURING:
            raise StateError("read_frame() before start_capture()")
        ok, frame = self._cap.read()
        if not ok or frame is None:
            self._state = SourceState.STOPPED
            raise StateError("end of video")
        seq = self._sequence
        self._sequence += 1
        return FramePacket(
            source_id=self.source_id,
            sequence_number=seq,
            timestamp_ns=int(seq * NS_PER_S / self._fps),
            host_monotonic_ns=time.monotonic_ns(),
            image_bytes=frame.tobytes(),
            pixel_format=PIXEL_FORMAT,
            resolution_px=(int(frame.shape[1]), int(frame.shape[0])),
        )

    def stop_capture(self) -> None:
        if self._state is SourceState.CAPTURING:
            self._state = SourceState.STOPPED

    def close(self) -> None:
        cap, self._cap = self._cap, None
        if cap is not None:
            cap.release()
        self._state = SourceState.CLOSED
