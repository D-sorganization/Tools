"""A live DirectShow camera as a :class:`FrameSource`, decoded by ffmpeg.

One ffmpeg process per camera writes raw BGR frames to a pipe; this class
reads them frame by frame. Timestamps come from the frame index and the
mode's nominal rate, not from arrival time: the camera's clock is far
steadier than pipe scheduling, and a speed measured over a dozen frames at
60 fps is sensitive to jitter of a millisecond. The host monotonic time is
recorded alongside for anyone who needs wall-clock alignment.
"""

from __future__ import annotations

import subprocess
import time
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

from .dshow import (
    CaptureMode,
    dshow_device_ref,
    ffmpeg_raw_frame_args,
    output_frame_size,
)

PIXEL_FORMAT = "bgr24"
DEFAULT_MODE = CaptureMode()
STOP_TIMEOUT_S = 3.0
NS_PER_S = 1_000_000_000


def default_ffmpeg_exe() -> str:
    """The bundled imageio-ffmpeg binary (never a bare ``ffmpeg`` on PATH)."""
    import imageio_ffmpeg

    return str(imageio_ffmpeg.get_ffmpeg_exe())


class FfmpegDirectShowSource(FrameSource):
    """Frames from one Windows camera, addressed by its PnP instance id.

    ``width``/``fps``/``lowres`` shrink the stream for preview or tracking
    work; the reported ``capabilities`` describe what is *emitted*, not the
    sensor. ``popen`` is injectable so the lifecycle is tested without a
    device.
    """

    def __init__(
        self,
        camera_instance_id: str,
        mode: CaptureMode = DEFAULT_MODE,
        *,
        width: int | None = None,
        fps: int | None = None,
        lowres: int = 0,
        ffmpeg_exe: str | None = None,
        popen: Any = None,
    ) -> None:
        require(
            bool(camera_instance_id.strip()), "camera instance id must be non-empty"
        )
        self._instance_id = camera_instance_id.strip()
        self._device_ref = dshow_device_ref(self._instance_id)
        self._mode = mode
        self._width, self._fps, self._lowres = width, fps, lowres
        self._ffmpeg = ffmpeg_exe
        self._popen = popen or subprocess.Popen
        self._proc: Any = None
        self._state = SourceState.UNINITIALIZED
        self._sequence = 0
        self._started_ns = 0
        out_w, out_h = output_frame_size(mode, width)
        if lowres:
            out_w, out_h = out_w >> lowres, out_h >> lowres
        self._shape = (max(out_w, 1), max(out_h, 1))
        self._identity = CameraIdentity(
            provider_id="ffmpeg-dshow",
            device_id=self._instance_id,
            transport="usb",
        )

    # -- FrameSource contract -----------------------------------------------------
    @property
    def source_id(self) -> str:
        return f"dshow:{self._instance_id}"

    @property
    def identity(self) -> CameraIdentity:
        return self._identity

    @property
    def capabilities(self) -> CameraCapabilities:
        return CameraCapabilities(
            resolutions_px=(self._shape,),
            frame_rates_hz=(float(self.output_fps),),
            pixel_formats=(PIXEL_FORMAT,),
            shutter=ShutterKind.ROLLING,
            hardware_trigger=FeatureSupport(
                SupportLevel.UNSUPPORTED, "UVC camera has no trigger input"
            ),
            device_timestamps=FeatureSupport(
                SupportLevel.DEGRADED, "timestamps derive from frame index x rate"
            ),
        )

    @property
    def state(self) -> SourceState:
        return self._state

    @property
    def output_fps(self) -> int:
        return self._fps if self._fps is not None else self._mode.fps

    @property
    def frame_shape(self) -> tuple[int, int]:
        """``(width, height)`` of emitted frames."""
        return self._shape

    def initialize(self) -> None:
        """Resolve the decoder; no device is touched until :meth:`start_capture`."""
        if self._ffmpeg is None:
            self._ffmpeg = default_ffmpeg_exe()
        self._state = SourceState.INITIALIZED

    def start_capture(self) -> None:
        """Launch ffmpeg. Raises ``StateError`` if it dies before the first byte."""
        require(self._state is not SourceState.UNINITIALIZED, "initialize() first")
        if self._proc is not None:
            return
        args = ffmpeg_raw_frame_args(
            str(self._ffmpeg),
            self._device_ref,
            self._mode,
            width=self._width,
            fps=self._fps,
            lowres=self._lowres,
        )
        self._proc = self._popen(
            args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        self._sequence = 0
        self._started_ns = time.monotonic_ns()
        self._state = SourceState.CAPTURING

    def read_frame(self, timeout_seconds: float = 5.0) -> FramePacket:
        """The next frame. Raises ``StateError`` when the stream has ended.

        Postcondition: ``sequence_number`` increases by one per call and
        ``timestamp_ns`` equals ``sequence_number / fps`` in nanoseconds.
        """
        require(timeout_seconds > 0, "timeout must be positive", timeout_seconds)
        proc = self._proc
        if proc is None or proc.stdout is None:
            raise StateError("read_frame() before start_capture()")
        width, height = self._shape
        need = width * height * 3
        buf = bytearray()
        while len(buf) < need:
            chunk = proc.stdout.read(need - len(buf))
            if not chunk:
                self._state = SourceState.STOPPED
                raise StateError(f"camera stream ended: {self._stderr_tail()}")
            buf.extend(chunk)
        seq = self._sequence
        self._sequence += 1
        return FramePacket(
            source_id=self.source_id,
            sequence_number=seq,
            timestamp_ns=seq * NS_PER_S // self.output_fps,
            host_monotonic_ns=time.monotonic_ns(),
            image_bytes=bytes(buf),
            pixel_format=PIXEL_FORMAT,
            resolution_px=self._shape,
        )

    def stop_capture(self) -> None:
        """End the ffmpeg process; safe to call repeatedly."""
        proc, self._proc = self._proc, None
        if proc is None:
            return
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=STOP_TIMEOUT_S)
            except subprocess.TimeoutExpired:
                proc.kill()
        for stream in (proc.stdin, proc.stdout, proc.stderr):
            if stream is not None:
                stream.close()
        self._state = SourceState.STOPPED

    def close(self) -> None:
        self.stop_capture()
        self._state = SourceState.CLOSED

    # -- helpers --------------------------------------------------------------------
    def _stderr_tail(self) -> str:
        proc = self._proc
        if proc is None or proc.stderr is None:
            return "no stderr"
        try:
            data = proc.stderr.read() or b""
        except (OSError, ValueError):
            return "stderr unavailable"
        return data[-300:].decode("utf-8", "replace").strip() or "ffmpeg exited"
