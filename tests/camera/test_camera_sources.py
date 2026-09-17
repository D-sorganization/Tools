"""The shared camera layer: device refs, ffmpeg arguments, and both sources.

The ffmpeg source is driven through an injected fake process so its whole
lifecycle — start, frame framing, index-based timestamps, stream end,
stop — runs on a machine with no camera. Argument builders are pinned to
the exact command UpstreamDrift validated on the lab rig.
"""

from __future__ import annotations

import io
import subprocess
from pathlib import Path

import numpy as np
import pytest

from shared.python.camera import (
    CaptureMode,
    FfmpegDirectShowSource,
    VideoFileSource,
    dshow_device_ref,
    ffmpeg_raw_frame_args,
    output_frame_size,
)
from shared.python.contracts import StateError
from shared.python.sidekick.lab.mocap.acquisition import FrameSource, SourceState

pytestmark = pytest.mark.unit

INSTANCE = "USB\\VID_32E4&PID_5234&MI_00\\9&2A7EE39F&0&0000"


# -- pure builders -------------------------------------------------------------------
def test_device_ref_matches_directshow_naming() -> None:
    ref = dshow_device_ref(INSTANCE)
    assert ref == (
        "@device_pnp_\\\\?\\usb#vid_32e4&pid_5234&mi_00#9&2a7ee39f&0&0000"
        "#{65e8773d-8f56-11d0-a3b9-00a0c9223196}\\global"
    )
    with pytest.raises(ValueError, match="non-empty"):
        dshow_device_ref("   ")


def test_capture_mode_contract_and_codec() -> None:
    assert CaptureMode().ffmpeg_codec == "mjpeg"
    assert CaptureMode(fourcc="YUY2").ffmpeg_codec == "yuy2"
    with pytest.raises(ValueError):
        CaptureMode(width=0)
    with pytest.raises(ValueError):
        CaptureMode(fourcc="MJ")


def test_output_frame_size_keeps_aspect_and_even_height() -> None:
    mode = CaptureMode(width=1920, height=1200)
    assert output_frame_size(mode, None) == (1920, 1200)
    assert output_frame_size(mode, 640) == (640, 400)
    assert output_frame_size(CaptureMode(width=1280, height=720), 480)[1] % 2 == 0
    with pytest.raises(ValueError):
        output_frame_size(mode, 0)


def test_ffmpeg_args_are_the_validated_rig_command() -> None:
    mode = CaptureMode()
    plain = ffmpeg_raw_frame_args("ffmpeg", "ref", mode)
    assert plain[:6] == ["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "dshow"]
    assert plain[plain.index("-vcodec") + 1] == "mjpeg"
    assert plain[plain.index("-video_size") + 1] == "1920x1200"
    assert plain[plain.index("-framerate") + 1] == "60"
    assert plain[plain.index("-i") + 1] == "video=ref"
    assert "-vf" not in plain and plain[-5:] == [
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "pipe:1",
    ]
    shrunk = ffmpeg_raw_frame_args("ffmpeg", "ref", mode, width=640, fps=12, lowres=2)
    assert shrunk[shrunk.index("-vf") + 1] == "fps=12,scale=640:-2"
    assert shrunk[shrunk.index("-lowres:v") + 1] == "2"
    assert shrunk.index("-lowres:v") < shrunk.index("-i")  # an input option
    with pytest.raises(ValueError):
        ffmpeg_raw_frame_args("ffmpeg", "ref", mode, lowres=4)
    with pytest.raises(ValueError):
        ffmpeg_raw_frame_args("", "ref", mode)


# -- ffmpeg source through a fake process ----------------------------------------------
class FakeProc:
    """Stands in for ffmpeg: serves ``frames`` raw BGR frames then EOF."""

    def __init__(self, frames: list[bytes], stderr: bytes = b"") -> None:
        self.stdin = io.BytesIO()
        self.stdout = io.BytesIO(b"".join(frames))
        self.stderr = io.BytesIO(stderr)
        self.terminated = False
        self.killed = False

    def poll(self) -> int | None:
        return None if not self.terminated else 0

    def terminate(self) -> None:
        self.terminated = True

    def wait(self, timeout: float | None = None) -> int:
        return 0

    def kill(self) -> None:
        self.killed = True


def test_ffmpeg_source_lifecycle_frames_and_index_timestamps() -> None:
    mode = CaptureMode(width=8, height=4, fps=60)
    frames = [bytes([i]) * (8 * 4 * 3) for i in range(3)]
    made: list[FakeProc] = []

    def popen(args: list[str], **kwargs: object) -> FakeProc:
        assert args[args.index("-i") + 1].startswith("video=@device_pnp_")
        proc = FakeProc(frames, stderr=b"[dshow] finished")
        made.append(proc)
        return proc

    src = FfmpegDirectShowSource(INSTANCE, mode, ffmpeg_exe="ffmpeg", popen=popen)
    assert isinstance(src, FrameSource)
    assert src.state is SourceState.UNINITIALIZED
    with pytest.raises(ValueError, match="initialize"):
        src.start_capture()
    src.initialize()
    src.start_capture()
    assert src.state is SourceState.CAPTURING and src.frame_shape == (8, 4)
    a, b, c = (src.read_frame() for _ in range(3))
    assert (a.sequence_number, b.sequence_number, c.sequence_number) == (0, 1, 2)
    # Each stamp is computed from its index, never accumulated, so no drift.
    assert b.timestamp_ns == 1_000_000_000 // 60
    assert c.timestamp_ns == 2 * 1_000_000_000 // 60
    assert a.image_bytes == frames[0] and a.resolution_px == (8, 4)
    assert a.pixel_format == "bgr24" and a.source_id.startswith("dshow:")
    with pytest.raises(StateError, match="stream ended"):
        src.read_frame()
    src.close()
    assert src.state is SourceState.CLOSED and made[0].terminated


def test_ffmpeg_source_capabilities_describe_the_emitted_stream() -> None:
    src = FfmpegDirectShowSource(INSTANCE, width=640, fps=12, lowres=1, ffmpeg_exe="f")
    caps = src.capabilities
    assert caps.resolutions_px == ((320, 200),)  # 640x400 halved by lowres
    assert caps.frame_rates_hz == (12.0,)
    assert src.identity.provider_id == "ffmpeg-dshow"
    with pytest.raises(ValueError):
        FfmpegDirectShowSource("  ")


# -- video file source -----------------------------------------------------------------
def test_video_file_source_replays_a_clip(tmp_path: Path) -> None:
    cv2 = pytest.importorskip("cv2")
    path = tmp_path / "clip.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 30.0, (32, 16))
    for i in range(5):
        writer.write(np.full((16, 32, 3), i * 40, dtype=np.uint8))
    writer.release()
    src = VideoFileSource(path, fps=60)
    assert isinstance(src, FrameSource)
    src.initialize()
    assert src.fps == 60 and src.frame_count == 5
    src.start_capture()
    first = src.read_frame()
    assert first.resolution_px == (32, 16) and first.sequence_number == 0
    second = src.read_frame()
    assert second.timestamp_ns == int(1e9 / 60)
    for _ in range(3):
        src.read_frame()
    with pytest.raises(StateError, match="end of video"):
        src.read_frame()
    src.close()
    assert src.state is SourceState.CLOSED
    with pytest.raises(ValueError, match="must exist"):
        VideoFileSource(tmp_path / "missing.avi")


def test_popen_default_is_subprocess() -> None:
    src = FfmpegDirectShowSource(INSTANCE, ffmpeg_exe="f")
    assert src._popen is subprocess.Popen


def test_pure_builders_importable_without_sidekick(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sys

    for mod in list(sys.modules):
        if mod.startswith("shared.python.camera"):
            monkeypatch.delitem(sys.modules, mod, raising=False)

    acq_mod = "shared.python.sidekick.lab.mocap.acquisition"
    monkeypatch.setitem(sys.modules, "shared.python.sidekick", None)
    monkeypatch.setitem(sys.modules, "shared.python.sidekick.lab", None)
    monkeypatch.setitem(sys.modules, "shared.python.sidekick.lab.mocap", None)
    monkeypatch.setitem(sys.modules, acq_mod, None)
    monkeypatch.setitem(sys.modules, "sidekick", None)

    import shared.python.camera as cam
    from shared.python.camera import (
        CaptureMode as CM,
    )
    from shared.python.camera import (
        dshow_device_ref as dref,
    )
    from shared.python.camera import (
        ffmpeg_raw_frame_args as fargs,
    )
    from shared.python.camera import (
        output_frame_size as osize,
    )

    assert cam.CaptureMode is CM
    assert callable(dref)
    assert callable(fargs)
    assert callable(osize)
    assert "FfmpegDirectShowSource" in dir(cam)
    assert "VideoFileSource" in dir(cam)
    with pytest.raises(ModuleNotFoundError):
        _ = cam.FfmpegDirectShowSource
    with pytest.raises(ModuleNotFoundError):
        _ = cam.VideoFileSource


def test_source_classes_subclass_frame_source() -> None:
    import shared.python.camera as cam

    assert issubclass(cam.FfmpegDirectShowSource, FrameSource)
    assert issubclass(cam.VideoFileSource, FrameSource)
