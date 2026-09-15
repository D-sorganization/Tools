"""DirectShow device references and the ffmpeg command that reads them.

Pure functions: nothing here touches a device, so every line is unit-tested
on any machine. Ported from UpstreamDrift's ``motion_capture/rig/recorder.py``
and ``preview_source.py`` (the same authors), where the arguments were
validated against the lab rig; kept byte-compatible so a UpstreamDrift session
recorded with one and a Tools preview opened with the other agree.
"""

from __future__ import annotations

from dataclasses import dataclass

from shared.python.contracts import require

DSHOW_DEVICE_PREFIX = "@device_pnp_\\\\?\\"
DSHOW_VIDEO_CATEGORY = "{65e8773d-8f56-11d0-a3b9-00a0c9223196}"
DEFAULT_RTBUFSIZE = "256M"


@dataclass(frozen=True)
class CaptureMode:
    """A capture mode the camera advertises: size, rate and compression.

    Invariants: positive width, height and fps; a four-character FOURCC.
    """

    width: int = 1920
    height: int = 1200
    fps: int = 60
    fourcc: str = "MJPG"

    def __post_init__(self) -> None:
        require(
            self.width > 0 and self.height > 0, "mode size", (self.width, self.height)
        )
        require(self.fps > 0, "mode fps", self.fps)
        require(len(self.fourcc) == 4, "fourcc is four characters", self.fourcc)

    @property
    def ffmpeg_codec(self) -> str:
        """The ffmpeg input codec name for this FOURCC."""
        return "mjpeg" if self.fourcc.upper() == "MJPG" else self.fourcc.lower()


def dshow_device_ref(camera_instance_id: str) -> str:
    """The ffmpeg ``video=`` reference for a Windows camera instance id.

    ``camera_instance_id`` is the PnP instance id Windows reports, e.g.
    ``USB\\VID_32E4&PID_5234&MI_00\\6&FADBF3B&0&0000``. DirectShow names the
    same device ``@device_pnp_\\\\?\\usb#vid_32e4&pid_5234&mi_00#6&fadbf3b&0&0000``
    followed by ``#{category GUID}\\global`` (lower-case, backslashes to
    ``#``, the video-category GUID appended).
    Precondition: a non-empty id. Postcondition: the result starts with
    :data:`DSHOW_DEVICE_PREFIX` and ends with ``\\global``.
    """
    require(bool(camera_instance_id.strip()), "camera instance id must be non-empty")
    body = camera_instance_id.strip().lower().replace("\\", "#")
    return f"{DSHOW_DEVICE_PREFIX}{body}#{DSHOW_VIDEO_CATEGORY}\\global"


def output_frame_size(mode: CaptureMode, width: int | None) -> tuple[int, int]:
    """``(width, height)`` of the frames ffmpeg emits after an optional downscale.

    ``None`` keeps the capture size. A downscale keeps the aspect ratio and
    rounds the height to an even number, as ffmpeg's ``scale=W:-2`` does.
    Precondition: a positive width when given.
    """
    if width is None:
        return mode.width, mode.height
    require(width > 0, "output width must be positive", width)
    height = int(round(mode.height * width / mode.width / 2) * 2)
    return width, max(height, 2)


def ffmpeg_raw_frame_args(
    ffmpeg_exe: str,
    device_ref: str,
    mode: CaptureMode,
    *,
    width: int | None = None,
    fps: int | None = None,
    lowres: int = 0,
) -> list[str]:
    """ffmpeg command that decodes a DirectShow camera to raw BGR on stdout.

    ``width`` downscales the output (aspect kept), ``fps`` resamples the
    output rate, ``lowres`` asks the MJPEG decoder for a 1/2^n-size decode
    (cheap enough to run beside a recording without starving it).
    Preconditions: non-empty executable and device reference; positive
    ``width``/``fps`` when given; ``lowres`` in 0..3.
    """
    require(bool(ffmpeg_exe), "ffmpeg executable must be named")
    require(bool(device_ref), "device reference must be non-empty")
    require(fps is None or fps > 0, "output fps must be positive", fps)
    require(0 <= lowres <= 3, "lowres is 0..3", lowres)
    filters: list[str] = []
    if fps is not None:
        filters.append(f"fps={fps}")
    if width is not None:
        out_w, _ = output_frame_size(mode, width)
        filters.append(f"scale={out_w}:-2")
    args = [
        ffmpeg_exe,
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "dshow",
        "-rtbufsize",
        DEFAULT_RTBUFSIZE,
        "-vcodec",
        mode.ffmpeg_codec,
        "-video_size",
        f"{mode.width}x{mode.height}",
        "-framerate",
        str(mode.fps),
    ]
    if lowres:
        args += ["-lowres:v", str(lowres)]
    args += ["-i", f"video={device_ref}"]
    if filters:
        args += ["-vf", ",".join(filters)]
    args += ["-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1"]
    return args
