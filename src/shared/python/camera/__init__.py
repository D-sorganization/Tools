"""Live USB camera sources for the fleet (the DRY leaf for camera capture).

UpstreamDrift's capture rig proved, on the lab's three ELP AR0234 cameras,
that the only path which streams them at their full 1920x1200@60 is ffmpeg
addressing each camera by its DirectShow device reference: OpenCV's Media
Foundation backend hangs opening the third unit and its DirectShow backend
refuses the full mode by index. This package carries that path as an
implementation of :class:`sidekick.lab.mocap.acquisition.FrameSource`, so
every tool that needs a live camera — the putting launch monitor first,
UpstreamDrift's rig when it adopts the vendored copy — opens devices the
same way.

:class:`VideoFileSource` replays a recording through the same contract, so a
pipeline is tested on a file exactly as it runs on a camera.
"""

from __future__ import annotations

from .dshow import (
    DSHOW_DEVICE_PREFIX,
    DSHOW_VIDEO_CATEGORY,
    CaptureMode,
    dshow_device_ref,
    ffmpeg_raw_frame_args,
    output_frame_size,
)
from .ffmpeg_source import FfmpegDirectShowSource
from .video_file_source import VideoFileSource

__all__ = [
    "DSHOW_DEVICE_PREFIX",
    "DSHOW_VIDEO_CATEGORY",
    "CaptureMode",
    "FfmpegDirectShowSource",
    "VideoFileSource",
    "dshow_device_ref",
    "ffmpeg_raw_frame_args",
    "output_frame_size",
]
