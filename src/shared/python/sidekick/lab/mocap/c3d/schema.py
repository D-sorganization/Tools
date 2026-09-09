"""Typed contracts and schemas for C3D biomechanical exchange (TOOLS-M9 #4716).

Follows C3D standard specification and ADR-008.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .._validation import require_finite, require_text

__all__ = [
    "BIOMECHANICAL_MARKER_MAX_M",
    "BIOMECHANICAL_MARKER_MIN_M",
    "C3DAnalogChannel",
    "C3DEvent",
    "C3DForcePlatform",
    "C3DHeader",
    "C3DPointChannel",
]

BIOMECHANICAL_MARKER_MIN_M: float = 0.001
BIOMECHANICAL_MARKER_MAX_M: float = 10.0


@dataclass(frozen=True, slots=True)
class C3DHeader:
    point_count: int
    analog_channels_per_frame: int
    first_frame: int
    last_frame: int
    max_interpolation_gap: int
    scale_factor: float
    data_start_block: int
    analog_samples_per_frame: int
    frame_rate_hz: float

    def __post_init__(self) -> None:
        if self.point_count < 0:
            raise ValueError("point_count cannot be negative")
        if self.analog_channels_per_frame < 0:
            raise ValueError("analog_channels_per_frame cannot be negative")
        if self.first_frame < 0:
            raise ValueError("first_frame cannot be negative")
        if self.last_frame < self.first_frame:
            raise ValueError("last_frame must be >= first_frame")
        if require_finite(self.scale_factor, "scale_factor") == 0.0:
            raise ValueError("scale_factor cannot be 0.0")
        if self.analog_samples_per_frame < 0:
            raise ValueError("analog_samples_per_frame cannot be negative")
        if require_finite(self.frame_rate_hz, "frame_rate_hz") < 0.0:
            raise ValueError("frame_rate_hz cannot be negative")

    @property
    def frame_count(self) -> int:
        return self.last_frame - self.first_frame + 1

    @property
    def analog_sample_rate_hz(self) -> float:
        return self.frame_rate_hz * float(self.analog_samples_per_frame)


@dataclass(frozen=True, slots=True)
class C3DEvent:
    label: str
    time_s: float

    def __post_init__(self) -> None:
        require_text(self.label, "label")
        require_finite(self.time_s, "time_s")


@dataclass(frozen=True, slots=True)
class C3DPointChannel:
    label: str
    coordinates_xyz: tuple[tuple[float, float, float], ...]
    residuals: tuple[float, ...]
    camera_masks: tuple[int, ...]
    units: str = "mm"

    def __post_init__(self) -> None:
        require_text(self.label, "label")
        require_text(self.units, "units")
        n = len(self.coordinates_xyz)
        if len(self.residuals) != n:
            raise ValueError("residuals length must match coordinates_xyz")
        if len(self.camera_masks) != n:
            raise ValueError("camera_masks length must match coordinates_xyz")


@dataclass(frozen=True, slots=True)
class C3DAnalogChannel:
    label: str
    values: tuple[float, ...]
    units: str = "V"
    scale: float = 1.0
    offset: float = 0.0

    def __post_init__(self) -> None:
        require_text(self.label, "label")
        require_text(self.units, "units")
        require_finite(self.scale, "scale")
        require_finite(self.offset, "offset")


@dataclass(frozen=True, slots=True)
class C3DForcePlatform:
    plate_index: int
    channel_mapping: tuple[tuple[str, str], ...]
    corners_xyz_m: tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ]
    origin_xyz_m: tuple[float, float, float]
    platform_type: Literal[1, 2, 3, 4] = 2

    def __post_init__(self) -> None:
        if self.plate_index <= 0:
            raise ValueError("plate_index must be >= 1")
        if len(self.corners_xyz_m) != 4:
            raise ValueError("corners_xyz_m must contain 4 corners")
