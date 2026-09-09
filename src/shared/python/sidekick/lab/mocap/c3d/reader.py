"""C3D binary parser and structured data extraction (TOOLS-M9 #4716).

Supports standard C3D binary structure with fail-closed bounds checking.
"""

from __future__ import annotations

import struct
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .schema import (
    C3DAnalogChannel,
    C3DEvent,
    C3DHeader,
    C3DPointChannel,
)

C3D_HEADER_MAGIC_BYTE = 0x50
C3D_HEADER_LENGTH = 2


def validate_c3d_header_magic(file_path: Path) -> None:
    """Validate the C3D header magic byte (0x50 in 2nd byte) before full parsing."""
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    with file_path.open("rb") as f:
        header = f.read(C3D_HEADER_LENGTH)
    if len(header) < C3D_HEADER_LENGTH or header[1] != C3D_HEADER_MAGIC_BYTE:
        raise ValueError(f"Not a valid C3D file: {file_path}")


def parse_c3d_header(data: bytes) -> C3DHeader:
    """Parse raw header block (512 bytes) of a C3D file."""
    if len(data) < 512:
        raise ValueError("C3D header block must be at least 512 bytes")
    if data[1] != C3D_HEADER_MAGIC_BYTE:
        raise ValueError("Invalid C3D header magic byte")

    point_count = struct.unpack_from("<H", data, 2)[0]
    analog_channels_per_frame = struct.unpack_from("<H", data, 4)[0]
    first_frame = struct.unpack_from("<H", data, 6)[0]
    last_frame = struct.unpack_from("<H", data, 8)[0]
    max_interpolation_gap = struct.unpack_from("<H", data, 10)[0]
    scale_factor = struct.unpack_from("<f", data, 12)[0]
    data_start_block = struct.unpack_from("<H", data, 16)[0]
    analog_samples_per_frame = struct.unpack_from("<H", data, 18)[0]
    frame_rate_hz = struct.unpack_from("<f", data, 20)[0]

    return C3DHeader(
        point_count=point_count,
        analog_channels_per_frame=analog_channels_per_frame,
        first_frame=first_frame,
        last_frame=last_frame,
        max_interpolation_gap=max_interpolation_gap,
        scale_factor=scale_factor,
        data_start_block=data_start_block,
        analog_samples_per_frame=analog_samples_per_frame,
        frame_rate_hz=frame_rate_hz,
    )


class C3DContainer:
    """In-memory representation of C3D points, analogs, events, and parameters."""

    def __init__(
        self,
        header: C3DHeader,
        points: Sequence[C3DPointChannel] = (),
        analogs: Sequence[C3DAnalogChannel] = (),
        events: Sequence[C3DEvent] = (),
        parameters: dict[str, Any] | None = None,
    ) -> None:
        self.header = header
        self.points = tuple(points)
        self.analogs = tuple(analogs)
        self.events = tuple(events)
        self.parameters = parameters or {}

    def get_point(self, label: str) -> C3DPointChannel | None:
        clean = label.strip()
        for p in self.points:
            if p.label == clean:
                return p
        return None

    def get_analog(self, label: str) -> C3DAnalogChannel | None:
        clean = label.strip()
        for a in self.analogs:
            if a.label == clean:
                return a
        return None
