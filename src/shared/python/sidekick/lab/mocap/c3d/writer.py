"""C3D binary writer and serialization (TOOLS-M9 #4716).

Generates standard C3D binary files conforming to the 512-byte block architecture.
"""

from __future__ import annotations

import struct
from pathlib import Path

from .reader import C3D_HEADER_MAGIC_BYTE, C3DContainer
from .schema import C3DHeader


def serialize_c3d_header(header: C3DHeader) -> bytes:
    """Serialize a C3DHeader into a 512-byte header block."""
    buffer = bytearray(512)
    # Byte 0: parameter block pointer (default 2), Byte 1: magic byte 0x50
    buffer[0] = 2
    buffer[1] = C3D_HEADER_MAGIC_BYTE
    struct.pack_into("<H", buffer, 2, header.point_count)
    struct.pack_into("<H", buffer, 4, header.analog_channels_per_frame)
    struct.pack_into("<H", buffer, 6, header.first_frame)
    struct.pack_into("<H", buffer, 8, header.last_frame)
    struct.pack_into("<H", buffer, 10, header.max_interpolation_gap)
    struct.pack_into("<f", buffer, 12, header.scale_factor)
    struct.pack_into("<H", buffer, 16, header.data_start_block)
    struct.pack_into("<H", buffer, 18, header.analog_samples_per_frame)
    struct.pack_into("<f", buffer, 20, header.frame_rate_hz)
    return bytes(buffer)


def write_c3d_file(output_path: Path, container: C3DContainer) -> None:
    """Write an in-memory C3DContainer to a valid C3D binary file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    header_bytes = serialize_c3d_header(container.header)

    with output_path.open("wb") as f:
        f.write(header_bytes)
        # Parameter and data blocks can be appended deterministically
        # Minimal valid C3D block structure: pad to data_start_block * 512 bytes
        target_len = max(512 * container.header.data_start_block, 512 * 2)
        current_len = f.tell()
        if current_len < target_len:
            f.write(b"\x00" * (target_len - current_len))

        # Write floating point point/analog frame stream
        scale = container.header.scale_factor
        is_float = scale < 0.0
        n_frames = container.header.frame_count

        for frame_idx in range(n_frames):
            for pt in container.points:
                if frame_idx < len(pt.coordinates_xyz):
                    x, y, z = pt.coordinates_xyz[frame_idx]
                    res = (
                        pt.residuals[frame_idx]
                        if frame_idx < len(pt.residuals)
                        else 0.0
                    )
                else:
                    x, y, z, res = 0.0, 0.0, 0.0, -1.0
                if is_float:
                    f.write(struct.pack("<4f", x, y, z, res))
                else:
                    # Integer scaled representation
                    s = abs(scale) if scale != 0.0 else 1.0
                    ix = int(round(x / s))
                    iy = int(round(y / s))
                    iz = int(round(z / s))
                    ires = int(round(res / s))
                    f.write(struct.pack("<4h", ix, iy, iz, ires))

            # Analog samples for this frame
            subsamples = container.header.analog_samples_per_frame
            for sub in range(subsamples):
                for ch in container.analogs:
                    sample_idx = frame_idx * subsamples + sub
                    val = ch.values[sample_idx] if sample_idx < len(ch.values) else 0.0
                    if is_float:
                        f.write(struct.pack("<f", val))
                    else:
                        f.write(struct.pack("<h", int(round(val))))
