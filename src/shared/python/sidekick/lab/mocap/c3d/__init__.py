"""C3D biomechanical data exchange (TOOLS-M9 #4716)."""

from .converters import compute_center_of_pressure, unit_scale_factor
from .reader import (
    C3D_HEADER_LENGTH,
    C3D_HEADER_MAGIC_BYTE,
    C3DContainer,
    parse_c3d_header,
    validate_c3d_header_magic,
)
from .schema import (
    BIOMECHANICAL_MARKER_MAX_M,
    BIOMECHANICAL_MARKER_MIN_M,
    C3DAnalogChannel,
    C3DEvent,
    C3DForcePlatform,
    C3DHeader,
    C3DPointChannel,
)
from .writer import serialize_c3d_header, write_c3d_file

__all__ = [
    "BIOMECHANICAL_MARKER_MAX_M",
    "BIOMECHANICAL_MARKER_MIN_M",
    "C3DAnalogChannel",
    "C3DContainer",
    "C3DEvent",
    "C3DForcePlatform",
    "C3DHeader",
    "C3DPointChannel",
    "C3D_HEADER_LENGTH",
    "C3D_HEADER_MAGIC_BYTE",
    "compute_center_of_pressure",
    "parse_c3d_header",
    "serialize_c3d_header",
    "unit_scale_factor",
    "validate_c3d_header_magic",
    "write_c3d_file",
]
