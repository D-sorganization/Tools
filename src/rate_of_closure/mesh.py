"""STL clubhead meshes for bounded flat-shaded display normalization.

Both STL
flavours are handled with nothing beyond numpy: binary (80-byte header,
uint32 triangle count, 50-byte records) and ASCII (``solid``/``facet``/
``vertex`` text). A user-supplied mesh arrives in arbitrary units and
orientation, so :func:`normalize_head` maps it into a bounded display
envelope (AffineDrift frame: x target, y up, z right). STL units, physical
face direction, and original handedness are not encoded or inferred:

1. Degenerate (zero-area) triangles are dropped.
2. Axes are permuted by bounding-box extent — largest to z (heel-toe
   width), middle to x (face-to-back depth), smallest to y (crown
   height), with a compensating sign that keeps the transform proper-handed.
3. The bounding box is centered on the origin and scaled uniformly so
   the depth (x extent) equals :data:`HEAD_DEPTH_M`.

The TypeScript twin (``web/src/model/mesh.ts``) implements the same
rules and is parity-tested against the numbers pinned here.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ._contracts import ensure, require

__all__ = [
    "HEAD_DEPTH_M",
    "MAX_HEAD_SPAN_M",
    "MAX_IMPORTED_MESH_TRIANGLES",
    "MAX_RENDER_MESH_TRIANGLES",
    "MAX_STL_BYTES",
    "HeadMesh",
    "load_head_mesh",
    "normalize_head",
    "parse_stl",
    "snapshot_head_mesh",
    "triangle_normals",
    "write_ascii_stl",
    "write_binary_stl",
]

#: Canonical face-to-back depth of the head envelope [m]; matches the
#: procedural wireframe's ``_BODY_DEPTH`` in the PyQt6 club view.
HEAD_DEPTH_M = 0.11
MAX_HEAD_SPAN_M = 0.33
MAX_STL_BYTES = 2 * 1024 * 1024
MAX_IMPORTED_MESH_TRIANGLES = 2_048
MAX_RENDER_MESH_TRIANGLES = 4_096

_BINARY_HEADER_BYTES = 80
_BINARY_RECORD = np.dtype(
    [("normal", "<f4", (3,)), ("vertices", "<f4", (3, 3)), ("attr", "<u2")]
)
_ASCII_VERTEX = re.compile(r"vertex\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)")
_MIN_AREA = 1e-15


@dataclass(frozen=True)
class HeadMesh:
    """A normalized clubhead mesh ready for rendering.

    Attributes:
        triangles: ``(n, 3, 3)`` float64 vertex array in the canonical
            head frame (meters; bounding box centered on the origin).
        normals: ``(n, 3)`` unit normals recomputed from the winding.
    """

    triangles: np.ndarray
    normals: np.ndarray

    def __post_init__(self) -> None:
        raw_triangles = np.asarray(self.triangles)
        raw_normals = np.asarray(self.normals)
        require(
            raw_triangles.dtype.kind in "fiu" and raw_normals.dtype.kind in "fiu",
            "mesh geometry must use real numeric values",
        )
        triangles = np.array(self.triangles, dtype=np.float64, copy=True)
        require(
            triangles.ndim == 3 and triangles.shape[1:] == (3, 3),
            "mesh triangles must be (n, 3, 3)",
        )
        require(
            0 < triangles.shape[0] <= MAX_RENDER_MESH_TRIANGLES,
            "mesh must contain 1 to 4,096 triangles",
        )
        supplied_normals = raw_normals
        require(
            supplied_normals.shape == (triangles.shape[0], 3),
            "mesh normals must be (n, 3)",
        )
        require(
            bool(np.isfinite(triangles).all() and np.isfinite(supplied_normals).all()),
            "mesh geometry must be finite",
        )
        normals = triangle_normals(triangles)
        triangles = np.frombuffer(triangles.tobytes(), dtype=np.float64).reshape(
            triangles.shape
        )
        normals = np.frombuffer(normals.tobytes(), dtype=np.float64).reshape(
            normals.shape
        )
        object.__setattr__(self, "triangles", triangles)
        object.__setattr__(self, "normals", normals)


def parse_stl(data: bytes) -> np.ndarray:
    """Parse STL bytes (binary or ASCII) into ``(n, 3, 3)`` triangles.

    Format detection follows the standard heuristic: a file whose length
    matches the binary record layout is binary (binary headers may start
    with ``solid`` too), otherwise a leading ``solid`` keyword means
    ASCII.
    """
    require(isinstance(data, (bytes, bytearray)), "data must be bytes")
    require(len(data) > 0, "data must not be empty")
    require(len(data) <= MAX_STL_BYTES, "STL must not exceed 2 MiB")
    triangles = (
        _parse_binary(bytes(data))
        if _looks_binary(bytes(data))
        else _parse_ascii(bytes(data))
    )
    ensure(triangles.shape[0] > 0, "STL contained no triangles")
    ensure(bool(np.isfinite(triangles).all()), "STL vertices must be finite")
    return triangles


def _looks_binary(data: bytes) -> bool:
    if len(data) < _BINARY_HEADER_BYTES + 4:
        return False
    count = int(
        np.frombuffer(data, dtype="<u4", count=1, offset=_BINARY_HEADER_BYTES)[0]
    )
    matches_layout = (
        len(data) == _BINARY_HEADER_BYTES + 4 + count * _BINARY_RECORD.itemsize
    )
    if matches_layout:
        require(
            count <= MAX_IMPORTED_MESH_TRIANGLES,
            "STL must not exceed 2,048 triangles",
        )
    return matches_layout


def _parse_binary(data: bytes) -> np.ndarray:
    count = int(
        np.frombuffer(data, dtype="<u4", count=1, offset=_BINARY_HEADER_BYTES)[0]
    )
    require(count > 0, "binary STL declares zero triangles")
    records = np.frombuffer(
        data, dtype=_BINARY_RECORD, count=count, offset=_BINARY_HEADER_BYTES + 4
    )
    return records["vertices"].astype(np.float64)


def _parse_ascii(data: bytes) -> np.ndarray:
    try:
        text = data.decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        require(False, "ASCII STL must be valid UTF-8")
        raise AssertionError("unreachable") from error
    require(
        text.lstrip()[:5].lower() == "solid",
        "not a valid STL: neither binary layout nor ASCII 'solid'",
    )
    vertices: list[tuple[float, float, float]] = []
    for match in _ASCII_VERTEX.finditer(text):
        require(
            len(vertices) < MAX_IMPORTED_MESH_TRIANGLES * 3,
            "STL must not exceed 2,048 triangles",
        )
        values = match.groups()
        vertices.append((float(values[0]), float(values[1]), float(values[2])))
    require(len(vertices) > 0, "ASCII STL contains no vertex lines")
    require(
        len(vertices) % 3 == 0,
        "ASCII STL vertex count must be a multiple of 3",
        len(vertices),
    )
    flat = np.asarray(vertices, dtype=np.float64)
    return flat.reshape(-1, 3, 3)


def triangle_normals(triangles: np.ndarray) -> np.ndarray:
    """Unit normals from vertex winding, ``(n, 3)``.

    Degenerate triangles get a zero normal rather than NaN so callers
    can filter or shade them safely.
    """
    tris = np.asarray(triangles, dtype=np.float64)
    require(tris.ndim == 3 and tris.shape[1:] == (3, 3), "triangles must be (n, 3, 3)")
    cross = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    length = np.linalg.norm(cross, axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        normals = np.where(length > _MIN_AREA, cross / length, 0.0)
    return np.asarray(normals, dtype=np.float64)


def normalize_head(triangles: np.ndarray, depth_m: float = HEAD_DEPTH_M) -> np.ndarray:
    """Map arbitrary triangles onto the canonical head envelope.

    See the module docstring for the three rules (drop degenerates,
    permute axes by extent, center and scale to ``depth_m``).
    """
    require(depth_m > 0.0, "depth_m must be positive", depth_m)
    tris = np.asarray(triangles, dtype=np.float64)
    require(tris.ndim == 3 and tris.shape[1:] == (3, 3), "triangles must be (n, 3, 3)")
    require(bool(np.isfinite(tris).all()), "triangles must be finite")

    require(
        0 < tris.shape[0] <= MAX_RENDER_MESH_TRIANGLES,
        "mesh must contain 1 to 4,096 triangles",
    )
    magnitude = float(np.max(np.abs(tris)))
    require(
        np.isfinite(magnitude) and magnitude > 0.0,
        "mesh has no non-degenerate triangles",
    )
    scaled = tris / magnitude
    areas = np.linalg.norm(
        np.cross(scaled[:, 1] - scaled[:, 0], scaled[:, 2] - scaled[:, 0]), axis=1
    )
    scaled = scaled[areas > _MIN_AREA]
    tris = tris[areas > _MIN_AREA]
    require(tris.shape[0] > 0, "mesh has no non-degenerate triangles")

    flat = scaled.reshape(-1, 3)
    extents = flat.max(axis=0) - flat.min(axis=0)
    require(bool((extents > 0.0).all()), "mesh must have volume on all axes", extents)

    # Stable extent ordering: [smallest, middle, largest] source axes.
    order = np.argsort(extents, kind="stable")
    # middle -> x (depth), smallest -> y (height), largest -> z (width).
    permutation = np.array([order[1], order[0], order[2]])
    scaled = scaled[:, :, permutation]
    inversions = sum(
        int(permutation[left] > permutation[right])
        for left in range(3)
        for right in range(left + 1, 3)
    )
    if inversions % 2:
        scaled[:, :, 2] *= -1.0

    flat = scaled.reshape(-1, 3)
    center = (flat.max(axis=0) + flat.min(axis=0)) / 2.0
    scale = depth_m / extents[order[1]]
    normalized = (scaled - center) * scale
    normalized[normalized == 0.0] = 0.0

    span = normalized.reshape(-1, 3).max(axis=0) - normalized.reshape(-1, 3).min(axis=0)
    ensure(bool(np.isclose(span[0], depth_m, rtol=1e-9)), "depth must normalize")
    ensure(span[2] >= span[0] >= span[1], "extent ordering z >= x >= y must hold")
    require(
        bool((span <= MAX_HEAD_SPAN_M).all()),
        "normalized mesh span exceeds 0.330 m",
    )
    return np.asarray(normalized, dtype=np.float64)


def load_head_mesh(path: str | Path, depth_m: float = HEAD_DEPTH_M) -> HeadMesh:
    """Parse an STL file and normalize it into a renderable head mesh."""
    stl_path = Path(path)
    require(stl_path.suffix.lower() == ".stl", "mesh file must use the .stl suffix")
    require(stl_path.is_file(), "STL path must be an existing file", str(stl_path))
    with stl_path.open("rb") as stream:
        data = stream.read(MAX_STL_BYTES + 1)
    require(len(data) <= MAX_STL_BYTES, "STL must not exceed 2 MiB")
    triangles = normalize_head(parse_stl(data), depth_m)
    return HeadMesh(triangles=triangles, normals=triangle_normals(triangles))


def snapshot_head_mesh(mesh: HeadMesh) -> HeadMesh:
    """Validate and defensively freeze a mesh at an adoption boundary."""
    return HeadMesh(mesh.triangles, mesh.normals)


def write_binary_stl(triangles: np.ndarray, header: str = "rate_of_closure") -> bytes:
    """Serialize ``(n, 3, 3)`` triangles as a binary STL byte string."""
    tris = np.asarray(triangles, dtype=np.float64)
    require(tris.ndim == 3 and tris.shape[1:] == (3, 3), "triangles must be (n, 3, 3)")
    require(tris.shape[0] > 0, "cannot write an empty STL")
    records = np.zeros(tris.shape[0], dtype=_BINARY_RECORD)
    records["normal"] = triangle_normals(tris).astype(np.float32)
    records["vertices"] = tris.astype(np.float32)
    head = header.encode("ascii", errors="replace")[:_BINARY_HEADER_BYTES]
    head = head.ljust(_BINARY_HEADER_BYTES, b"\0")
    count = np.array([tris.shape[0]], dtype="<u4")
    return head + count.tobytes() + records.tobytes()


def write_ascii_stl(triangles: np.ndarray, name: str = "rate_of_closure") -> bytes:
    """Serialize ``(n, 3, 3)`` triangles as an ASCII STL byte string."""
    tris = np.asarray(triangles, dtype=np.float64)
    require(tris.ndim == 3 and tris.shape[1:] == (3, 3), "triangles must be (n, 3, 3)")
    require(tris.shape[0] > 0, "cannot write an empty STL")
    normals = triangle_normals(tris)
    lines = [f"solid {name}"]
    for tri, normal in zip(tris, normals, strict=True):
        lines.append(f"  facet normal {normal[0]:.9e} {normal[1]:.9e} {normal[2]:.9e}")
        lines.append("    outer loop")
        for vertex in tri:
            lines.append(
                f"      vertex {vertex[0]:.9e} {vertex[1]:.9e} {vertex[2]:.9e}"
            )
        lines.append("    endloop")
        lines.append("  endfacet")
    lines.append(f"endsolid {name}")
    return ("\n".join(lines) + "\n").encode("ascii")
