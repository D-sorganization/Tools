"""STL clubhead meshes: pure-numpy parser, writer, and head normalization.

Supports the optional photorealistic-clubhead rendering mode. Both STL
flavours are handled with nothing beyond numpy: binary (80-byte header,
uint32 triangle count, 50-byte records) and ASCII (``solid``/``facet``/
``vertex`` text). A user-supplied mesh arrives in arbitrary units and
orientation, so :func:`normalize_head` maps it onto the same canonical
envelope the procedural wireframe uses (AffineDrift frame: x target,
y up, z right):

1. Degenerate (zero-area) triangles are dropped.
2. Axes are permuted by bounding-box extent — largest to z (heel-toe
   width), middle to x (face-to-back depth), smallest to y (crown
   height) — the proportions of every driver head, so the face plate
   ends up facing +x.
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
    "HeadMesh",
    "load_head_mesh",
    "normalize_head",
    "parse_stl",
    "triangle_normals",
    "write_ascii_stl",
    "write_binary_stl",
]

#: Canonical face-to-back depth of the head envelope [m]; matches the
#: procedural wireframe's ``_BODY_DEPTH`` in the PyQt6 club view.
HEAD_DEPTH_M = 0.11

_BINARY_HEADER_BYTES = 80
_BINARY_RECORD: np.dtype[np.void] = np.dtype(
    [("normal", "<f4", (3,)), ("vertices", "<f4", (3, 3)), ("attr", "<u2")]
)
_ASCII_VERTEX = re.compile(rb"vertex\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)")
_MIN_AREA = 1e-20


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


def parse_stl(data: bytes) -> np.ndarray:
    """Parse STL bytes (binary or ASCII) into ``(n, 3, 3)`` triangles.

    Format detection follows the standard heuristic: a file whose length
    matches the binary record layout is binary (binary headers may start
    with ``solid`` too), otherwise a leading ``solid`` keyword means
    ASCII.
    """
    require(isinstance(data, (bytes, bytearray)), "data must be bytes")
    require(len(data) > 0, "data must not be empty")
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
    return len(data) == _BINARY_HEADER_BYTES + 4 + count * _BINARY_RECORD.itemsize


def _parse_binary(data: bytes) -> np.ndarray:
    count = int(
        np.frombuffer(data, dtype="<u4", count=1, offset=_BINARY_HEADER_BYTES)[0]
    )
    require(count > 0, "binary STL declares zero triangles")
    records = np.frombuffer(
        data, dtype=_BINARY_RECORD, count=count, offset=_BINARY_HEADER_BYTES + 4
    )
    vertices: np.ndarray = records["vertices"].astype(np.float64)
    return vertices


def _parse_ascii(data: bytes) -> np.ndarray:
    require(
        data.lstrip()[:5].lower() == b"solid",
        "not a valid STL: neither binary layout nor ASCII 'solid'",
    )
    matches = _ASCII_VERTEX.findall(data)
    require(len(matches) > 0, "ASCII STL contains no vertex lines")
    require(
        len(matches) % 3 == 0,
        "ASCII STL vertex count must be a multiple of 3",
        len(matches),
    )
    flat = np.array([[float(c) for c in m] for m in matches], dtype=np.float64)
    triangles: np.ndarray = flat.reshape(-1, 3, 3)
    return triangles


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
    unit_normals: np.ndarray = np.asarray(normals, dtype=np.float64)
    return unit_normals


def normalize_head(triangles: np.ndarray, depth_m: float = HEAD_DEPTH_M) -> np.ndarray:
    """Map arbitrary triangles onto the canonical head envelope.

    See the module docstring for the three rules (drop degenerates,
    permute axes by extent, center and scale to ``depth_m``).
    """
    require(depth_m > 0.0, "depth_m must be positive", depth_m)
    tris = np.asarray(triangles, dtype=np.float64)
    require(tris.ndim == 3 and tris.shape[1:] == (3, 3), "triangles must be (n, 3, 3)")
    require(bool(np.isfinite(tris).all()), "triangles must be finite")

    areas = np.linalg.norm(
        np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0]), axis=1
    )
    tris = tris[areas > _MIN_AREA]
    require(tris.shape[0] > 0, "mesh has no non-degenerate triangles")

    flat = tris.reshape(-1, 3)
    extents = flat.max(axis=0) - flat.min(axis=0)
    require(bool((extents > 0.0).all()), "mesh must have volume on all axes", extents)

    # Stable extent ordering: [smallest, middle, largest] source axes.
    order = np.argsort(extents, kind="stable")
    # middle -> x (depth), smallest -> y (height), largest -> z (width).
    permutation = np.array([order[1], order[0], order[2]])
    tris = tris[:, :, permutation]

    flat = tris.reshape(-1, 3)
    center = (flat.max(axis=0) + flat.min(axis=0)) / 2.0
    scale = depth_m / extents[order[1]]
    normalized = (tris - center) * scale

    span = normalized.reshape(-1, 3).max(axis=0) - normalized.reshape(-1, 3).min(axis=0)
    ensure(bool(np.isclose(span[0], depth_m, rtol=1e-9)), "depth must normalize")
    ensure(span[2] >= span[0] >= span[1], "extent ordering z >= x >= y must hold")
    canonical: np.ndarray = np.asarray(normalized, dtype=np.float64)
    return canonical


def load_head_mesh(path: str | Path, depth_m: float = HEAD_DEPTH_M) -> HeadMesh:
    """Parse an STL file and normalize it into a renderable head mesh."""
    stl_path = Path(path)
    require(stl_path.is_file(), "STL path must be an existing file", str(stl_path))
    triangles = normalize_head(parse_stl(stl_path.read_bytes()), depth_m)
    return HeadMesh(triangles=triangles, normals=triangle_normals(triangles))


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
    payload: bytes = head + count.tobytes() + records.tobytes()
    return payload


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
