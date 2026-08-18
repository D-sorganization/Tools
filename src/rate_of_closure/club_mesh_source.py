"""Immutable, provenance-bound clubhead display sources."""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from rate_of_closure._contracts import require
from rate_of_closure.mesh import (
    HEAD_DEPTH_M,
    MAX_IMPORTED_MESH_TRIANGLES,
    MAX_STL_BYTES,
    HeadMesh,
    normalize_head,
    parse_stl,
    snapshot_head_mesh,
    triangle_normals,
)

IMPORT_NORMALIZATION_REVISION = "roc-stl-display-v1"
_MAX_SOURCE_NAME_CHARS = 64
_UNSAFE_NAME = re.compile(r"[\x00-\x1f\x7f-\x9f\u202a-\u202e\u2066-\u2069]")


@dataclass(frozen=True)
class ClubMeshSource:
    """One fully validated clubhead source adopted by the renderer."""

    generation: int
    kind: str
    mesh: HeadMesh | None
    hosel: tuple[float, float, float] | None
    geometric_centroid: tuple[float, float, float] | None
    status: str
    sha256: str | None
    raw_bytes: int | None = None
    raw_triangles: int | None = None
    retained_triangles: int | None = None
    normalization_revision: str | None = None

    def __post_init__(self) -> None:
        """Reject inconsistent source metadata before renderer adoption."""
        require(
            type(self.generation) is int and 0 <= self.generation <= 2**53 - 1,
            "generation must be a nonnegative safe integer",
        )
        require(
            self.kind in {"procedural", "generated", "imported"},
            "unknown mesh source kind",
        )
        require(
            (self.kind == "procedural") == (self.mesh is None),
            "source kind and mesh disagree",
        )
        for point in (self.hosel, self.geometric_centroid):
            require(
                point is None
                or (
                    len(point) == 3
                    and all(
                        type(value) in {int, float} and math.isfinite(value)
                        for value in point
                    )
                ),
                "source points must be finite 3-vectors",
            )
        provenance = (
            self.sha256,
            self.raw_bytes,
            self.raw_triangles,
            self.retained_triangles,
            self.normalization_revision,
        )
        if self.kind == "procedural":
            require(
                self.hosel is None and self.geometric_centroid is None,
                "procedural metadata must be absent",
            )
            require(
                all(value is None for value in provenance),
                "procedural provenance must be absent",
            )
        if self.kind == "generated":
            require(
                self.hosel is not None and self.geometric_centroid is not None,
                "generated metadata is required",
            )
            require(
                all(value is None for value in provenance),
                "generated import provenance must be absent",
            )
        if self.kind == "imported":
            mesh = cast(HeadMesh, self.mesh)
            raw_triangles = cast(int, self.raw_triangles)
            retained_triangles = cast(int, self.retained_triangles)
            require(
                self.hosel is None and self.geometric_centroid is None,
                "imported metadata must be absent",
            )
            require(
                self.sha256 is not None
                and re.fullmatch(r"[0-9a-f]{64}", self.sha256) is not None,
                "imported source needs SHA-256",
            )
            require(
                type(self.raw_bytes) is int and 0 < self.raw_bytes <= MAX_STL_BYTES,
                "imported source needs byte provenance",
            )
            require(
                type(self.raw_triangles) is int
                and 0 < self.raw_triangles <= MAX_IMPORTED_MESH_TRIANGLES,
                "imported source needs raw triangle provenance",
            )
            require(
                type(self.retained_triangles) is int
                and 0 < retained_triangles <= raw_triangles,
                "imported retained triangle provenance is invalid",
            )
            require(
                retained_triangles == mesh.triangles.shape[0],
                "retained triangle provenance disagrees",
            )
            require(
                self.normalization_revision == IMPORT_NORMALIZATION_REVISION,
                "normalization revision disagrees",
            )


def _clean_source_name(name: str) -> str:
    basename = Path(name.replace("\\", "/")).name or "mesh.stl"
    return _UNSAFE_NAME.sub("�", basename)[:_MAX_SOURCE_NAME_CHARS]


def procedural_mesh_source(generation: int = 0) -> ClubMeshSource:
    """Return the fixed procedural wireframe source."""
    return ClubMeshSource(
        generation,
        "procedural",
        None,
        None,
        None,
        f"Procedural head; fixed {HEAD_DEPTH_M:.3f} m face-to-back display envelope",
        None,
    )


def generated_mesh_source(
    mesh: HeadMesh,
    label: str,
    generation: int,
    *,
    hosel: tuple[float, float, float],
    geometric_centroid: tuple[float, float, float],
) -> ClubMeshSource:
    """Snapshot trusted generated SI geometry and its geometric metadata."""
    require(
        all(
            type(value) in {int, float} and math.isfinite(value)
            for point in (hosel, geometric_centroid)
            for value in point
        ),
        "generated points must use real finite numbers",
    )
    return ClubMeshSource(
        generation,
        "generated",
        snapshot_head_mesh(mesh),
        (float(hosel[0]), float(hosel[1]), float(hosel[2])),
        (
            float(geometric_centroid[0]),
            float(geometric_centroid[1]),
            float(geometric_centroid[2]),
        ),
        f"Generated representative {_clean_source_name(label)}; authored SI geometry",
        None,
    )


def imported_mesh_source(path: str | Path, generation: int) -> ClubMeshSource:
    """Read and snapshot one bounded STL with byte-bound provenance."""
    stl_path = Path(path)
    require(stl_path.suffix.lower() == ".stl", "mesh file must use the .stl suffix")
    require(stl_path.is_file(), "STL path must be an existing file", str(stl_path))
    with stl_path.open("rb") as stream:
        raw = stream.read(MAX_STL_BYTES + 1)
    require(len(raw) <= MAX_STL_BYTES, "STL must not exceed 2 MiB")
    parsed = parse_stl(raw)
    raw_triangles = int(parsed.shape[0])
    triangles = normalize_head(parsed)
    mesh = HeadMesh(triangles, triangle_normals(triangles))
    digest = hashlib.sha256(raw).hexdigest()
    status = (
        f"Imported {_clean_source_name(stl_path.name)}; {len(raw)} bytes; "
        f"{raw_triangles} raw / {mesh.triangles.shape[0]} retained triangles; "
        f"SHA-256 {digest[:12]}…; "
        f"{IMPORT_NORMALIZATION_REVISION}: unitless axes permuted by stable extent "
        "and sign adjusted only to preserve handedness, "
        "0.110 m depth, span ≤0.330 m; no physical registration or mass "
        "centroid inferred"
    )
    return ClubMeshSource(
        generation,
        "imported",
        mesh,
        None,
        None,
        status,
        digest,
        len(raw),
        raw_triangles,
        int(mesh.triangles.shape[0]),
        IMPORT_NORMALIZATION_REVISION,
    )
