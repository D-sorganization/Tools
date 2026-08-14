from __future__ import annotations

import numpy as np
import pytest

from rate_of_closure.club import (
    CLUB_LIBRARY,
    head_cog,
    hosel_point,
    parametric_head_mesh,
)
from rate_of_closure.club_mesh_source import (
    IMPORT_NORMALIZATION_REVISION,
    ClubMeshSource,
    generated_mesh_source,
    imported_mesh_source,
    procedural_mesh_source,
)
from rate_of_closure.mesh import write_binary_stl


def _solid() -> np.ndarray:
    vertices = np.array(
        [
            [0, 0, 0],
            [2, 0, 0],
            [2, 4, 0],
            [0, 4, 0],
            [0, 0, 6],
            [2, 0, 6],
            [2, 4, 6],
            [0, 4, 6],
        ],
        dtype=float,
    )
    faces = [
        (0, 3, 2),
        (0, 2, 1),
        (4, 5, 6),
        (4, 6, 7),
        (0, 1, 5),
        (0, 5, 4),
        (2, 3, 7),
        (2, 7, 6),
        (0, 4, 7),
        (0, 7, 3),
        (1, 2, 6),
        (1, 6, 5),
    ]
    return np.asarray([[vertices[index] for index in face] for face in faces])


def test_sources_distinguish_fixed_and_authored_geometry() -> None:
    procedural = procedural_mesh_source(2)
    assert "0.110 m" in procedural.status
    spec = CLUB_LIBRARY["Mallet Putter"]
    report = head_cog(spec)
    generated = generated_mesh_source(
        parametric_head_mesh(spec),
        spec.name,
        3,
        hosel=hosel_point(spec),
        geometric_centroid=report.cog,
    )
    assert generated.mesh is not None
    assert generated.mesh.triangles.shape[0] == 2_176
    assert "authored SI geometry" in generated.status


def test_imported_identity_is_digest_not_filename_and_never_infers_centroid(
    tmp_path,
) -> None:  # type: ignore[no-untyped-def]
    path = tmp_path / "\u202ehead.stl"
    path.write_bytes(write_binary_stl(_solid()))
    source = imported_mesh_source(path, 4)
    assert source.sha256 is not None and len(source.sha256) == 64
    assert "\u202e" not in source.status
    assert source.geometric_centroid is None
    assert source.raw_bytes == path.stat().st_size
    assert source.raw_triangles == 12
    assert source.retained_triangles == 12
    assert source.normalization_revision == IMPORT_NORMALIZATION_REVISION
    assert "no physical registration or mass centroid inferred" in source.status
    assert source.mesh is not None and not source.mesh.triangles.flags.writeable


@pytest.mark.parametrize("generation", [True, -1, 2**53])
def test_source_generation_rejects_non_safe_integers(generation: object) -> None:
    with pytest.raises(Exception, match="generation"):
        procedural_mesh_source(generation)  # type: ignore[arg-type]


def test_source_kinds_reject_cross_kind_metadata() -> None:
    with pytest.raises(Exception, match="provenance"):
        ClubMeshSource(0, "procedural", None, None, None, "bad", None, 1)


def test_source_dataclass_rejects_coercive_generated_points() -> None:
    spec = CLUB_LIBRARY["Mallet Putter"]
    mesh = parametric_head_mesh(spec)
    with pytest.raises(Exception, match="finite 3-vectors"):
        ClubMeshSource(
            0,
            "generated",
            mesh,
            (True, 0.0, 0.0),  # type: ignore[arg-type]
            (0.0, 0.0, 0.0),
            "forged",
            None,
        )
