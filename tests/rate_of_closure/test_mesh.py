"""Tests for the STL clubhead mesh parser and head normalization.

Round-trips both STL flavours through the pure-numpy writer/parser and
pins the normalization numbers that the TypeScript twin
(``web/src/model/mesh.test.ts``) asserts verbatim, keeping the two
implementations in lock-step.
"""

from __future__ import annotations

import numpy as np
import pytest

from rate_of_closure._contracts import PreconditionError
from rate_of_closure.mesh import (
    HEAD_DEPTH_M,
    load_head_mesh,
    normalize_head,
    parse_stl,
    triangle_normals,
    write_ascii_stl,
    write_binary_stl,
)
from rate_of_closure.scripts.generate_example_head import (
    ASSET_PATH,
    build_example_head,
)

pytestmark = pytest.mark.unit

#: Shared ASCII parity fixture — byte-for-byte identical in the vitest
#: suite; both parsers must produce these exact vertices.
PARITY_ASCII = b"""solid parity
  facet normal 0 0 1
    outer loop
      vertex 0 0 0
      vertex 1.5e-1 0 0
      vertex 0 2.5E-1 0
    endloop
  endfacet
endsolid parity
"""


def _box_triangles(
    lo: tuple[float, float, float], hi: tuple[float, float, float]
) -> np.ndarray:
    """Cuboid as 12 triangles (two per face, consistent winding)."""
    x0, y0, z0 = lo
    x1, y1, z1 = hi
    c = np.array(
        [
            [x0, y0, z0],
            [x1, y0, z0],
            [x1, y1, z0],
            [x0, y1, z0],
            [x0, y0, z1],
            [x1, y0, z1],
            [x1, y1, z1],
            [x0, y1, z1],
        ]
    )
    quads = [
        (0, 3, 2, 1),  # z = z0
        (4, 5, 6, 7),  # z = z1
        (0, 1, 5, 4),  # y = y0
        (2, 3, 7, 6),  # y = y1
        (0, 4, 7, 3),  # x = x0
        (1, 2, 6, 5),  # x = x1
    ]
    tris = []
    for a, b, cc, d in quads:
        tris.append(c[[a, b, cc]])
        tris.append(c[[a, cc, d]])
    return np.array(tris)


class TestParser:
    def test_binary_round_trip(self) -> None:
        tris = build_example_head()
        parsed = parse_stl(write_binary_stl(tris))
        assert parsed.shape == tris.shape
        np.testing.assert_allclose(parsed, tris, atol=1e-6)

    def test_ascii_round_trip(self) -> None:
        tris = build_example_head()
        parsed = parse_stl(write_ascii_stl(tris))
        np.testing.assert_allclose(parsed, tris, atol=1e-6)

    def test_binary_wins_even_with_solid_header(self) -> None:
        """A binary file whose header starts with 'solid' still parses."""
        tris = _box_triangles((0, 0, 0), (1, 1, 1))
        parsed = parse_stl(write_binary_stl(tris, header="solid tricky header"))
        np.testing.assert_allclose(parsed, tris, atol=1e-6)

    def test_ascii_parity_fixture(self) -> None:
        parsed = parse_stl(PARITY_ASCII)
        expected = np.array([[[0, 0, 0], [0.15, 0, 0], [0, 0.25, 0]]])
        np.testing.assert_allclose(parsed, expected)

    def test_rejects_empty_and_garbage(self) -> None:
        with pytest.raises(PreconditionError):
            parse_stl(b"")
        with pytest.raises(PreconditionError):
            parse_stl(b"not an stl at all")
        # Truncated binary falls through to the ASCII path and fails.
        with pytest.raises(PreconditionError):
            parse_stl(write_binary_stl(_box_triangles((0, 0, 0), (1, 1, 1)))[:100])


class TestNormals:
    def test_unit_length_and_direction(self) -> None:
        tris = np.array([[[0, 0, 0], [1, 0, 0], [0, 1, 0]]], dtype=float)
        normals = triangle_normals(tris)
        np.testing.assert_allclose(normals, [[0, 0, 1]])

    def test_degenerate_gets_zero_normal(self) -> None:
        tris = np.array([[[0, 0, 0], [1, 1, 1], [2, 2, 2]]], dtype=float)
        np.testing.assert_allclose(triangle_normals(tris), [[0, 0, 0]])


class TestNormalizeHead:
    """Pinned numbers — mirrored verbatim in web/src/model/mesh.test.ts."""

    def test_cuboid_pinned_mapping(self) -> None:
        # Extents (2, 6, 4): largest (y=6) -> z, middle (z=4) -> x,
        # smallest (x=2) -> y; scale = 0.11 / 4 = 0.0275.
        tris = normalize_head(_box_triangles((0, 0, 0), (2, 6, 4)))
        flat = tris.reshape(-1, 3)
        np.testing.assert_allclose(
            flat.max(axis=0) - flat.min(axis=0), [0.11, 0.055, 0.165]
        )
        np.testing.assert_allclose(
            (flat.max(axis=0) + flat.min(axis=0)) / 2, [0, 0, 0], atol=1e-15
        )
        # The old-frame origin corner lands at this exact point.
        corner = np.array([-0.055, -0.0275, -0.0825])
        assert (np.abs(flat - corner).sum(axis=1) < 1e-12).any()

    def test_degenerate_triangles_are_dropped(self) -> None:
        tris = _box_triangles((0, 0, 0), (2, 6, 4))
        degenerate = np.zeros((1, 3, 3))
        out = normalize_head(np.concatenate([tris, degenerate]))
        assert out.shape[0] == tris.shape[0]

    def test_all_degenerate_rejected(self) -> None:
        with pytest.raises(PreconditionError):
            normalize_head(np.zeros((3, 3, 3)))

    def test_flat_mesh_rejected(self) -> None:
        flat_tri = np.array([[[0, 0, 0], [1, 0, 0], [0, 1, 0]]], dtype=float)
        with pytest.raises(PreconditionError):
            normalize_head(flat_tri)


class TestExampleAsset:
    def test_asset_is_current_and_loads(self) -> None:
        """The shipped STL matches a fresh deterministic regeneration."""
        expected = write_binary_stl(
            build_example_head(), header="rate_of_closure example head"
        )
        assert ASSET_PATH.read_bytes() == expected

    def test_example_head_normalizes_near_identity(self) -> None:
        mesh = load_head_mesh(ASSET_PATH)
        flat = mesh.triangles.reshape(-1, 3)
        extents = flat.max(axis=0) - flat.min(axis=0)
        np.testing.assert_allclose(extents, [HEAD_DEPTH_M, 0.062, 0.124], atol=1e-6)
        norms = np.linalg.norm(mesh.normals, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-9)
