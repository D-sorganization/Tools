"""Golden and adversarial tests for bounded confidence surfaces."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from rate_of_closure.ui.pyqt6.variation_geometry_rendering import (
    draw_confidence_ellipsoid_mesh,
)
from rate_of_closure.ui.pyqt6.variation_plot_helpers import equal_3d_axes
from rate_of_closure.variation.confidence_ellipsoid_mesh import (
    MAX_ELLIPSOID_TRIANGLES,
    MAX_ELLIPSOID_VERTICES,
    MAX_RENDERED_ELLIPSOIDS,
    build_confidence_ellipsoid_mesh,
)
from shared.python.contracts import ContractViolationError
from shared.python.swing_sim.variation import ESTIMABLE

_FIXTURE = (
    Path(__file__).parents[2]
    / "src/rate_of_closure/web/src/model/__fixtures__"
    / "confidence_ellipsoid_mesh_golden_v1.json"
)


def test_mesh_matches_cross_toolkit_golden_and_excludes_rank_deficiency() -> None:
    fixture = json.loads(_FIXTURE.read_text(encoding="utf-8"))
    budget = fixture["budget"]
    mesh = build_confidence_ellipsoid_mesh(
        np.asarray(fixture["centersM"]),
        np.asarray(fixture["principalFrames"]).transpose(0, 2, 1),
        np.asarray(fixture["semiAxisLengthsM"]),
        tuple(fixture["adequacy"]),
        fixture["coordinateFrame"],
        longitude_segments=budget["longitudeSegments"],
        latitude_segments=budget["latitudeSegments"],
        max_ellipsoids=budget["maxEllipsoids"],
        max_vertices=budget["maxVertices"],
        max_triangles=budget["maxTriangles"],
    )
    assert mesh.sample_indices == tuple(fixture["sampleIndices"])
    np.testing.assert_allclose(mesh.vertices_m, fixture["verticesM"], atol=1e-14)
    np.testing.assert_array_equal(mesh.triangles, fixture["triangles"])


def test_default_budget_is_bounded_and_retains_temporal_endpoints() -> None:
    samples = 1_000
    mesh = build_confidence_ellipsoid_mesh(
        np.column_stack((np.arange(samples), np.zeros(samples), np.zeros(samples))),
        np.broadcast_to(np.eye(3), (samples, 3, 3)),
        np.ones((samples, 3)),
        (ESTIMABLE,) * samples,
        "app_frame:x_target,y_up,z_right",
    )
    assert len(mesh.sample_indices) == MAX_RENDERED_ELLIPSOIDS
    assert mesh.sample_indices[0] == 0 and mesh.sample_indices[-1] == samples - 1
    assert mesh.vertices_m.shape[0] <= MAX_ELLIPSOID_VERTICES
    assert mesh.triangles.shape[0] <= MAX_ELLIPSOID_TRIANGLES


def test_matplotlib_renderer_maps_app_y_up_to_visual_z() -> None:
    mesh = build_confidence_ellipsoid_mesh(
        np.zeros((1, 3)),
        np.eye(3)[None],
        np.ones((1, 3)),
        (ESTIMABLE,),
        "app_frame:x_target,y_up,z_right",
        longitude_segments=4,
        latitude_segments=2,
    )
    axes = Figure().add_subplot(projection="3d")
    draw_confidence_ellipsoid_mesh(axes, mesh)
    collection = next(
        item for item in axes.collections if isinstance(item, Poly3DCollection)
    )
    assert collection.get_alpha() == pytest.approx(0.16)
    assert collection.get_label() == "_nolegend_"


def test_equal_axes_keep_surface_vertices_in_the_camera_bounds() -> None:
    axes = Figure().add_subplot(projection="3d")
    overlay = SimpleNamespace(positions_m=np.asarray([[[0.0, 0.0, 0.0]]]))

    equal_3d_axes(axes, overlay, np.asarray([[10.0, 0.0, 0.0]]))

    assert axes.get_xlim()[1] >= 10.0


@pytest.mark.parametrize(
    ("frame", "axes", "semi_axes"),
    [
        ("other", np.eye(3)[None], np.ones((1, 3))),
        ("app_frame:x_target,y_up,z_right", (2 * np.eye(3))[None], np.ones((1, 3))),
        ("app_frame:x_target,y_up,z_right", np.eye(3)[None], np.zeros((1, 3))),
    ],
)
def test_estimable_geometry_fails_closed(
    frame: str, axes: np.ndarray, semi_axes: np.ndarray
) -> None:
    with pytest.raises(ContractViolationError):
        build_confidence_ellipsoid_mesh(
            np.zeros((1, 3)), axes, semi_axes, (ESTIMABLE,), frame
        )
