"""Golden and adversarial tests for bounded confidence surfaces."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from matplotlib.figure import Figure

from rate_of_closure.ui.pyqt6 import variation_geometry_rendering
from rate_of_closure.ui.pyqt6.variation_geometry_rendering import (
    draw_confidence_ellipsoid_mesh,
)
from rate_of_closure.ui.pyqt6.variation_plot_helpers import equal_3d_axes
from rate_of_closure.variation import confidence_ellipsoid_mesh as mesh_authority
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
        np.asarray([[[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]]),
        np.asarray([[2.0, 1.0, 0.5]]),
        (ESTIMABLE,),
        "app_frame:x_target,y_up,z_right",
        longitude_segments=4,
        latitude_segments=2,
    )
    captured: dict[str, object] = {}

    class CapturingCollection:
        def __init__(self, triangles: np.ndarray, **style: object) -> None:
            captured["triangles"] = triangles
            captured["style"] = style

        def set_label(self, label: str) -> None:
            captured["label"] = label

    class CapturingAxes:
        def add_collection3d(self, collection: object) -> None:
            captured["collection"] = collection

    original = variation_geometry_rendering.Poly3DCollection
    variation_geometry_rendering.Poly3DCollection = CapturingCollection  # type: ignore[assignment]
    axes = CapturingAxes()
    try:
        draw_confidence_ellipsoid_mesh(axes, mesh)
    finally:
        variation_geometry_rendering.Poly3DCollection = original

    app_triangles = mesh.vertices_m[mesh.triangles]
    np.testing.assert_allclose(captured["triangles"], app_triangles[:, :, [0, 2, 1]])
    assert captured["style"] == {
        "facecolor": "#22d3ee",
        "edgecolor": "#67e8f9",
        "linewidth": 0.25,
        "alpha": 0.16,
        "zsort": "average",
    }
    assert captured["label"] == "_nolegend_"


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


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("longitude_segments", True),
        ("longitude_segments", 2),
        ("longitude_segments", 12.5),
        ("longitude_segments", 13),
        ("latitude_segments", True),
        ("latitude_segments", 1),
        ("latitude_segments", 6.5),
        ("latitude_segments", 7),
        ("max_ellipsoids", True),
        ("max_ellipsoids", -1),
        ("max_ellipsoids", 1.5),
        ("max_ellipsoids", 49),
        ("max_vertices", True),
        ("max_vertices", -1),
        ("max_vertices", 100.5),
        ("max_vertices", 2_977),
        ("max_triangles", True),
        ("max_triangles", -1),
        ("max_triangles", 100.5),
        ("max_triangles", 5_761),
    ],
)
def test_tessellation_and_budget_fields_are_strictly_hard_capped(
    field: str, value: object
) -> None:
    kwargs = {field: value}
    with pytest.raises(ContractViolationError):
        build_confidence_ellipsoid_mesh(
            np.zeros((2, 3)),
            np.broadcast_to(np.eye(3), (2, 3, 3)),
            np.ones((2, 3)),
            (ESTIMABLE, ESTIMABLE),
            "app_frame:x_target,y_up,z_right",
            **kwargs,  # type: ignore[arg-type]
        )


def test_zero_capacity_returns_empty_without_tessellation_allocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_if_called(*_args: object) -> tuple[np.ndarray, np.ndarray]:
        raise AssertionError("unit sphere must not be allocated at zero capacity")

    monkeypatch.setattr(mesh_authority, "_unit_sphere", fail_if_called)
    mesh = build_confidence_ellipsoid_mesh(
        np.zeros((1, 3)),
        np.eye(3)[None],
        np.ones((1, 3)),
        (ESTIMABLE,),
        "app_frame:x_target,y_up,z_right",
        max_vertices=0,
    )
    assert mesh.vertices_m.shape == (0, 3)
    assert mesh.triangles.shape == (0, 3)
    assert mesh.vertices_per_ellipsoid == 62
    assert mesh.triangles_per_ellipsoid == 120


def test_transformed_vertex_overflow_fails_closed() -> None:
    with pytest.raises(ContractViolationError, match="vertices must be finite"):
        build_confidence_ellipsoid_mesh(
            np.full((1, 3), 1.0e308),
            np.eye(3)[None],
            np.full((1, 3), 1.0e308),
            (ESTIMABLE,),
            "app_frame:x_target,y_up,z_right",
        )
