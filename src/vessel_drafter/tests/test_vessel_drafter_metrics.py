import numpy as np
import pytest

from vessel_drafter.analysis.vessel_drafter_metrics import (
    build_material_metrics_report,
)
from vessel_drafter.models.vessel_drafter import DEFAULT_VESSEL_DRAFTER_LAYOUT
from vessel_drafter.preview.vessel_drafter_scene import build_vessel_3d_scene
from vessel_drafter.preview.vessel_drafter_view_options import (
    Vessel3DViewOptions,
)


def test_material_metrics_report_includes_refractory_totals() -> None:
    pytest.importorskip("build123d")
    report = build_material_metrics_report(DEFAULT_VESSEL_DRAFTER_LAYOUT)
    metrics_by_label = {item.label: item for item in report.component_metrics}

    assert metrics_by_label["hot_face_refractory"].volume_in3 > 0.0
    assert metrics_by_label["hot_face_refractory"].volume_ft3 > 0.0
    assert metrics_by_label["hot_face_refractory"].surface_area_ft2 > 0.0
    assert metrics_by_label["hot_face_refractory"].mass_lb > 0.0
    assert metrics_by_label["hot_face_refractory"].density_lb_per_ft3 > 0.0
    assert metrics_by_label["hot_face_refractory"].thermal_conductivity_w_per_mk > 0.0
    assert metrics_by_label["hot_face_refractory"].thermal_expansion_um_per_m_c > 0.0
    assert report.refractory_total_volume_in3 > 0.0
    assert report.refractory_total_volume_ft3 > 0.0
    assert report.refractory_total_surface_area_ft2 > 0.0
    assert report.refractory_total_mass_lb > 0.0
    assert report.refractory_total_volume_in3 == pytest.approx(
        sum(
            item.volume_in3
            for item in report.component_metrics
            if item.category == "refractory"
        )
    )
    assert report.refractory_total_volume_ft3 == pytest.approx(
        sum(
            item.volume_ft3
            for item in report.component_metrics
            if item.category == "refractory"
        )
    )


def test_three_d_scene_contains_meshes_for_layers_and_electrodes() -> None:
    scene = build_vessel_3d_scene(DEFAULT_VESSEL_DRAFTER_LAYOUT)
    labels = {mesh.label for mesh in scene.meshes}

    assert "glass_bath" in labels
    assert "steel_shell" in labels
    assert "electrode_1" in labels
    assert all(len(mesh.faces) > 0 for mesh in scene.meshes)
    assert all(len(mesh.vertices) > 0 for mesh in scene.meshes)


def test_three_d_scene_respects_visible_label_filter() -> None:
    scene = build_vessel_3d_scene(
        DEFAULT_VESSEL_DRAFTER_LAYOUT,
        visible_labels={"glass_bath", "steel_shell"},
    )

    assert {mesh.label for mesh in scene.meshes} == {"glass_bath", "steel_shell"}


def test_three_d_scene_stays_within_preview_face_budget() -> None:
    scene = build_vessel_3d_scene(DEFAULT_VESSEL_DRAFTER_LAYOUT)

    assert sum(len(mesh.faces) for mesh in scene.meshes) < 20_000


def test_three_d_scene_section_cut_reduces_preview_faces() -> None:
    full_scene = build_vessel_3d_scene(DEFAULT_VESSEL_DRAFTER_LAYOUT)
    split_scene = build_vessel_3d_scene(
        DEFAULT_VESSEL_DRAFTER_LAYOUT,
        view_options=Vessel3DViewOptions(
            split_enabled=True,
            split_angle_degrees=30.0,
        ),
    )

    assert {
        "glass_bath",
        "hot_face_refractory",
        "ifb",
        "duraboard",
        "steel_shell",
    }.issubset({mesh.label for mesh in split_scene.meshes})
    assert {
        "glass_bath",
        "hot_face_refractory",
        "ifb",
        "duraboard",
        "steel_shell",
    }.issubset({mesh.label for mesh in full_scene.meshes})
    assert sum(len(mesh.faces) for mesh in split_scene.meshes) < sum(
        len(mesh.faces) for mesh in full_scene.meshes
    )

    assert split_scene.bounds != full_scene.bounds


def test_three_d_scene_split_adds_material_faces_on_section_plane() -> None:
    split_scene = build_vessel_3d_scene(
        DEFAULT_VESSEL_DRAFTER_LAYOUT,
        view_options=Vessel3DViewOptions(
            split_enabled=True,
            split_angle_degrees=0.0,
        ),
    )
    hot_face_mesh = next(
        mesh for mesh in split_scene.meshes if mesh.label == "hot_face_refractory"
    )

    assert any(
        np.all(np.abs(polygon[:, 1]) < 1e-6) for polygon in hot_face_mesh.polygons
    )
