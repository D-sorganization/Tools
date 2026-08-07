"""Headless PyQt coverage for the canonical three-dimensional target workflow."""

from __future__ import annotations

import json

import numpy as np
import pytest

pytest.importorskip("PyQt6")
pytest.importorskip("pytestqt")

from rate_of_closure.ui.pyqt6.flight_explorer_tab import (  # noqa: E402
    FlightExplorerTab,
)
from rate_of_closure.ui.pyqt6.flight_view import FlightView  # noqa: E402
from rate_of_closure.ui.pyqt6.simulation_tab import SimulationTab  # noqa: E402
from rate_of_closure.ui.pyqt6.spatial_target_panel import (  # noqa: E402
    SpatialTargetPanel,
)
from rate_of_closure.ui.pyqt6.spatial_target_trajectory import (  # noqa: E402
    LandingSurfaceResolutionError,
    trajectory_target_miss,
)
from shared.python.swing_sim.solver import (  # noqa: E402
    BoxTolerance,
    SpatialTarget,
    SphereTolerance,
    SurfaceCircleTolerance,
    TargetPoint,
)

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


@pytest.fixture
def panel(qtbot):  # type: ignore[no-untyped-def]
    widget = SpatialTargetPanel()
    qtbot.addWidget(widget)
    return widget


def test_default_is_explicit_surface_target_with_accessible_context(panel) -> None:  # type: ignore[no-untyped-def]
    target = panel.target()
    assert target.kind == "landing_area"
    assert target.point.app_coordinates_m == pytest.approx((230.0, 0.0, 0.0))
    assert target.point.source_frame == "app"
    assert target.ground_source == "course.surface/default"
    assert "canonical app frame" in panel.summary_label().text()
    assert "x downrange, y up, z right" in panel.summary_label().text()
    assert "metres" in panel.summary_label().text()
    assert "surface circle radius 10.00 m" in panel.summary_label().text()
    assert panel.summary_label().accessibleName() == "Current Spatial Target"


def test_invalid_numeric_text_is_visible_and_does_not_emit(panel, qtbot) -> None:  # type: ignore[no-untyped-def]
    previous = panel.current_target()
    with qtbot.assertNotEmitted(panel.targetChanged):
        panel.coordinate_edit("x").setText("not-a-number")

    assert not panel.is_valid()
    assert panel.current_target() == previous
    assert "Downrange x must be a finite number" in panel.validation_label().text()
    assert panel.coordinate_edit("x").property("validationState") == "error"
    assert "invalid" in panel.coordinate_edit("x").accessibleDescription().lower()
    with pytest.raises(ValueError, match="Downrange x"):
        panel.target()
    assert not panel._copy_button.isEnabled()


def test_invalid_value_must_be_corrected_before_frame_change(panel) -> None:  # type: ignore[no-untyped-def]
    panel.coordinate_edit("x").setText("-")
    panel.frame_combo().setCurrentIndex(1)

    assert panel.frame_combo().currentData() == "app"
    assert panel.coordinate_edit("x").text() == "-"
    assert "Correct invalid entries" in panel.validation_label().text()


@pytest.mark.parametrize(
    ("edit", "value", "message"),
    [
        ("label", "", "label must be non-empty"),
        ("ground", "", "ground_source must be non-empty"),
        ("tolerance", "0", "greater than zero"),
    ],
)
def test_all_editor_boundaries_surface_contract_errors(
    panel, edit: str, value: str, message: str
) -> None:  # type: ignore[no-untyped-def]
    widget = {
        "label": panel._label_edit,
        "ground": panel._ground_edit,
        "tolerance": panel.tolerance_edit("primary"),
    }[edit]
    widget.setText(value)
    assert not panel.is_valid()
    assert message in panel.validation_label().text()


def test_negative_coordinates_and_frame_change_preserve_physical_point(
    panel, qtbot
) -> None:  # type: ignore[no-untyped-def]
    panel.coordinate_edit("third").setText("-7.5")
    canonical = panel.target().point.app_coordinates_m
    assert canonical == pytest.approx((230.0, 0.0, -7.5))

    with qtbot.waitSignal(panel.targetChanged, timeout=2000) as changed:
        panel.frame_combo().setCurrentIndex(1)

    target = changed.args[0]
    assert target.point.app_coordinates_m == pytest.approx(canonical)
    assert target.point.coordinates_in("flight") == pytest.approx((230.0, 7.5, 0.0))
    assert target.point.source_frame == "flight"
    assert panel.coordinate_label("second").text() == "Left y [m]"
    assert panel.coordinate_label("third").text() == "Up z [m]"
    assert "authored in flight frame" in panel.summary_label().text()


def test_aerial_box_builds_strict_canonical_target(panel) -> None:  # type: ignore[no-untyped-def]
    panel.kind_combo().setCurrentIndex(1)
    panel.tolerance_combo().setCurrentIndex(1)
    panel.coordinate_edit("x").setText("120")
    panel.coordinate_edit("second").setText("24")
    panel.coordinate_edit("third").setText("-3")
    panel.tolerance_edit("primary").setText("4")
    panel.tolerance_edit("secondary").setText("5")
    panel.tolerance_edit("tertiary").setText("6")

    target = panel.target()
    assert target.kind == "aerial_waypoint"
    assert target.point.app_coordinates_m == pytest.approx((120.0, 24.0, -3.0))
    assert isinstance(target.tolerance, BoxTolerance)
    assert target.tolerance.half_extents_m == pytest.approx((4.0, 5.0, 6.0))
    assert target.elevation_source == "absolute"
    assert target.ground_source is None


def test_json_round_trip_and_invalid_paste_are_explicit(panel) -> None:  # type: ignore[no-untyped-def]
    expected = SpatialTarget(
        label="Apex gate",
        kind="aerial_waypoint",
        point=TargetPoint.from_frame((137.5, 3.25, 24.25), "flight"),
        tolerance=BoxTolerance((2.0, 3.0, 4.0)),
        elevation_source="absolute",
    )
    panel.load_target_json(panel.serialize_target(expected))
    assert panel.target() == expected
    assert panel.target_json() == panel.serialize_target(expected)

    panel.load_target_json("{")
    assert not panel.is_valid()
    assert "Could not load target JSON" in panel.validation_label().text()
    assert panel.current_target() == expected


def test_flight_view_renders_target_in_all_projections(qtbot) -> None:  # type: ignore[no-untyped-def]
    view = FlightView()
    qtbot.addWidget(view)
    target = SpatialTarget(
        label="Apex gate",
        kind="aerial_waypoint",
        point=TargetPoint(120.0, 30.0, -8.0),
        tolerance=BoxTolerance((4.0, 5.0, 6.0)),
        elevation_source="absolute",
    )
    view.set_spatial_target(target)
    view.set_trajectory(
        np.array([[0.0, 0.0, 0.0], [100.0, 28.0, -6.0], [180.0, 0.0, 0.0]])
    )

    assert view.spatial_target() == target
    carry, height, lateral = view.extents_m()
    assert carry >= 124.0
    assert height >= 35.0
    assert lateral >= 14.0
    labels = [text.get_text() for axes in view._figure.axes for text in axes.texts]
    assert labels.count("Apex gate") == 3


def test_aerial_target_uses_closest_trajectory_passage_not_landing() -> None:
    target = SpatialTarget(
        label="Gate",
        kind="aerial_waypoint",
        point=TargetPoint(100.0, 20.0, 0.0),
        tolerance=BoxTolerance((2.0, 2.0, 2.0)),
        elevation_source="absolute",
    )
    miss = trajectory_target_miss(
        target,
        np.array([[0.0, 0.0, 0.0], [101.0, 21.0, 1.0], [200.0, 0.0, 0.0]]),
    )
    assert miss.accepted


def test_landing_target_projects_ball_center_to_flat_course_surface() -> None:
    """Ball radius or tee elevation cannot create a false vertical landing miss."""
    target = SpatialTarget(
        label="Default green",
        kind="landing_area",
        point=TargetPoint(230.0, 0.0, 4.0),
        tolerance=SurfaceCircleTolerance(0.01),
        elevation_source="course_surface",
        ground_source="course.surface/default",
    )
    positions = np.array(
        [
            [225.0, 4.0, 2.0],
            [230.0, 0.02135 + 0.0381, 4.0],
        ]
    )

    miss = trajectory_target_miss(target, positions)

    assert miss.accepted
    assert miss.distance_m == pytest.approx(0.0)
    assert miss.vector_m == pytest.approx((0.0, 0.0, 0.0))


@pytest.mark.parametrize(
    ("elevation_m", "ground_source", "code"),
    [
        (2.5, "course.surface/default", "UNSUPPORTED_SURFACE_ELEVATION"),
        (0.0, "course.surface/unknown", "UNRESOLVED_GROUND_SOURCE"),
    ],
)
def test_landing_target_fails_closed_without_terrain_resolution(
    elevation_m: float, ground_source: str, code: str
) -> None:
    target = SpatialTarget(
        label="Unresolved green",
        kind="landing_area",
        point=TargetPoint(230.0, elevation_m, 0.0),
        tolerance=SurfaceCircleTolerance(10.0),
        elevation_source="course_surface",
        ground_source=ground_source,
    )

    with pytest.raises(LandingSurfaceResolutionError) as caught:
        trajectory_target_miss(target, np.array([[230.0, 0.02135, 0.0]]))

    assert caught.value.code == code


def test_flight_view_refuses_to_render_unresolved_surface(qtbot) -> None:  # type: ignore[no-untyped-def]
    view = FlightView()
    qtbot.addWidget(view)
    unsupported = SpatialTarget(
        label="Floating green",
        kind="landing_area",
        point=TargetPoint(230.0, 2.0, 0.0),
        tolerance=SurfaceCircleTolerance(10.0),
        elevation_source="course_surface",
        ground_source="course.surface/default",
    )

    with pytest.raises(
        LandingSurfaceResolutionError, match="resolved flat course surface"
    ):
        view.set_spatial_target(unsupported)
    assert view.spatial_target() is None


@pytest.mark.parametrize(
    "tolerance",
    [SphereTolerance(1.0), BoxTolerance((1.0, 1.0, 1.0))],
)
def test_aerial_target_detects_between_sample_volume_crossing(tolerance) -> None:  # type: ignore[no-untyped-def]
    target = SpatialTarget(
        label="Narrow gate",
        kind="aerial_waypoint",
        point=TargetPoint(100.0, 20.0, 0.0),
        tolerance=tolerance,
        elevation_source="absolute",
    )
    positions = np.array([[95.0, 20.0, 0.0], [105.0, 20.0, 0.0]])
    assert not target.miss(positions[0]).accepted
    assert not target.miss(positions[1]).accepted

    miss = trajectory_target_miss(target, positions)

    assert miss.accepted
    assert miss.distance_m == 0.0


def test_explorer_target_updates_plot_and_reports_signed_miss(qtbot) -> None:  # type: ignore[no-untyped-def]
    explorer = FlightExplorerTab()
    qtbot.addWidget(explorer)
    panel = explorer._spatial_target_panel
    panel.coordinate_edit("x").setText("245")
    exploration = explorer.run_now()

    assert exploration is not None
    assert explorer.flight_view().spatial_target() == panel.target()
    assert explorer.flight_view().course_layout().green_distance_m == pytest.approx(
        245.0
    )
    assert "Landing miss:" in panel.miss_label().text()
    assert "long" in panel.miss_label().text() or "short" in panel.miss_label().text()


def test_invalid_target_draft_cannot_crash_post_run_refresh(qtbot) -> None:  # type: ignore[no-untyped-def]
    explorer = FlightExplorerTab()
    qtbot.addWidget(explorer)
    panel = explorer._spatial_target_panel
    plotted = explorer.flight_view().spatial_target()
    panel.coordinate_edit("x").setText("-")

    assert explorer.run_now() is not None
    assert explorer.flight_view().spatial_target() == plotted
    assert "finite number" in panel.validation_label().text()
    assert "unavailable" in panel.miss_label().text().lower()


@pytest.mark.parametrize(
    ("edit", "value", "diagnostic"),
    [
        ("elevation", "1.5", "flat course surface"),
        ("ground", "course.surface/unknown", "not resolved"),
    ],
)
def test_panel_rejects_unresolved_landing_surfaces_without_moving_plot(
    panel, edit: str, value: str, diagnostic: str
) -> None:  # type: ignore[no-untyped-def]
    previous = panel.current_target()
    widget = (
        panel.coordinate_edit("second") if edit == "elevation" else panel._ground_edit
    )
    widget.setText(value)

    assert not panel.is_valid()
    assert panel.current_target() == previous
    assert diagnostic in panel.validation_label().text()


def test_integrated_session_uses_same_target_editor_and_plot(qtbot) -> None:  # type: ignore[no-untyped-def]
    tab = SimulationTab()
    qtbot.addWidget(tab)
    panel = tab._spatial_target_panel
    panel.kind_combo().setCurrentIndex(1)
    panel.coordinate_edit("x").setText("90")
    panel.coordinate_edit("second").setText("18")

    assert tab.flight_view().spatial_target() == panel.target()
    assert tab.flight_view().spatial_target().point.elevation_m == pytest.approx(18.0)
    assert not tab.solver_panel().target_panel().isEnabled()
    assert "landing optimizer" in tab.solver_panel().target_panel().toolTip()
    tab.stop()


def test_integrated_landing_target_stays_synced_with_legacy_solver(qtbot) -> None:  # type: ignore[no-untyped-def]
    tab = SimulationTab()
    qtbot.addWidget(tab)
    panel = tab._spatial_target_panel
    panel.coordinate_edit("x").setText("175")
    panel.coordinate_edit("third").setText("-4")

    region = tab.solver_panel().target_panel().region()
    assert region.distance_m == pytest.approx(175.0)
    assert region.lateral_m == pytest.approx(-4.0)
    assert tab.view().course_layout().green_distance_m == pytest.approx(175.0)
    tab.stop()


def test_simulation_invalid_target_draft_runs_without_refresh_crash(qtbot) -> None:  # type: ignore[no-untyped-def]
    tab = SimulationTab()
    qtbot.addWidget(tab)
    panel = tab._spatial_target_panel
    plotted = tab.flight_view().spatial_target()
    panel.coordinate_edit("x").setText("-")

    assert tab.run_now() is not None
    assert tab.flight_view().spatial_target() == plotted
    assert "finite number" in panel.validation_label().text()
    assert "unavailable" in panel.miss_label().text().lower()
    tab.stop()


def test_inspector_round_trips_target_and_setup_atomically(qtbot) -> None:  # type: ignore[no-untyped-def]
    tab = SimulationTab()
    qtbot.addWidget(tab)
    panel = tab._spatial_target_panel
    panel.kind_combo().setCurrentIndex(1)
    panel.coordinate_edit("x").setText("140")
    panel.coordinate_edit("second").setText("24")
    assert tab.run_now() is not None

    document = tab.inspector().run_document()
    assert document["spatial_target"] == json.loads(panel.target_json())
    assert document["solver_manifest"]["target"] == document["spatial_target"]

    replacement = SpatialTarget(
        label="Replacement gate",
        kind="aerial_waypoint",
        point=TargetPoint(110.0, 18.0, -2.0),
        tolerance=SphereTolerance(3.0),
        elevation_source="absolute",
    )
    imported = {
        "format": "rate_of_closure.simulation_run/2",
        "spatial_target": json.loads(panel.serialize_target(replacement)),
        "parameters": {"ball_setup": {"support_mode": "ground", "tee_height_m": 0.0}},
    }
    tab.inspector().load_settings_document(imported)
    assert panel.target() == replacement
    assert tab._ball_setup_control.setup().support_mode.value == "ground"

    previous_target = panel.target()
    previous_setup = tab._ball_setup_control.setup()
    corrupt = {
        **imported,
        "spatial_target": {**imported["spatial_target"], "units": "yd"},
        "parameters": {"ball_setup": {"support_mode": "tee", "tee_height_m": 0.04}},
    }
    with pytest.raises(ValueError, match="units"):
        tab.inspector().load_settings_document(corrupt)
    assert panel.target() == previous_target
    assert tab._ball_setup_control.setup() == previous_setup
    tab.stop()
