"""Headless PyQt camera interactions for swing/impact and flight viewports."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("PyQt6")
pytest.importorskip("pytestqt")

from rate_of_closure.application.camera_commands import (  # noqa: E402
    CameraCommandId,
    FaceOnSide,
)
from rate_of_closure.club import get_club  # noqa: E402
from rate_of_closure.model import ImpactScenario  # noqa: E402
from rate_of_closure.simulation import SimulationConfig, run_simulation  # noqa: E402
from rate_of_closure.ui.pyqt6.flight_view import FlightView  # noqa: E402
from rate_of_closure.ui.pyqt6.simulation_view import SimulationView  # noqa: E402

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


@pytest.fixture(scope="module")
def reference_run():  # type: ignore[no-untyped-def]
    return run_simulation(
        SimulationConfig(
            scenario=ImpactScenario(clubhead_speed_mph=105.0),
            club=get_club("Driver 10.5°"),
        )
    )


@pytest.mark.parametrize("view_type", [SimulationView, FlightView])
def test_every_3d_view_exposes_accessible_stable_camera_commands(
    qtbot, view_type
) -> None:  # type: ignore[no-untyped-def]
    view = view_type()
    qtbot.addWidget(view)
    controls = view.camera_controls()
    expected = {
        CameraCommandId.VIEW_ISOMETRIC.value,
        CameraCommandId.VIEW_FACE_ON.value,
        CameraCommandId.VIEW_DOWN_THE_LINE.value,
        CameraCommandId.VIEW_OVERHEAD.value,
        CameraCommandId.AUTO_FIT.value,
        CameraCommandId.RECENTER.value,
        CameraCommandId.TRACK_SUBJECT.value,
    }
    assert set(controls.command_widgets()) == expected
    for command_id, widget in controls.command_widgets().items():
        assert widget.property("cameraCommandId") == command_id
        assert widget.toolTip()
        assert widget.focusPolicy().value != 0
    assert controls.minimumSizeHint().width() <= 700


def test_simulation_camera_starts_fitted_and_tracks_the_clubhead(
    qtbot, reference_run
) -> None:  # type: ignore[no-untyped-def]
    """The share-ready swing scene should not open tiny or lose its subject."""
    view = SimulationView()
    qtbot.addWidget(view)
    view.set_run(reference_run)

    state = view.camera_state()
    assert state.tracking_enabled
    assert state.auto_fit_enabled
    assert state.zoom == pytest.approx(2.0)
    assert view.camera_controls().track.isChecked()
    assert view.camera_controls().auto_fit.isChecked()

    subject = view._display(reference_run.swing_positions[0])
    limits = (
        view._axes.get_xlim3d(),
        view._axes.get_ylim3d(),
        view._axes.get_zlim3d(),
    )
    for coordinate, bounds in zip(subject, limits, strict=True):
        assert bounds[0] < coordinate < bounds[1]
    assert view.scene_extent_m() < view._camera_base_half_extent_m()


def test_flight_camera_starts_fitted_and_tracks_the_ball(qtbot, reference_run) -> None:  # type: ignore[no-untyped-def]
    view = FlightView()
    qtbot.addWidget(view)
    view.set_run(reference_run)

    state = view.camera_state()
    assert state.tracking_enabled
    assert state.auto_fit_enabled
    assert state.zoom == pytest.approx(2.0)
    assert view.camera_controls().track.isChecked()
    assert view.camera_controls().auto_fit.isChecked()
    assert view.camera_subject_in_frame()


def test_simulation_snap_views_are_exact_idempotent_and_face_side_explicit(
    qtbot, reference_run
) -> None:  # type: ignore[no-untyped-def]
    view = SimulationView()
    qtbot.addWidget(view)
    view.set_run(reference_run)

    view.apply_camera_command(CameraCommandId.VIEW_DOWN_THE_LINE)
    assert float(view._axes.elev) == pytest.approx(0.0)
    assert float(view._axes.azim) == pytest.approx(-90.0)
    first_limits = (
        view._axes.get_xlim3d(),
        view._axes.get_ylim3d(),
        view._axes.get_zlim3d(),
    )
    view.apply_camera_command(CameraCommandId.VIEW_DOWN_THE_LINE)
    assert (
        view._axes.get_xlim3d(),
        view._axes.get_ylim3d(),
        view._axes.get_zlim3d(),
    ) == first_limits

    view.set_face_on_side(FaceOnSide.LEFT)
    view.apply_camera_command(CameraCommandId.VIEW_FACE_ON)
    assert float(view._axes.elev) == pytest.approx(0.0)
    assert abs(float(view._axes.azim)) == pytest.approx(180.0)
    view.apply_camera_command(CameraCommandId.VIEW_OVERHEAD)
    assert float(view._axes.elev) == pytest.approx(90.0)
    assert float(view._axes.azim) == pytest.approx(-90.0)


def test_tracking_keeps_complete_swing_subject_inside_preserved_zoom(
    qtbot, reference_run
) -> None:  # type: ignore[no-untyped-def]
    view = SimulationView()
    qtbot.addWidget(view)
    view.set_run(reference_run)
    view.set_camera_zoom(2.0)
    view.set_camera_tracking(True)

    for time_s, subject in zip(
        reference_run.swing_times, reference_run.swing_positions, strict=True
    ):
        view.set_playback_time(float(time_s))
        display_subject = view._display(subject)
        limits = (
            view._axes.get_xlim3d(),
            view._axes.get_ylim3d(),
            view._axes.get_zlim3d(),
        )
        for coordinate, bounds in zip(display_subject, limits, strict=True):
            assert bounds[0] < coordinate < bounds[1]
        assert view.camera_zoom() == pytest.approx(2.0)

    view.suspend_camera_tracking()
    frozen_target = np.asarray(view.camera_state().target_m)
    view.set_playback_time(0.0)
    np.testing.assert_allclose(view.camera_state().target_m, frozen_target)
    view.recenter_camera()
    assert not view.camera_state().tracking_suspended


def test_flight_tracking_follows_ball_without_rebuilding_camera_controls(
    qtbot, reference_run
) -> None:  # type: ignore[no-untyped-def]
    view = FlightView()
    qtbot.addWidget(view)
    view.set_run(reference_run)
    view.set_camera_zoom(1.7)
    view.set_camera_tracking(True)
    for time_s in np.linspace(0.0, view.playback_duration_s(), 9):
        view.set_playback_time(float(time_s))
        assert view.camera_zoom() == pytest.approx(1.7)
        assert view.camera_subject_in_frame()
