from __future__ import annotations

from time import process_time
from types import SimpleNamespace

import pytest
from PyQt6.QtCore import Qt
from PyQt6.QtTest import QTest

from rate_of_closure.club import (
    CLUB_LIBRARY,
    head_cog,
    hosel_point,
    parametric_head_mesh,
)
from rate_of_closure.club_camera import DEFAULT_CLUB_CAMERA, ClubCameraAction
from rate_of_closure.model import ImpactScenario
from rate_of_closure.ui.pyqt6.club_view import Club3DView


def test_keyboard_camera_status_focus_and_reset_button(qtbot) -> None:  # type: ignore[no-untyped-def]
    view = Club3DView()
    qtbot.addWidget(view)
    view.show()
    view.activateWindow()
    qtbot.waitExposed(view)
    view.set_scenario(ImpactScenario(clubhead_speed_mph=100.0))
    view._canvas.setFocus()
    qtbot.waitUntil(view._canvas.hasFocus)
    QTest.keyClick(view._canvas, Qt.Key.Key_Left)
    assert view._camera.azimuth_deg == 145.0
    assert "azimuth 145°" in view._status.text()
    assert view._canvas.hasFocus()
    QTest.keyClick(view._canvas, Qt.Key.Key_Home)
    assert view._camera == DEFAULT_CLUB_CAMERA
    assert view._canvas.hasFocus()
    QTest.mouseClick(view._reset_view_button, Qt.MouseButton.LeftButton)
    assert view._reset_view_button.hasFocus()


def test_generated_adoption_is_atomic_across_partial_draw_failure(
    qtbot,
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    from rate_of_closure.ui.pyqt6 import club_view_render

    view = Club3DView()
    qtbot.addWidget(view)
    view.set_scenario(ImpactScenario(clubhead_speed_mph=100.0))
    prior_source = view._source
    original = club_view_render.draw_club_view
    calls = 0

    def fail_after_draw(candidate: Club3DView) -> None:
        nonlocal calls
        calls += 1
        original(candidate)
        if calls == 1:
            raise OSError("planted renderer failure")

    monkeypatch.setattr(club_view_render, "draw_club_view", fail_after_draw)
    spec = CLUB_LIBRARY["Mallet Putter"]
    with pytest.raises(OSError, match="planted"):
        view.set_head_mesh(
            parametric_head_mesh(spec),
            hosel_point=hosel_point(spec),
            cog_point=head_cog(spec).cog,
            label=spec.name,
        )
    assert view._source is prior_source
    assert not view.has_mesh()


def test_worst_library_mesh_uses_bounded_playback_cadence(qtbot) -> None:  # type: ignore[no-untyped-def]
    view = Club3DView()
    qtbot.addWidget(view)
    view.resize(900, 650)
    view.set_scenario(ImpactScenario(clubhead_speed_mph=100.0))
    spec = CLUB_LIBRARY["Mallet Putter"]
    view.set_head_mesh(
        parametric_head_mesh(spec),
        hosel_point=hosel_point(spec),
        cog_point=head_cog(spec).cog,
        label=spec.name,
    )
    # This is a CPU-work budget, not a hosted-runner scheduling benchmark.
    # Wall time makes an otherwise deterministic render fail when sibling
    # xdist workers temporarily deschedule this process.
    started = process_time()
    view._draw()
    elapsed = process_time() - started
    assert view._timer.interval() == 200
    assert elapsed < 0.5


def test_playback_cycle_duration_is_independent_of_render_cadence(qtbot) -> None:  # type: ignore[no-untyped-def]
    view = Club3DView()
    qtbot.addWidget(view)
    view.set_scenario(ImpactScenario(clubhead_speed_mph=100.0))
    view._draw = lambda: None  # type: ignore[method-assign]
    for _ in range(10):
        view._advance()
    assert view._phase == pytest.approx((10 * 200 / 1920) % 1.0)


def test_native_orbit_pauses_playback_and_redraws_the_canonical_release(qtbot) -> None:  # type: ignore[no-untyped-def]
    view = Club3DView()
    qtbot.addWidget(view)
    view.set_scenario(ImpactScenario(clubhead_speed_mph=100.0))
    view._play_button.setChecked(True)
    view._on_orbit_started(SimpleNamespace(button=1))
    assert not view._timer.isActive()
    view._axes.azim = 725.0
    view._axes.elev = 95.0
    view._on_orbit_finished(SimpleNamespace(button=1))
    assert view._camera.elevation_deg == 80.0
    assert view._axes.elev == 80.0
    assert view._timer.isActive()


def test_scenario_ui_failure_restores_model_and_prior_image(
    qtbot,
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    from rate_of_closure.ui.pyqt6 import club_view_render

    view = Club3DView()
    qtbot.addWidget(view)
    prior = ImpactScenario(clubhead_speed_mph=100.0)
    view.set_scenario(prior)
    original = club_view_render.draw_club_view
    calls = 0

    def fail_after_clear(candidate: Club3DView) -> None:
        nonlocal calls
        calls += 1
        original(candidate)
        if calls == 1:
            candidate._axes.clear()
            raise OSError("planted post-clear failure")

    monkeypatch.setattr(club_view_render, "draw_club_view", fail_after_clear)
    assert not view.try_set_scenario(ImpactScenario(clubhead_speed_mph=110.0))
    assert view._scenario is prior
    assert len(view._axes.lines) > 0
    assert not view._error.isHidden()
    assert "image may be stale" in view._error.text()


def test_camera_rollback_failure_preserves_original_and_reports_stale_image(
    qtbot,
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    view = Club3DView()
    qtbot.addWidget(view)
    view.set_scenario(ImpactScenario(clubhead_speed_mph=100.0))
    prior = view._camera
    calls = 0

    def fail_candidate_and_rollback() -> None:
        nonlocal calls
        calls += 1
        raise OSError("candidate failure" if calls == 1 else "rollback failure")

    monkeypatch.setattr(view, "_draw", fail_candidate_and_rollback)
    view._try_camera_action(ClubCameraAction.LEFT)
    assert view._camera == prior
    assert "candidate failure" in view._error.text()
    assert "rollback failure" not in view._error.text()
    assert "image may be stale" in view._error.text()


def test_scenario_is_solved_once_and_presentation_redraws_reuse_result(
    qtbot,
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    from rate_of_closure.ui.pyqt6 import club_view

    view = Club3DView()
    qtbot.addWidget(view)
    actual_solve = club_view.solve
    calls = 0

    def counted_solve(scenario: ImpactScenario):  # type: ignore[no-untyped-def]
        nonlocal calls
        calls += 1
        return actual_solve(scenario)

    monkeypatch.setattr(club_view, "solve", counted_solve)
    view.set_scenario(ImpactScenario(clubhead_speed_mph=100.0))
    view._try_camera_action(ClubCameraAction.LEFT)
    view._try_redraw()
    view._advance()
    assert calls == 1
    view.set_scenario(ImpactScenario(clubhead_speed_mph=101.0))
    assert calls == 2


def test_successful_scenario_recovers_a_prior_render_alert(qtbot) -> None:  # type: ignore[no-untyped-def]
    view = Club3DView()
    qtbot.addWidget(view)
    view._set_error("Clubhead render failed: planted", "render")
    assert view.try_set_scenario(ImpactScenario(clubhead_speed_mph=100.0))
    assert view._error.isHidden()
    assert view._error_kind is None


def test_procedural_reference_marker_does_not_claim_a_mass_centroid(qtbot) -> None:  # type: ignore[no-untyped-def]
    view = Club3DView()
    qtbot.addWidget(view)
    view.set_scenario(ImpactScenario(clubhead_speed_mph=100.0))
    labels = view._axes.get_legend_handles_labels()[1]
    assert "scenario reference" in labels
    assert "volumetric CG" not in labels
