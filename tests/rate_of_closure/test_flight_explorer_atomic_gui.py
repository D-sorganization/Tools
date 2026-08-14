"""First-failure and no-recompute checks for the PyQt flight explorer."""

from __future__ import annotations

import numpy as np
from PyQt6.QtCore import Qt

import rate_of_closure.ui.pyqt6.flight_explorer_run as run_module
from rate_of_closure.ui.pyqt6.flight_explorer_tab import FlightExplorerTab


def test_first_execution_failure_is_honest_empty_and_bounded(
    qtbot, monkeypatch
) -> None:  # type: ignore[no-untyped-def]
    def fail_execution(*_args: object) -> None:
        raise OSError("\x00" + "planted first execution failure " * 30)

    monkeypatch.setattr(run_module, "explore_with_optional_wind", fail_execution)
    tab = FlightExplorerTab()
    qtbot.addWidget(tab)
    tab.show()
    assert tab.run_now() is None
    assert tab.accepted_study() is None
    assert len(tab.flight_view().trajectory()) == 0
    assert not tab._flight_panel.controls.play_button.isEnabled()
    assert not tab._flight_panel.controls.landing_button.isEnabled()
    assert "No accepted flight is available" in tab._error_status.text()
    assert "remains displayed" not in tab._error_status.text()
    assert "\x00" not in tab._error_status.text()
    assert len(tab._error_status.text()) <= 240


def test_selection_playback_units_and_display_do_not_rerun_solver(
    qtbot, monkeypatch
) -> None:  # type: ignore[no-untyped-def]
    tab = FlightExplorerTab()
    qtbot.addWidget(tab)
    tab.show()
    assert tab.run_now() is not None
    calls = 0
    original = run_module.explore_with_optional_wind

    def count_execution(*args: object):  # type: ignore[no-untyped-def]
        nonlocal calls
        calls += 1
        return original(*args)

    monkeypatch.setattr(run_module, "explore_with_optional_wind", count_execution)
    canvas = tab.flight_view()._canvas
    canvas.setFocus()
    qtbot.keyClick(canvas, Qt.Key.Key_Home)
    tab._flight_panel.controls.jump_to_landing()
    tab._speed_unit_combo.setCurrentText("m/s")
    tab.flight_view().display_check("top").setChecked(False)
    tab.flight_view().display_check("top").setChecked(True)
    assert calls == 0


def test_post_render_ui_failure_restores_complete_prior_publication(
    qtbot, monkeypatch
) -> None:  # type: ignore[no-untyped-def]
    tab = FlightExplorerTab()
    qtbot.addWidget(tab)
    tab.show()
    assert tab.run_now() is not None
    accepted = tab.accepted_study()
    assert accepted is not None
    view = tab.flight_view()
    canvas = view._canvas
    canvas.setFocus()
    qtbot.keyClick(canvas, Qt.Key.Key_End)
    prior_view = view.trajectory().copy()
    prior_target = tab._target_workflow._positions.copy()
    prior_selection = view.selected_raw_index()
    prior_time = tab._flight_panel.controls.current_time_s()
    prior_rows = {key: row.value_label.text() for key, row in tab._rows.items()}
    prior_deltas = tab.wind_controls.delta_texts()
    prior_sample = tab._sample_status.text()

    tab._direct_spins["launch_angle_deg"].setValue(
        tab._direct_spins["launch_angle_deg"].value() + 1.0
    )
    prior_context = tab._context_status.text()

    def fail_comparison(_comparison: object) -> None:
        raise RuntimeError("planted post-render comparison failure")

    monkeypatch.setattr(tab.wind_controls, "set_comparison", fail_comparison)
    assert tab.run_now() is None
    assert tab.accepted_study() is accepted
    assert tab._generation == accepted.generation
    assert tab.last_exploration() is accepted.exploration
    assert tab.wind_comparison is accepted.comparison
    assert np.array_equal(view.trajectory(), prior_view)
    assert np.array_equal(tab._target_workflow._positions, prior_target)
    assert view.selected_raw_index() == prior_selection
    assert tab._flight_panel.controls.current_time_s() == prior_time
    assert {key: row.value_label.text() for key, row in tab._rows.items()} == prior_rows
    assert tab.wind_controls.delta_texts() == prior_deltas
    assert tab._context_status.text() == prior_context
    assert tab._sample_status.text() == prior_sample
    assert "planted post-render comparison failure" in tab._error_status.text()
    assert "prior accepted flight remains displayed" in tab._error_status.text().lower()


def test_nested_view_rollback_forces_prior_authority_when_pixels_are_stale(
    qtbot, monkeypatch
) -> None:  # type: ignore[no-untyped-def]
    tab = FlightExplorerTab()
    qtbot.addWidget(tab)
    tab.show()
    assert tab.run_now() is not None
    accepted = tab.accepted_study()
    assert accepted is not None
    view = tab.flight_view()
    canvas = view._canvas
    canvas.setFocus()
    qtbot.keyClick(canvas, Qt.Key.Key_End)
    prior_selection = view.selected_raw_index()
    prior_time = tab._flight_panel.controls.current_time_s()
    tab._direct_spins["launch_angle_deg"].setValue(
        tab._direct_spins["launch_angle_deg"].value() + 1.0
    )
    original_draw = canvas.draw
    draw_calls = 0

    def staged_draw() -> None:
        nonlocal draw_calls
        draw_calls += 1
        original_draw()
        if draw_calls == 2:
            raise RuntimeError("planted prior repaint failure")

    def fail_comparison(_comparison: object) -> None:
        raise RuntimeError("planted post-render comparison failure")

    monkeypatch.setattr(canvas, "draw", staged_draw)
    monkeypatch.setattr(tab.wind_controls, "set_comparison", fail_comparison)
    assert tab.run_now() is None
    assert draw_calls == 3
    assert tab.accepted_study() is accepted
    assert np.array_equal(view.trajectory(), accepted.exploration.positions)
    assert view._sample_plan is accepted.plan
    assert view.selected_raw_index() == prior_selection
    assert tab._flight_panel.controls.current_time_s() == prior_time
    assert not canvas.has_pending_draw()
    assert "plot restoration failed" in tab._error_status.text().lower()
    assert "image may be stale" in tab._error_status.text().lower()


def test_post_render_failure_force_restores_target_without_recomputation(
    qtbot, monkeypatch
) -> None:  # type: ignore[no-untyped-def]
    tab = FlightExplorerTab()
    qtbot.addWidget(tab)
    tab.show()
    assert tab.run_now() is not None
    accepted = tab.accepted_study()
    assert accepted is not None
    prior_target = tab._target_workflow._positions.copy()
    prior_miss = tab._target_workflow._panel.miss_label().text()
    tab._direct_spins["launch_angle_deg"].setValue(
        tab._direct_spins["launch_angle_deg"].value() + 1.0
    )
    original_refresh = tab._target_workflow._refresh_miss

    def fail_if_prior_is_recomputed() -> None:
        if np.array_equal(tab._target_workflow._positions, prior_target):
            raise RuntimeError("planted prior target refresh failure")
        original_refresh()

    def fail_comparison(_comparison: object) -> None:
        raise RuntimeError("planted post-render comparison failure")

    monkeypatch.setattr(
        tab._target_workflow, "_refresh_miss", fail_if_prior_is_recomputed
    )
    monkeypatch.setattr(tab.wind_controls, "set_comparison", fail_comparison)
    assert tab.run_now() is None
    assert tab.accepted_study() is accepted
    assert np.array_equal(tab._target_workflow._positions, prior_target)
    assert tab._target_workflow._panel.miss_label().text() == prior_miss
