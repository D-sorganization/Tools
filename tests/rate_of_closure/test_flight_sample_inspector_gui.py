"""Rendered interaction and transaction tests for the PyQt flight inspector."""

from __future__ import annotations

import numpy as np
import pytest
from matplotlib.backend_bases import MouseButton, MouseEvent
from PyQt6.QtCore import Qt

from rate_of_closure.ui.pyqt6.flight_explorer_tab import FlightExplorerTab
from rate_of_closure.units import SPEED_UNITS
from shared.python.swing_sim.flight import LaunchDirectionConvention


@pytest.fixture
def accepted_tab(qtbot):  # type: ignore[no-untyped-def]
    tab = FlightExplorerTab()
    qtbot.addWidget(tab)
    tab.resize(1200, 760)
    tab.show()
    assert tab.run_now() is not None
    return tab


def test_keyboard_selection_drives_the_single_playback_owner(
    accepted_tab, qtbot
) -> None:  # type: ignore[no-untyped-def]
    tab = accepted_tab
    view = tab.flight_view()
    canvas = view._canvas
    controls = tab._flight_panel.controls
    accepted = tab.accepted_study()
    assert accepted is not None
    assert canvas.accessibleName() == "Flight trajectory sample inspector"
    assert "Home and End" in canvas.accessibleDescription()

    canvas.setFocus()
    qtbot.keyClick(canvas, Qt.Key.Key_Home)
    assert view.selected_raw_index() == 0
    assert controls.current_time_s() == pytest.approx(
        accepted.plan.raw_sample(0).time_s
    )
    assert "source sample 1/" in tab._sample_status.text()

    controls.jump_to_landing()
    assert controls.current_time_s() == pytest.approx(controls._duration_s)
    qtbot.keyClick(canvas, Qt.Key.Key_Home)
    assert controls.current_time_s() == pytest.approx(0.0)
    assert canvas.hasFocus()


@pytest.mark.parametrize("dpi_scale", [1.0, 1.5])
def test_pointer_selects_primary_but_ignores_distinct_calm_and_miss(
    qtbot, dpi_scale: float
) -> None:  # type: ignore[no-untyped-def]
    tab = FlightExplorerTab()
    qtbot.addWidget(tab)
    tab.resize(1200, 760)
    tab.show()
    tab.wind_controls.enabled_check.setChecked(True)
    tab.wind_controls.speed_spin.setValue(100.0)
    tab.wind_controls.bearing_spin.setValue(90.0)
    assert tab.run_now() is not None
    accepted = tab.accepted_study()
    assert accepted is not None and accepted.calm_comparison is not None
    view = tab.flight_view()
    canvas = view._canvas
    view._figure.set_dpi(100 * dpi_scale)
    view._draw(sync=True)
    assert not canvas.has_pending_draw()
    tab.activateWindow()
    canvas.setFocus()
    qtbot.waitUntil(canvas.hasFocus)
    axes = view._inspector_axes["top"]
    current_pixels = np.array(
        [
            axes.transData.transform((sample.downrange_m, sample.right_m))
            for sample in accepted.plan.samples
        ]
    )
    primary_index = 0
    primary_x, primary_y = current_pixels[primary_index]
    canvas.callbacks.process(
        "button_press_event",
        MouseEvent(
            "button_press_event",
            canvas,
            primary_x,
            primary_y,
            button=MouseButton.LEFT,
        ),
    )
    assert view.selected_raw_index() == primary_index
    assert canvas.hasFocus()
    selected_time = tab._flight_panel.controls.current_time_s()

    calm_pixels = np.array(
        [
            axes.transData.transform((position[0], position[2]))
            for position in accepted.calm_comparison.positions
        ]
    )
    calm_distances = np.min(
        np.linalg.norm(calm_pixels[:, None, :] - current_pixels[None, :, :], axis=2),
        axis=1,
    )
    calm_x, calm_y = calm_pixels[int(np.argmax(calm_distances))]
    assert float(np.max(calm_distances)) > 12.0
    canvas.callbacks.process(
        "button_press_event",
        MouseEvent(
            "button_press_event",
            canvas,
            calm_x,
            calm_y,
            button=MouseButton.LEFT,
        ),
    )
    assert view.selected_raw_index() == primary_index
    assert tab._flight_panel.controls.current_time_s() == pytest.approx(selected_time)

    corners = np.array(
        [
            [axes.bbox.xmin + 2, axes.bbox.ymin + 2],
            [axes.bbox.xmax - 2, axes.bbox.ymax - 2],
        ]
    )
    miss_distances = np.min(
        np.linalg.norm(corners[:, None, :] - current_pixels[None, :, :], axis=2),
        axis=1,
    )
    miss_x, miss_y = corners[int(np.argmax(miss_distances))]
    assert float(np.max(miss_distances)) > 12.0
    canvas.callbacks.process(
        "button_press_event",
        MouseEvent(
            "button_press_event",
            canvas,
            miss_x,
            miss_y,
            button=MouseButton.LEFT,
        ),
    )
    assert view.selected_raw_index() == primary_index
    assert canvas.hasFocus()


def test_target_post_assignment_failure_retains_every_accepted_owner(
    accepted_tab, monkeypatch
) -> None:  # type: ignore[no-untyped-def]
    tab = accepted_tab
    accepted = tab.accepted_study()
    assert accepted is not None
    old_view = tab.flight_view().trajectory().copy()
    old_target = tab._target_workflow._positions.copy()
    old_miss = tab._spatial_target_panel.miss_label().text()
    tab._flight_panel.controls.jump_to_landing()
    old_time = tab._flight_panel.controls.current_time_s()
    tab._direct_spins["launch_angle_deg"].setValue(12.0)

    def fail_after_assignment() -> None:
        raise RuntimeError("planted target residual failure")

    monkeypatch.setattr(tab._target_workflow, "_refresh_miss", fail_after_assignment)
    assert tab.run_now() is None
    assert tab.accepted_study() is accepted
    assert np.array_equal(tab.flight_view().trajectory(), old_view)
    assert np.array_equal(tab._target_workflow._positions, old_target)
    assert tab._spatial_target_panel.miss_label().text() == old_miss
    assert tab._flight_panel.controls.current_time_s() == pytest.approx(old_time)
    assert "prior accepted flight remains displayed" in tab._error_status.text().lower()


def test_view_render_failure_rolls_back_target_playback_and_authority(
    accepted_tab, monkeypatch
) -> None:  # type: ignore[no-untyped-def]
    tab = accepted_tab
    view = tab.flight_view()
    accepted = tab.accepted_study()
    assert accepted is not None
    old_view = view.trajectory().copy()
    old_target = tab._target_workflow._positions.copy()
    old_miss = tab._spatial_target_panel.miss_label().text()
    tab._flight_panel.controls.jump_to_landing()
    old_time = tab._flight_panel.controls.current_time_s()
    tab._direct_spins["launch_angle_deg"].setValue(12.0)
    original_draw = view._draw
    calls = 0

    def fail_after_drawing_candidate(*, sync: bool = False) -> None:
        nonlocal calls
        calls += 1
        original_draw(sync=sync)
        if calls == 1:
            raise RuntimeError("planted view publication failure")

    monkeypatch.setattr(view, "_draw", fail_after_drawing_candidate)
    assert tab.run_now() is None
    assert tab.accepted_study() is accepted
    assert np.array_equal(view.trajectory(), old_view)
    assert np.array_equal(tab._target_workflow._positions, old_target)
    assert tab._spatial_target_panel.miss_label().text() == old_miss
    assert tab._flight_panel.controls.current_time_s() == pytest.approx(old_time)


def test_double_view_failure_reports_unrestored_pixels_honestly(
    accepted_tab, qtbot, monkeypatch
) -> None:  # type: ignore[no-untyped-def]
    tab = accepted_tab
    view = tab.flight_view()
    accepted = tab.accepted_study()
    assert accepted is not None
    original_canvas_draw = view._canvas.draw
    calls = 0

    def publish_then_fail_and_block_rollback() -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            original_canvas_draw()
            raise RuntimeError("candidate pixels published")
        raise RuntimeError("prior pixels could not be restored")

    tab._direct_spins["launch_angle_deg"].setValue(12.0)
    monkeypatch.setattr(view._canvas, "draw", publish_then_fail_and_block_rollback)
    assert tab.run_now() is None
    assert tab.accepted_study() is accepted
    assert calls == 2
    assert "prior accepted authority is retained" in tab._error_status.text().lower()
    assert "image may be stale or unavailable" in tab._error_status.text().lower()
    assert "remains displayed" not in tab._error_status.text().lower()
    assert not view._canvas.has_pending_draw()
    monkeypatch.undo()
    assert tab.run_now() is not None
    assert tab.accepted_study() is not accepted
    assert tab._error_status.text() == ""
    qtbot.waitUntil(lambda: not view._canvas.has_pending_draw())
    view._canvas.draw_idle()
    assert view._canvas.has_pending_draw()
    view._canvas.cancel_pending_draw()
    assert not view._canvas.has_pending_draw()


def test_selection_render_failure_recovers_without_erasing_scientific_errors(
    accepted_tab, qtbot, monkeypatch
) -> None:  # type: ignore[no-untyped-def]
    tab = accepted_tab
    view = tab.flight_view()
    canvas = view._canvas
    controls = tab._flight_panel.controls
    canvas.setFocus()
    qtbot.keyClick(canvas, Qt.Key.Key_Home)
    original_draw = view._draw
    calls = 0

    def fail_once_after_drawing(*, sync: bool = False) -> None:
        nonlocal calls
        calls += 1
        original_draw(sync=sync)
        if calls == 1:
            raise RuntimeError("planted selection render failure")

    monkeypatch.setattr(view, "_draw", fail_once_after_drawing)
    qtbot.keyClick(canvas, Qt.Key.Key_End)
    assert view.selected_raw_index() == 0
    assert controls.current_time_s() == pytest.approx(0.0)
    assert tab._error_origin == "selection"
    assert tab._error_status.text()

    qtbot.keyClick(canvas, Qt.Key.Key_End)
    accepted = tab.accepted_study()
    assert accepted is not None
    assert view.selected_raw_index() == accepted.plan.raw_count - 1
    assert controls.current_time_s() == pytest.approx(controls._duration_s)
    assert tab._error_status.text() == ""
    assert tab._error_origin is None

    tab._show_error(RuntimeError("retained scientific failure"))
    qtbot.keyClick(canvas, Qt.Key.Key_Home)
    assert tab._error_status.text()
    assert tab._error_origin == "scientific"


def test_double_selection_failure_warns_that_marker_pixels_may_be_stale(
    accepted_tab, qtbot, monkeypatch
) -> None:  # type: ignore[no-untyped-def]
    tab = accepted_tab
    view = tab.flight_view()
    canvas = view._canvas
    controls = tab._flight_panel.controls
    canvas.setFocus()
    qtbot.keyClick(canvas, Qt.Key.Key_Home)
    original_canvas_draw = canvas.draw
    calls = 0

    def publish_then_fail_and_block_rollback() -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            original_canvas_draw()
            raise RuntimeError("candidate marker pixels published")
        raise RuntimeError("prior marker pixels could not be restored")

    monkeypatch.setattr(canvas, "draw", publish_then_fail_and_block_rollback)
    qtbot.keyClick(canvas, Qt.Key.Key_End)
    assert view.selected_raw_index() == 0
    assert controls.current_time_s() == pytest.approx(0.0)
    assert calls == 2
    assert "plot restoration failed" in tab._error_status.text().lower()
    assert "image may be stale or unavailable" in tab._error_status.text().lower()
    assert not canvas.has_pending_draw()
    monkeypatch.undo()
    qtbot.keyClick(canvas, Qt.Key.Key_End)
    accepted = tab.accepted_study()
    assert accepted is not None
    assert view.selected_raw_index() == accepted.plan.raw_count - 1
    assert tab._error_status.text() == ""
    qtbot.waitUntil(lambda: not canvas.has_pending_draw())


def test_public_bundle_identity_and_delivery_ignore_direction_presentation(
    accepted_tab,
) -> None:  # type: ignore[no-untyped-def]
    tab = accepted_tab
    first = tab.accepted_study()
    assert first is not None
    assert tab.last_exploration() is first.exploration

    tab._mode_combo.setCurrentIndex(1)
    tab._speed_spin.setValue(112.0)
    assert tab.run_now() is not None
    delivery = tab.accepted_study()
    assert delivery is not None and delivery is not first
    assert delivery.generation == first.generation + 1
    assert delivery.context.direction_convention is LaunchDirectionConvention.APP_NATIVE
    displayed = tab._context_status.text()

    tab._direction_convention_combo.setCurrentIndex(1)
    assert tab._context_status.text() == displayed
    assert "trackman" not in delivery.context.label().lower()


def test_speed_unit_switch_preserves_canonical_value_and_domain(accepted_tab) -> None:  # type: ignore[no-untyped-def]
    tab = accepted_tab
    before = tab.speed_mps()
    before_mph = tab.speed_mph()
    before_accepted = tab.accepted_study()
    assert before_accepted is not None
    displayed = tab._context_status.text()
    for unit, factor in SPEED_UNITS.items():
        tab._speed_unit_combo.setCurrentText(unit)
        assert tab.speed_mps() == before
        assert tab.speed_mph() == before_mph
        assert tab._context_status.text() == displayed
        assert tab._speed_spin.minimum() == pytest.approx(1.0 / factor)
        assert tab._speed_spin.maximum() == pytest.approx(250.0 / factor)
        assert tab._speed_spin.value() == pytest.approx(before_mph / factor)
    assert tab.run_now() is not None
    accepted = tab.accepted_study()
    assert accepted is not None
    assert (
        accepted.context.expected_launch.ball_speed
        == before_accepted.context.expected_launch.ball_speed
    )
    for unit, factor in SPEED_UNITS.items():
        tab._speed_unit_combo.setCurrentText(unit)
        tab._speed_spin.setValue(100.0 / factor)
        assert tab.speed_mph() == pytest.approx(tab._speed_spin.value() * factor)


def test_active_scientific_spin_changes_mark_prior_without_waiting_for_blur(
    accepted_tab,
) -> None:  # type: ignore[no-untyped-def]
    tab = accepted_tab
    tab._direct_spins["launch_angle_deg"].stepUp()
    assert tab._context_status.text().startswith("Prior result — inputs changed:")

    assert tab.run_now() is not None
    tab.wind_controls.speed_spin.stepUp()
    assert tab._context_status.text().startswith("Displayed flight:")
    tab.wind_controls.enabled_check.setChecked(True)
    assert tab.run_now() is not None
    tab.wind_controls.speed_spin.stepUp()
    assert tab._context_status.text().startswith("Prior result — inputs changed:")

    tab.wind_controls.enabled_check.setChecked(False)
    tab._mode_combo.setCurrentIndex(1)
    assert tab.run_now() is not None
    tab._delivery_spins["dynamic_loft_deg"].stepUp()
    assert tab._context_status.text().startswith("Prior result — inputs changed:")
