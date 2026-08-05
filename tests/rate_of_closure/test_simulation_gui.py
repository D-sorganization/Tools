"""PyQt6 GUI smoke tests for the Simulation tab (epic #4103).

Headless-safe (Agg-compatible matplotlib embedding, timers stopped).
Covers: tab presence in the main window, a full run populating launch
rows / scene / inspector, scrubber-driven reruns, playback controls
(rate presets, frame stepping, loop), scene toggles, sourced hover
guidance on every new input, and export-button gating.
"""

from __future__ import annotations

import pytest

pytest.importorskip("PyQt6")
pytest.importorskip("pytestqt")

from rate_of_closure.derivation import LAUNCH_EXPLANATIONS  # noqa: E402
from rate_of_closure.model import ImpactScenario  # noqa: E402
from rate_of_closure.simulation import SimulationRun  # noqa: E402
from rate_of_closure.ui.pyqt6.main_window import RateOfClosureMainWindow  # noqa: E402
from rate_of_closure.ui.pyqt6.simulation_tab import (  # noqa: E402
    LAUNCH_ROWS,
    SimulationTab,
)
from rate_of_closure.ui.pyqt6.simulation_view import RATE_PRESETS  # noqa: E402

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


@pytest.fixture
def tab(qtbot):  # type: ignore[no-untyped-def]
    widget = SimulationTab()
    qtbot.addWidget(widget)
    widget.set_scenario(ImpactScenario(clubhead_speed_mph=113.0))
    yield widget
    widget.stop()


@pytest.fixture
def ran_tab(tab, qtbot):  # type: ignore[no-untyped-def]
    with qtbot.waitSignal(tab.runCompleted, timeout=10000):
        tab.run_now()
    return tab


class TestSimulationTab:
    def test_main_window_hosts_the_simulation_tab(self, qtbot) -> None:  # type: ignore[no-untyped-def]
        window = RateOfClosureMainWindow()
        qtbot.addWidget(window)
        try:
            tabs = window.centralWidget().findChildren(SimulationTab)
            assert tabs, "main window must host the Simulation tab"
        finally:
            window._club_view.stop()
            window._simulation_tab.stop()

    def test_every_launch_row_has_an_explanation(self) -> None:
        for field, _label, _unit in LAUNCH_ROWS:
            assert field in LAUNCH_EXPLANATIONS, field

    def test_run_populates_launch_rows(self, ran_tab) -> None:  # type: ignore[no-untyped-def]
        assert isinstance(ran_tab.last_run(), SimulationRun)
        for field, _label, _unit in LAUNCH_ROWS:
            assert ran_tab._rows[field].value_label.text() != "—", field

    def test_clicking_launch_row_shows_explanation(self, ran_tab) -> None:  # type: ignore[no-untyped-def]
        ran_tab._rows["carry_m"].clicked.emit("carry_m")
        html = ran_tab._explanation.toHtml()
        assert "Carry Distance" in html
        assert "flight model" in html

    def test_new_inputs_carry_sourced_guidance(self, tab) -> None:  # type: ignore[no-untyped-def]
        widgets = [
            tab._source_combo,
            tab._club_combo,
            tab._flight_combo,
            tab._scrub_slider,
            *tab._tilt_spins.values(),
            tab.view()._ball_check,
            tab.view()._ground_check,
            tab.view()._screw_check,
        ]
        for widget in widgets:
            assert "Suggested range" in widget.toolTip(), widget
            assert "Source:" in widget.toolTip(), widget

    def test_scrub_updates_tau_and_reruns(self, ran_tab) -> None:  # type: ignore[no-untyped-def]
        first_tau = ran_tab.last_run().impact_time_s
        ran_tab._scrub_slider.setValue(250)
        run = ran_tab.last_run()
        assert run.impact_time_s != pytest.approx(first_tau)
        assert "mph" in ran_tab._delivery_label.text()

    def test_auto_button_restores_max_speed_tau(self, ran_tab) -> None:  # type: ignore[no-untyped-def]
        ran_tab._scrub_slider.setValue(200)
        shifted = ran_tab.last_run().impact_time_s
        ran_tab._auto_tau_button.click()
        assert ran_tab.last_run().impact_time_s != pytest.approx(shifted)

    def test_pendulum_source_runs(self, tab, qtbot) -> None:  # type: ignore[no-untyped-def]
        tab._source_combo.setCurrentIndex(1)  # double pendulum
        with qtbot.waitSignal(tab.runCompleted, timeout=10000):
            run = tab.run_now()
        assert run is not None
        assert run.config.source_kind == "double_pendulum"

    def test_impact_model_selector_runs_interval_physics(self, tab, qtbot) -> None:  # type: ignore[no-untyped-def]
        tab._impact_model_combo.setCurrentText("Impact Interval (6-DOF)")
        with qtbot.waitSignal(tab.runCompleted, timeout=10000):
            run = tab.run_now()
        assert run is not None
        assert run.config.impact_model == "impact_interval"
        assert run.impact_interval is not None
        assert tab.impact_interval_view().run() is run
        assert tab.impact_interval_view()._position_slider.isEnabled()
        tab.impact_interval_view()._position_slider.setValue(500)
        assert "µs" in tab.impact_interval_view()._time_label.text()


class TestSimulationView:
    def test_rate_presets_cover_spec_and_round_trip(self, ran_tab) -> None:  # type: ignore[no-untyped-def]
        view = ran_tab.view()
        assert [rate for _name, rate in RATE_PRESETS] == [0.1, 0.25, 0.5, 1.0, 2.0]
        view.set_playback_rate(0.25)
        assert view.playback_rate() == pytest.approx(0.25)
        view.set_playback_rate(1.0)
        assert view.playback_rate() == pytest.approx(1.0)

    def test_frame_step_moves_by_one_sample(self, ran_tab) -> None:  # type: ignore[no-untyped-def]
        view = ran_tab.view()
        run = ran_tab.last_run()
        dt = float(run.swing_times[1] - run.swing_times[0])
        view.set_playback_time(0.010)
        view.step_frames(1)
        assert view.playback_time() == pytest.approx(0.010 + dt)
        view.step_frames(-2)
        assert view.playback_time() == pytest.approx(0.010 - dt)

    def test_playback_time_clamps_to_timeline(self, ran_tab) -> None:  # type: ignore[no-untyped-def]
        view = ran_tab.view()
        total = ran_tab.last_run().total_duration_s
        view.set_playback_time(total + 99.0)
        assert view.playback_time() == pytest.approx(total)
        view.set_playback_time(-1.0)
        assert view.playback_time() == pytest.approx(0.0)

    def test_play_pause_and_loop_toggle(self, ran_tab, qtbot) -> None:  # type: ignore[no-untyped-def]
        view = ran_tab.view()
        assert not view.is_playing()
        view._play_button.setChecked(True)
        assert view.is_playing()
        view._play_button.setChecked(False)
        assert not view.is_playing()
        view.set_looping(True)
        assert view._loop_check.isChecked()

    def test_scene_toggles_redraw_without_error(self, ran_tab) -> None:  # type: ignore[no-untyped-def]
        view = ran_tab.view()
        for check in (view._ball_check, view._ground_check, view._screw_check):
            check.setChecked(not check.isChecked())
        view.set_playback_time(ran_tab.last_run().impact_time_s)
        # Move into the flight phase too (different extent branch).
        view.set_playback_time(ran_tab.last_run().total_duration_s * 0.9)

    def test_screw_axis_overlay_appears_during_swing(self, ran_tab) -> None:  # type: ignore[no-untyped-def]
        view = ran_tab.view()
        view._screw_check.setChecked(True)
        view.set_playback_time(ran_tab.last_run().impact_time_s * 0.5)
        labels = [line.get_label() for line in view._axes.lines]
        assert any("screw axis" in str(label) for label in labels)


class TestInspector:
    def test_export_buttons_gate_on_run(self, tab, ran_tab) -> None:  # type: ignore[no-untyped-def]
        fresh = tab.inspector()
        assert not fresh._export_csv_button.isEnabled() or fresh.run() is not None
        inspector = ran_tab.inspector()
        assert inspector._export_csv_button.isEnabled()
        assert inspector._export_json_button.isEnabled()

    def test_table_populates_and_sorts(self, ran_tab) -> None:  # type: ignore[no-untyped-def]
        run = ran_tab.last_run()
        table = ran_tab.inspector()._table
        expected = len(run.swing_times) + len(run.flight_times)
        assert table.rowCount() == expected
        table.sortItems(1)  # by time ascending — numeric sort
        first = float(table.item(0, 1).data(0x0100))  # Qt.UserRole
        last = float(table.item(table.rowCount() - 1, 1).data(0x0100))
        assert first <= last

    def test_summary_mentions_club_and_carry(self, ran_tab) -> None:  # type: ignore[no-untyped-def]
        text = ran_tab.inspector()._summary_label.text()
        assert ran_tab.last_run().config.club.name in text
        assert "Carry" in text
