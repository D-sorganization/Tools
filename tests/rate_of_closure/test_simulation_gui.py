"""PyQt6 GUI smoke tests for the Simulation tab (epic #4103).

Headless-safe (Agg-compatible matplotlib embedding, timers stopped).
Covers: tab presence in the main window, a full run populating launch
rows / scene / inspector, scrubber-driven reruns, playback controls
(rate presets, frame stepping, loop), scene toggles, sourced hover
guidance on every new input, and export-button gating.
"""

from __future__ import annotations

import json

import pytest

pytest.importorskip("PyQt6")
pytest.importorskip("pytestqt")

from PyQt6.QtCore import QSettings  # noqa: E402

from rate_of_closure.derivation import LAUNCH_EXPLANATIONS  # noqa: E402
from rate_of_closure.model import ImpactScenario  # noqa: E402
from rate_of_closure.simulation import SimulationRun  # noqa: E402
from rate_of_closure.ui.pyqt6.main_window import RateOfClosureMainWindow  # noqa: E402
from rate_of_closure.ui.pyqt6.simulation_tab import (  # noqa: E402
    LAUNCH_ROWS,
    SimulationTab,
)
from rate_of_closure.ui.pyqt6.simulation_view import (  # noqa: E402
    RATE_PRESETS,
    SimulationView,
)

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
            assert isinstance(window._simulation_tab.last_run(), SimulationRun)
            assert window._simulation_tab.view()._run is not None
        finally:
            window._club_view.stop()
            window._simulation_tab.stop()

    def test_simulation_club_selection_uses_the_canonical_workbench_spec(
        self, qtbot
    ) -> None:  # type: ignore[no-untyped-def]
        window = RateOfClosureMainWindow()
        qtbot.addWidget(window)
        try:
            window._simulation_tab._club_combo.setCurrentText("Pitching Wedge")
            assert window._controls._club_combo.currentText() == "Pitching Wedge"
            config = window._simulation_tab.config()
            assert config.club == window._controls.club_spec()
            assert config.scenario.lie_angle_deg == 64.0
            assert config.scenario.com_to_face_mm == 11.0
        finally:
            window._club_view.stop()
            window._simulation_tab.stop()

    def test_every_launch_row_has_an_explanation(self) -> None:
        for field, _label, _unit in LAUNCH_ROWS:
            assert field in LAUNCH_EXPLANATIONS, field

    def test_default_simulation_club_is_representative_driver(self, tab) -> None:  # type: ignore[no-untyped-def]
        assert tab._club_combo.currentText() == "Driver 10.5°"

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
            *tab.view()._impact_layer_checks.values(),
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

    def test_source_change_discards_stale_manual_impact_time(self, ran_tab) -> None:  # type: ignore[no-untyped-def]
        assert ran_tab.last_run().impact_time_s == pytest.approx(0.03)
        ran_tab._source_combo.setCurrentIndex(2)  # triple pendulum
        assert ran_tab._tau is None
        run = ran_tab.run_now()
        assert run is not None
        assert run.config.source_kind == "triple_pendulum"
        assert run.swing_joints.shape[1] == 4


class TestSimulationView:
    def test_rate_presets_cover_spec_and_round_trip(self, ran_tab) -> None:  # type: ignore[no-untyped-def]
        view = ran_tab.view()
        assert [rate for _name, rate in RATE_PRESETS] == [0.1, 0.25, 0.5, 1.0, 2.0]
        view.set_playback_rate(0.25)
        assert view.playback_rate() == pytest.approx(0.25)
        view.set_playback_rate(1.0)
        assert view.playback_rate() == pytest.approx(1.0)

    def test_playback_rate_accepts_granular_values(self, ran_tab) -> None:  # type: ignore[no-untyped-def]
        view = ran_tab.view()
        view.set_playback_rate(1.35)
        assert view.playback_rate() == pytest.approx(1.35)

    def test_play_at_end_rewinds_and_restart_is_explicit(self, ran_tab) -> None:  # type: ignore[no-untyped-def]
        view = ran_tab.view()
        view.set_playback_time(ran_tab.last_run().total_duration_s)

        view._play_button.click()

        assert view.is_playing()
        assert view.playback_time() == pytest.approx(0.0)
        view._restart_button.click()
        assert not view.is_playing()
        assert view.playback_time() == pytest.approx(0.0)

    def test_path_trail_is_opt_in(self, ran_tab) -> None:  # type: ignore[no-untyped-def]
        view = ran_tab.view()
        assert not view._trail_check.isChecked()
        assert "clubhead path" not in {
            str(line.get_label()) for line in view._axes.lines
        }

        view._trail_check.setChecked(True)

        assert "clubhead path" in {str(line.get_label()) for line in view._axes.lines}

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

    def test_jump_to_impact_uses_the_canonical_inspection_event(self, ran_tab) -> None:  # type: ignore[no-untyped-def]
        view = ran_tab.view()
        run = ran_tab.last_run()
        view.set_playback_time(0.0)

        view._inspection_button.click()

        assert view.playback_time() == pytest.approx(run.inspection_time_s)
        assert view._inspection_button.text() == "Jump to Impact"
        assert "Contact-Point AoA" in view._impact_kinematics_readout.text()
        assert "Shaft AoA Contribution" in view._impact_kinematics_readout.text()
        assert (
            "Sasho Face-Center Rotation-Only AoA"
            in view._impact_kinematics_readout.text()
        )
        assert "AoA Method Options" in view._impact_kinematics_readout.text()
        assert "Geometry Basis" in view._impact_kinematics_readout.text()

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

    def test_impact_inspector_draws_engineering_geometry_and_vectors(
        self, ran_tab
    ) -> None:  # type: ignore[no-untyped-def]
        view = ran_tab.view()
        view._impact_check.setChecked(True)
        for check in view._impact_layer_checks.values():
            check.setChecked(True)
        view.jump_to_inspection_event()

        labels = {str(line.get_label()) for line in view._axes.lines}
        assert {
            "Physical Shaft Axis",
            "Wedge Face",
            "Leading Edge",
            "Face-Center Normal",
            "Face-Center Travel",
            "D-Plane Normal",
            "Arc Tangent",
            "Total Contact Velocity",
            "Rotation About Shaft",
            "Without Shaft Rotation",
            "Sasho Face-Center Rotation",
        } <= labels
        collection_labels = {
            str(collection.get_label()) for collection in view._axes.collections
        }
        assert "3D Spin-Loft Sector" in collection_labels

        view._impact_layer_checks["face_normal"].setChecked(False)
        view._impact_layer_checks["face_center_travel"].setChecked(False)
        view._impact_layer_checks["dplane_normal"].setChecked(False)
        view._impact_layer_checks["spin_loft_sector"].setChecked(False)
        view._impact_layer_checks["sasho_face_center_rotation"].setChecked(False)
        labels = {str(line.get_label()) for line in view._axes.lines}
        collection_labels = {
            str(collection.get_label()) for collection in view._axes.collections
        }
        assert "Face-Center Normal" not in labels
        assert "Face-Center Travel" not in labels
        assert "D-Plane Normal" not in labels
        assert "3D Spin-Loft Sector" not in collection_labels
        assert "Sasho Face-Center Rotation" not in labels

    def test_impact_layers_are_independent_and_persisted(self, qtbot, tmp_path) -> None:  # type: ignore[no-untyped-def]
        settings_path = tmp_path / "impact-layers.ini"
        first_settings = QSettings(str(settings_path), QSettings.Format.IniFormat)
        first = SimulationView(impact_settings=first_settings)
        qtbot.addWidget(first)
        assert first._impact_layer_checks is first._impact_layer_controls.checks
        first._impact_layer_checks["spin_loft_sector"].setChecked(False)
        first._impact_layer_checks["face_center_travel"].setChecked(False)
        first_settings.sync()

        second_settings = QSettings(str(settings_path), QSettings.Format.IniFormat)
        second = SimulationView(impact_settings=second_settings)
        qtbot.addWidget(second)

        assert second._impact_layer_checks is second._impact_layer_controls.checks
        assert not second._impact_layer_checks["spin_loft_sector"].isChecked()
        assert not second._impact_layer_checks["face_center_travel"].isChecked()
        assert second._impact_layer_checks["face_normal"].isChecked()
        assert second._impact_layer_checks["sasho_face_center_rotation"].isChecked()

    def test_impact_scene_exports_strict_data_and_true_vector_artwork(
        self, ran_tab, tmp_path
    ) -> None:  # type: ignore[no-untyped-def]
        view = ran_tab.view()
        data_path = view.export_impact_scene(tmp_path / "impact.json")
        vector_path = view.export_impact_scene(tmp_path / "impact.svg")

        assert '"format": "rate-of-closure.impact-scene/v3"' in data_path.read_text(
            encoding="utf-8"
        )
        payload = json.loads(data_path.read_text(encoding="utf-8"))
        assert payload["render_preferences"]["visible_layers"] == sorted(
            view.impact_visible_layers()
        )
        assert set(payload["render_preferences"]["camera"]) == {
            "elevation_deg",
            "azimuth_deg",
        }
        svg = vector_path.read_text(encoding="utf-8")
        assert "<svg" in svg
        assert "Physical Shaft Axis" in svg
        assert view.playback_time() == pytest.approx(
            ran_tab.last_run().inspection_time_s
        )

    def test_named_impact_camera_preserves_locked_physical_scaling(
        self, ran_tab
    ) -> None:  # type: ignore[no-untyped-def]
        view = ran_tab.view()
        view._impact_view.setCurrentText("Face-On")

        assert view._axes.azim == pytest.approx(-90.0)
        aspect = view._axes.get_box_aspect()
        assert aspect[0] / aspect[1] == pytest.approx(1.0)
        assert aspect[0] / aspect[2] == pytest.approx(2.0 / 1.4)

    def test_screw_axis_overlay_appears_during_swing(self, ran_tab) -> None:  # type: ignore[no-untyped-def]
        view = ran_tab.view()
        view._screw_check.setChecked(True)
        view.set_playback_time(ran_tab.last_run().impact_time_s * 0.5)
        labels = [line.get_label() for line in view._axes.lines]
        assert any("Screw Axis" in str(label) for label in labels)
        assert any("Helical Motion" in str(label) for label in labels)
        assert "Finite screw" in view._screw_readout.text()
        assert "Orbital" in view._screw_readout.text()
        assert "total = orbital + axial" in view._screw_readout.text()

    def test_screw_selector_exposes_club_and_articulated_joints(
        self, tab, qtbot
    ) -> None:  # type: ignore[no-untyped-def]
        tab._source_combo.setCurrentIndex(1)
        with qtbot.waitSignal(tab.runCompleted, timeout=10000):
            tab.run_now()
        view = tab.view()
        choices = [
            view._screw_entity.itemText(i) for i in range(view._screw_entity.count())
        ]
        assert choices == ["Club", "Shoulder Joint", "Wrist Joint"]
        view._screw_check.setChecked(True)
        view._screw_entity.setCurrentIndex(1)
        view.set_playback_time(0.5)
        assert "Shoulder Joint" in view._screw_readout.text()
        assert "Contribution" in view._screw_readout.text()


class TestInspector:
    def test_export_buttons_gate_on_run(self, tab, ran_tab) -> None:  # type: ignore[no-untyped-def]
        fresh = tab.inspector()
        assert not fresh._export_csv_button.isEnabled() or fresh.run() is not None
        inspector = ran_tab.inspector()
        assert inspector._export_csv_button.isEnabled()
        assert inspector._export_screw_csv_button.isEnabled()
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
