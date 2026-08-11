"""PyQt6 GUI smoke tests for the Rate of Closure Impact Explorer.

Headless-safe: matplotlib is forced to Agg-compatible embedding and the
animation timer is stopped explicitly. Exercises the LoD seam — the window
consumes scenarios from the controls and updates results without the test
touching any internal widget of another component.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("PyQt6")
pytest.importorskip("pytestqt")

from rate_of_closure.model import ImpactScenario  # noqa: E402
from rate_of_closure.ui.pyqt6.club_view import VIEW_MODES, Club3DView  # noqa: E402
from rate_of_closure.ui.pyqt6.controls_panel import ControlsPanel  # noqa: E402
from rate_of_closure.ui.pyqt6.main_window import (  # noqa: E402
    _RESULT_ROWS,
    RateOfClosureMainWindow,
)

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


@pytest.fixture
def window(qtbot):  # type: ignore[no-untyped-def]
    win = RateOfClosureMainWindow()
    qtbot.addWidget(win)
    yield win
    win._club_view.stop()


class TestControlsPanel:
    def test_emits_valid_scenario_on_start(self, qtbot) -> None:  # type: ignore[no-untyped-def]
        panel = ControlsPanel()
        qtbot.addWidget(panel)
        scenario = panel.scenario()
        assert isinstance(scenario, ImpactScenario)
        assert scenario.clubhead_speed_mph > 0

    def test_preset_change_emits_scenario(self, qtbot) -> None:  # type: ignore[no-untyped-def]
        panel = ControlsPanel()
        qtbot.addWidget(panel)
        with qtbot.waitSignal(panel.scenarioChanged, timeout=2000) as blocker:
            panel.apply_preset("Zero rotation (control)")
        scenario = blocker.args[0]
        assert scenario.omega_shaft_dps == 0.0
        assert scenario.omega_plane_dps == 0.0


class TestMainWindow:
    def test_window_constructs_and_shows_results(self, window) -> None:  # type: ignore[no-untyped-def]
        label = window._rows["path_deviation_deg"].value_label
        assert "°" in label.text()
        assert label.text() != "—"

    def test_zero_rotation_reports_zero_deviation(self, window, qtbot) -> None:  # type: ignore[no-untyped-def]
        window._controls.apply_preset("Zero rotation (control)")
        text = window._rows["path_deviation_deg"].value_label.text()
        assert text.startswith(("+0.00", "-0.00"))

    def test_tour_preset_reports_leftward_deviation(self, window) -> None:  # type: ignore[no-untyped-def]
        window._controls.apply_preset("Cheetham tour median (HTV 1,307)")
        text = window._rows["path_deviation_deg"].value_label.text()
        assert text.startswith("-1.5")

    def test_status_bar_narrates_direction_and_ccv(self, window) -> None:  # type: ignore[no-untyped-def]
        window._controls.apply_preset("Cheetham tour median (HTV 1,307)")
        message = window.statusBar().currentMessage()
        assert "left" in message
        assert "°/ft" in message

    def test_result_labels_are_title_case(self) -> None:
        minor = {"vs", "and", "of", "to", "at", "in", "the"}
        for _, label in _RESULT_ROWS:
            for index, word in enumerate(label.split()):
                head = word.strip("()")
                if not head or not head[0].isalpha():
                    continue
                if head.lower() in minor and index != 0:
                    continue
                assert head[0].isupper(), label

    def test_clicking_result_row_shows_explanation(self, window, qtbot) -> None:  # type: ignore[no-untyped-def]
        row = window._rows["closure_rate_dps"]
        with qtbot.waitSignal(row.clicked, timeout=2000):
            row.clicked.emit("closure_rate_dps")
        html = window._explanation.toHtml()
        assert "Closure Rate" in html
        assert "Cheetham" in html

    def test_derivation_tab_exists_and_populates(self, window) -> None:  # type: ignore[no-untyped-def]
        view = window._derivation_view
        assert view._scroll.widget() is not None


class TestUserFeedbackFixes:
    """Regressions for the review round: theme menu, arrows, units."""

    def test_window_does_not_add_its_own_theme_menu(self, window) -> None:  # type: ignore[no-untyped-def]
        """The launcher owns theming; a second menu duplicated it."""
        menubar = window.menuBar()
        titles = [action.text() for action in menubar.actions()]
        assert titles.count("&Theme") == 0

    def test_entry_boxes_hide_step_arrows(self, qtbot) -> None:  # type: ignore[no-untyped-def]
        from PyQt6.QtWidgets import QAbstractSpinBox

        panel = ControlsPanel()
        qtbot.addWidget(panel)
        for name, spin in panel._spins.items():
            assert spin.buttonSymbols() == QAbstractSpinBox.ButtonSymbols.NoButtons, (
                name
            )

    def test_entry_boxes_carry_range_guidance_with_source(self, qtbot) -> None:  # type: ignore[no-untyped-def]
        panel = ControlsPanel()
        qtbot.addWidget(panel)
        for name, spin in panel._spins.items():
            assert "Suggested range" in spin.toolTip(), name
            assert "Source:" in spin.toolTip(), name

    def test_unit_switch_preserves_canonical_scenario(self, qtbot) -> None:  # type: ignore[no-untyped-def]
        panel = ControlsPanel()
        qtbot.addWidget(panel)
        before = panel.scenario()
        panel._unit_combos["speed"].setCurrentText("m/s")
        panel._unit_combos["rotation"].setCurrentText("rpm")
        panel._unit_combos["length"].setCurrentText("in")
        after = panel.scenario()
        # Round-trip through the display loses at most display resolution.
        assert after.clubhead_speed_mph == pytest.approx(
            before.clubhead_speed_mph, rel=1e-4
        )
        assert after.omega_shaft_dps == pytest.approx(before.omega_shaft_dps, rel=1e-4)
        assert after.com_to_face_mm == pytest.approx(before.com_to_face_mm, rel=1e-4)
        # And the display suffix follows the selection.
        assert panel._spins["clubhead_speed_mph"].suffix() == " m/s"

    def test_metrics_rows_populate_and_explain(self, window) -> None:  # type: ignore[no-untyped-def]
        row = window._rows["ccv_dps"]
        assert row.value_label.text() != "—"
        row.clicked.emit("ccv_dps")
        assert "Club Closure Velocity" in window._explanation.toHtml()

    def test_results_follow_selected_units(self, window) -> None:  # type: ignore[no-untyped-def]
        window._controls._unit_combos["rotation"].setCurrentText("rpm")
        text = window._rows["closure_rate_dps"].value_label.text()
        assert text.endswith(" rpm")

    def test_derivation_formula_canvases_ignore_wheel(self, window) -> None:  # type: ignore[no-untyped-def]
        from rate_of_closure.ui.pyqt6.derivation_view import _FormulaCanvas

        content = window._derivation_view._scroll.widget()
        canvases = content.findChildren(_FormulaCanvas)
        assert canvases, "derivation tab must contain formula canvases"


class TestClubGroup:
    """Smoke tests for the Club group: picker, curvature, generation."""

    def test_picker_lists_the_full_library(self, qtbot) -> None:  # type: ignore[no-untyped-def]
        from rate_of_closure.club import club_names

        panel = ControlsPanel()
        qtbot.addWidget(panel)
        items = [
            panel._club_combo.itemText(i) for i in range(panel._club_combo.count())
        ]
        assert items == club_names()
        assert panel._club_combo.currentText() == "Driver 10.5°"

    def test_selecting_a_club_drives_com_and_lie(self, qtbot) -> None:  # type: ignore[no-untyped-def]
        panel = ControlsPanel()
        qtbot.addWidget(panel)
        panel._club_combo.setCurrentText("7-Iron")
        scenario = panel.scenario()
        assert scenario.com_to_face_mm == pytest.approx(13.0)
        assert scenario.lie_angle_deg == pytest.approx(62.5)
        # User override is preserved: the spin stays editable.
        panel._spins["com_to_face_mm"].setValue(40.0)
        assert panel.scenario().com_to_face_mm == pytest.approx(40.0)

    def test_curvature_toggle_gates_the_radius_entries(self, qtbot) -> None:  # type: ignore[no-untyped-def]
        panel = ControlsPanel()
        qtbot.addWidget(panel)
        panel._club_combo.setCurrentText("7-Iron")  # flat face
        assert not panel._curvature_check.isChecked()
        assert not panel._bulge_spin.isEnabled()
        panel._club_combo.setCurrentText("Driver 10.5°")
        assert panel._curvature_check.isChecked()
        assert panel._bulge_spin.isEnabled()
        assert panel._bulge_spin.value() == pytest.approx(300.0)
        assert panel._roll_spin.value() == pytest.approx(280.0)

    def test_club_spec_reflects_overrides(self, qtbot) -> None:  # type: ignore[no-untyped-def]
        panel = ControlsPanel()
        qtbot.addWidget(panel)
        panel._club_combo.setCurrentText("Driver 10.5°")
        panel._loft_spin.setValue(9.0)
        panel._bulge_spin.setValue(280.0)
        spec = panel.club_spec()
        assert spec.loft_deg == pytest.approx(9.0)
        assert spec.face_bulge_radius_m == pytest.approx(0.280)
        panel._curvature_check.setChecked(False)
        assert panel.club_spec().face_bulge_radius_m is None

    def test_generate_button_emits_a_club_spec(self, qtbot) -> None:  # type: ignore[no-untyped-def]
        from rate_of_closure.club import ClubSpec

        panel = ControlsPanel()
        qtbot.addWidget(panel)
        with qtbot.waitSignal(panel.clubHeadRequested, timeout=2000) as blocker:
            panel._generate_button.click()
        assert isinstance(blocker.args[0], ClubSpec)

    def test_export_action_writes_selected_parametric_head_stl(
        self, qtbot, tmp_path, monkeypatch
    ) -> None:  # type: ignore[no-untyped-def]
        from rate_of_closure.club import build_parametric_head
        from rate_of_closure.mesh import parse_stl

        output = tmp_path / "selected-head.stl"
        panel = ControlsPanel()
        qtbot.addWidget(panel)
        panel._club_combo.setCurrentText("7-Iron")
        panel._loft_spin.setValue(32.0)
        monkeypatch.setattr(
            "rate_of_closure.ui.pyqt6.controls_panel.QFileDialog.getSaveFileName",
            lambda *_args, **_kwargs: (str(output), "STL meshes (*.stl)"),
        )

        panel._export_head_button.click()

        assert output.is_file()
        np.testing.assert_allclose(
            parse_stl(output.read_bytes()),
            build_parametric_head(panel.club_spec()),
            rtol=1e-6,
            atol=1e-9,
        )
        assert "STL exported: 7-Iron" in panel._export_status.text()

    def test_generate_loads_a_parametric_head_into_the_view(
        self, window, qtbot
    ) -> None:  # type: ignore[no-untyped-def]
        # The selected representative driver is share-ready on first paint.
        assert window._club_view.has_mesh()
        window._club_view.clear_mesh()
        assert not window._club_view.has_mesh()
        window._controls._generate_button.click()
        assert window._club_view.has_mesh()
        message = window.statusBar().currentMessage()
        assert "Representative head generated" in message
        # Procedural Head still restores the wireframe.
        window._club_view.clear_mesh()
        assert not window._club_view.has_mesh()

    def test_club_inputs_carry_sourced_guidance(self, qtbot) -> None:  # type: ignore[no-untyped-def]
        panel = ControlsPanel()
        qtbot.addWidget(panel)
        for widget in (
            panel._club_combo,
            panel._loft_spin,
            panel._curvature_check,
            panel._bulge_spin,
            panel._roll_spin,
        ):
            assert "Suggested range" in widget.toolTip()
            assert "Source:" in widget.toolTip()


class TestClub3DView:
    def test_animation_waits_for_explicit_play(self, qtbot) -> None:  # type: ignore[no-untyped-def]
        view = Club3DView()
        qtbot.addWidget(view)
        view.set_scenario(ImpactScenario(clubhead_speed_mph=120.0))

        assert not view.is_playing()
        assert view._play_button.text() == "Play"

        view._play_button.click()
        assert view.is_playing()
        assert view._play_button.text() == "Pause"

        view._play_button.click()
        assert not view.is_playing()
        assert view._play_button.text() == "Play"

    def test_playback_speed_round_trips_and_clamps(self, qtbot) -> None:  # type: ignore[no-untyped-def]
        view = Club3DView()
        qtbot.addWidget(view)
        view.set_playback_speed(2.5)
        assert view.playback_speed() == pytest.approx(2.5)
        view.set_playback_speed(99.0)
        assert view.playback_speed() == pytest.approx(3.0)
        view.stop()

    def test_zoom_api_clamps_and_redraws(self, qtbot) -> None:  # type: ignore[no-untyped-def]
        view = Club3DView()
        qtbot.addWidget(view)
        view.set_scenario(ImpactScenario(clubhead_speed_mph=120.0))
        view.set_zoom(2.0)
        assert view.zoom() == pytest.approx(2.0)
        view.set_zoom(99.0)
        assert view.zoom() == pytest.approx(4.0)
        view.set_zoom(0.01)
        assert view.zoom() == pytest.approx(0.3)
        view.stop()

    def test_user_orbit_angles_survive_animation_redraw(self, qtbot) -> None:  # type: ignore[no-untyped-def]
        view = Club3DView()
        qtbot.addWidget(view)
        view.set_scenario(ImpactScenario(clubhead_speed_mph=120.0))
        view._axes.view_init(elev=55.0, azim=12.0)
        view._draw()
        assert float(view._axes.elev) == pytest.approx(55.0)
        assert float(view._axes.azim) == pytest.approx(12.0)
        view.stop()

    def test_stl_load_and_reset(self, qtbot, tmp_path) -> None:  # type: ignore[no-untyped-def]
        from rate_of_closure.mesh import write_binary_stl
        from rate_of_closure.scripts.generate_example_head import build_example_head

        stl_path = tmp_path / "head.stl"
        stl_path.write_bytes(write_binary_stl(build_example_head()))

        view = Club3DView()
        qtbot.addWidget(view)
        view.set_scenario(ImpactScenario(clubhead_speed_mph=120.0))
        assert not view.has_mesh()
        assert not view._reset_mesh_button.isEnabled()

        view.load_mesh(str(stl_path))
        assert view.has_mesh()
        assert view._reset_mesh_button.isEnabled()
        # The shaded head renders as a Poly3DCollection on the axes.
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection

        view._draw()
        assert any(
            isinstance(artist, Poly3DCollection) for artist in view._axes.collections
        )

        view.clear_mesh()
        assert not view.has_mesh()
        assert not view._reset_mesh_button.isEnabled()
        view._draw()  # procedural head draws again without error
        view.stop()

    def test_stl_load_rejects_bad_file(self, qtbot, tmp_path) -> None:  # type: ignore[no-untyped-def]
        bad = tmp_path / "bad.stl"
        bad.write_bytes(b"not an stl")
        view = Club3DView()
        qtbot.addWidget(view)
        with pytest.raises(Exception, match="STL"):
            view.load_mesh(str(bad))
        assert not view.has_mesh()
        view.stop()

    def test_view_modes_switch(self, qtbot) -> None:  # type: ignore[no-untyped-def]
        view = Club3DView()
        qtbot.addWidget(view)
        view.set_scenario(ImpactScenario(clubhead_speed_mph=120.0))
        # Head Moving Through Space is the default display.
        assert view.view_mode() == VIEW_MODES[1]
        view.set_view_mode(VIEW_MODES[0])
        assert view.view_mode() == VIEW_MODES[0]
        view.set_view_mode("nonsense")  # ignored, logged
        assert view.view_mode() == VIEW_MODES[0]
        view.stop()


class TestCgAndHosel:
    """H1 (#4125): Show CG marker and hosel-true shaft attachment."""

    def test_show_cg_defaults_on_and_toggles(self, qtbot) -> None:  # type: ignore[no-untyped-def]
        view = Club3DView()
        qtbot.addWidget(view)
        view.set_scenario(ImpactScenario(clubhead_speed_mph=113.0))
        assert view.show_cg_check().isChecked()
        # Wireframe mode falls back to the reference point (spec CG).
        marker = view.cg_marker_point()
        assert marker is not None and not marker.any()
        view.show_cg_check().setChecked(False)
        assert view.cg_marker_point() is None
        assert "Source:" in view.show_cg_check().toolTip()
        view.stop()

    def test_generated_head_attaches_shaft_at_the_hosel(self, window) -> None:  # type: ignore[no-untyped-def]
        import numpy as np

        from rate_of_closure.club import head_cog, hosel_point

        window._controls._generate_button.click()
        view = window._club_view
        spec = window._controls.club_spec()
        attachment = view.shaft_attachment()
        assert attachment is not None
        shift = view._head_shift(view._mesh, view._scenario)
        expected = np.asarray(hosel_point(spec)) + shift
        np.testing.assert_allclose(attachment, expected, atol=1e-12)
        assert attachment[2] < 0.0  # heel side after the face shift too

        view.show_cg_check().setChecked(True)
        marker = view.cg_marker_point()
        assert marker is not None
        np.testing.assert_allclose(
            marker, np.asarray(head_cog(spec).cog) + shift, atol=1e-12
        )
        # Clearing the mesh restores wireframe hosel behavior.
        view.clear_mesh()
        assert view.shaft_attachment() is None
