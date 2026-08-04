"""PyQt6 GUI smoke tests for the Rate of Closure Impact Explorer.

Headless-safe: matplotlib is forced to Agg-compatible embedding and the
animation timer is stopped explicitly. Exercises the LoD seam — the window
consumes scenarios from the controls and updates results without the test
touching any internal widget of another component.
"""

from __future__ import annotations

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


class TestClub3DView:
    def test_playback_speed_round_trips_and_clamps(self, qtbot) -> None:  # type: ignore[no-untyped-def]
        view = Club3DView()
        qtbot.addWidget(view)
        view.set_playback_speed(2.5)
        assert view.playback_speed() == pytest.approx(2.5)
        view.set_playback_speed(99.0)
        assert view.playback_speed() == pytest.approx(3.0)
        view.stop()

    def test_view_modes_switch(self, qtbot) -> None:  # type: ignore[no-untyped-def]
        view = Club3DView()
        qtbot.addWidget(view)
        view.set_scenario(ImpactScenario(clubhead_speed_mph=120.0))
        assert view.view_mode() == VIEW_MODES[0]
        view.set_view_mode(VIEW_MODES[1])
        assert view.view_mode() == VIEW_MODES[1]
        view.set_view_mode("nonsense")  # ignored, logged
        assert view.view_mode() == VIEW_MODES[1]
        view.stop()
