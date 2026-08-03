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
from rate_of_closure.ui.pyqt6.controls_panel import ControlsPanel  # noqa: E402
from rate_of_closure.ui.pyqt6.main_window import (  # noqa: E402
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
        label = window._results_labels["path_deviation_deg"]
        assert "°" in label.text()
        assert label.text() != "—"

    def test_zero_rotation_reports_zero_deviation(self, window, qtbot) -> None:  # type: ignore[no-untyped-def]
        window._controls.apply_preset("Zero rotation (control)")
        text = window._results_labels["path_deviation_deg"].text()
        assert text.startswith(("+0.00", "-0.00"))

    def test_tour_preset_reports_leftward_deviation(self, window) -> None:  # type: ignore[no-untyped-def]
        window._controls.apply_preset("Cheetham tour median (HTV 1,307)")
        text = window._results_labels["path_deviation_deg"].text()
        assert text.startswith("-1.5")

    def test_status_bar_narrates_direction(self, window) -> None:  # type: ignore[no-untyped-def]
        window._controls.apply_preset("Cheetham tour median (HTV 1,307)")
        assert "left" in window.statusBar().currentMessage()
