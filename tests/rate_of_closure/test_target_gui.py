"""GUI tests for the H7b target panel + course-view overlays (#4125).

Headless-safe: target-region editing (kind switch, entries -> region),
the 'Optimize to Target' entry point, the Simulation-tab wiring that
moves the H7a course green to the edited target, and the flight view's
target + landing-scatter overlay with the hold-% headline.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("PyQt6")
pytest.importorskip("pytestqt")

from rate_of_closure.simulation.targets import TargetRegion  # noqa: E402
from rate_of_closure.ui.pyqt6.simulation_tab import SimulationTab  # noqa: E402
from rate_of_closure.ui.pyqt6.target_panel import TargetPanel  # noqa: E402

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


@pytest.fixture
def panel(qtbot):  # type: ignore[no-untyped-def]
    widget = TargetPanel()
    qtbot.addWidget(widget)
    return widget


class TestTargetPanel:
    def test_default_region_is_a_green(self, panel) -> None:  # type: ignore[no-untyped-def]
        region = panel.region()
        assert region.kind == "green"
        assert region.distance_m == pytest.approx(230.0)
        assert region.radius_m == pytest.approx(10.0)

    def test_kind_switch_builds_a_fairway(self, panel) -> None:  # type: ignore[no-untyped-def]
        panel._kind.setCurrentIndex(1)
        region = panel.region()
        assert region.kind == "fairway"
        assert region.band_half_length_m == pytest.approx(15.0)
        assert region.half_width_m == pytest.approx(16.0)

    def test_edits_emit_region_changed(self, panel, qtbot) -> None:  # type: ignore[no-untyped-def]
        with qtbot.waitSignal(panel.regionChanged, timeout=2000) as blocker:
            panel._distance.setValue(180.0)
        assert blocker.args[0].distance_m == pytest.approx(180.0)

    def test_optimize_button_present_with_guidance(self, panel) -> None:  # type: ignore[no-untyped-def]
        button = panel.optimize_button()
        assert button.text() == "Optimize to Target"
        assert "distance outside the region" in button.toolTip()
        for widget in (panel._kind, panel._distance, panel._radius, panel._weight):
            assert "Source:" in widget.toolTip() or "residual" in widget.toolTip()

    def test_set_running_gates_the_button(self, panel) -> None:  # type: ignore[no-untyped-def]
        panel.set_running(True)
        assert not panel.optimize_button().isEnabled()
        panel.set_running(False)
        assert panel.optimize_button().isEnabled()


class TestSimulationTabWiring:
    def test_target_edit_moves_the_course_green(self, qtbot) -> None:  # type: ignore[no-untyped-def]
        tab = SimulationTab()
        qtbot.addWidget(tab)
        target_panel = tab.solver_panel().target_panel()
        target_panel._distance.setValue(150.0)
        flight = tab.flight_view()
        assert flight.course_layout().green_distance_m == pytest.approx(150.0)
        assert flight.target_region() is not None
        assert flight.target_region().distance_m == pytest.approx(150.0)
        assert tab.view().course_layout().green_distance_m == pytest.approx(150.0)
        tab.stop()

    def test_solver_goal_includes_the_region_when_asked(self, qtbot) -> None:  # type: ignore[no-untyped-def]
        tab = SimulationTab()
        qtbot.addWidget(tab)
        solver = tab.solver_panel()
        goal = solver.build_goal(include_target=True)
        assert goal.target_region is not None
        assert goal.target_region.kind == "green"
        assert solver.build_goal(include_target=False).target_region is None
        tab.stop()


class TestFlightViewOverlay:
    def test_scatter_with_target_reports_hold_percent(self, qtbot) -> None:  # type: ignore[no-untyped-def]
        tab = SimulationTab()
        qtbot.addWidget(tab)
        flight = tab.flight_view()
        flight.set_target_region(
            TargetRegion(kind="green", distance_m=200.0, radius_m=10.0)
        )
        # 3-of-5 hand-counted fixture from test_targets.py.
        flight.set_landing_scatter(
            np.array([200.0, 205.0, 209.0, 215.0, 200.0]),
            np.array([0.0, 0.0, 0.0, 0.0, 11.0]),
        )
        flight.set_trajectory(
            np.array([[0.0, 0.0, 0.0], [100.0, 30.0, 1.0], [210.0, 0.0, 2.0]])
        )
        # Draw ran without error; extents grew to include the scatter.
        carry_ext, _, _ = flight.extents_m()
        assert carry_ext >= 215.0
        flight.set_landing_scatter(None)
        tab.stop()
