"""PyQt6 GUI smoke tests for the swing-kinetics feature (#4125 H2).

Headless-safe. Covers: the Kinetics sub-tab populating (plots + peak
table) from a double-pendulum run and clearing for unsupported
sources, the 'Show Kinetics' overlay checkbox drawing arcs/arrows in
the swing scene, overlay-geometry sanity, and hover guidance.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("PyQt6")
pytest.importorskip("pytestqt")

from PyQt6.QtWidgets import QTabWidget  # noqa: E402

from rate_of_closure.model import ImpactScenario  # noqa: E402
from rate_of_closure.simulation import kinetics_for_run  # noqa: E402
from rate_of_closure.ui.pyqt6.kinetics_overlay import (  # noqa: E402
    MAX_ARC_RADIUS_M,
    MAX_ARROW_LENGTH_M,
    overlay_frame,
)
from rate_of_closure.ui.pyqt6.kinetics_panel import (  # noqa: E402
    PEAK_TABLE_COLUMNS,
    KineticsPanel,
)
from rate_of_closure.ui.pyqt6.simulation_tab import SimulationTab  # noqa: E402

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


@pytest.fixture
def tab(qtbot):  # type: ignore[no-untyped-def]
    widget = SimulationTab()
    qtbot.addWidget(widget)
    widget.set_scenario(ImpactScenario(clubhead_speed_mph=113.0))
    yield widget
    widget.stop()


@pytest.fixture
def pendulum_tab(tab, qtbot):  # type: ignore[no-untyped-def]
    tab._source_combo.setCurrentIndex(1)  # double pendulum
    with qtbot.waitSignal(tab.runCompleted, timeout=15000):
        tab.run_now()
    return tab


class TestKineticsPanel:
    def test_sub_tab_is_hosted_by_the_simulation_tab(self, tab) -> None:  # type: ignore[no-untyped-def]
        holders = tab.findChildren(QTabWidget)
        labels = {
            holder.tabText(i) for holder in holders for i in range(holder.count())
        }
        assert "Kinetics" in labels

    def test_pendulum_run_populates_plots_and_peak_table(self, pendulum_tab) -> None:  # type: ignore[no-untyped-def]
        panel = pendulum_tab.kinetics_panel()
        table = panel.table()
        assert table.columnCount() == len(PEAK_TABLE_COLUMNS)
        assert table.rowCount() == 3  # shoulder, wrist, clubhead
        assert table.item(0, 0).text() == "shoulder"
        assert table.item(2, 0).text() == "clubhead"
        # Timing cells are % of the downswing.
        assert table.item(0, 2).text().endswith("%")
        assert len(panel._figure.axes) == 3
        titles = [axis.get_title() for axis in panel._figure.axes]
        assert titles == ["Joint Torques", "Joint Power", "Reaction Forces"]
        # Movement-optimizer label conventions (plot_renderer.py).
        assert panel._figure.axes[0].get_ylabel() == "Torque (N·m)"
        assert panel._figure.axes[1].get_ylabel() == "Power (W)"
        assert panel._figure.axes[2].get_ylabel() == "Force (N)"
        assert panel._figure.axes[2].get_xlabel() == "Time (s)"

    def test_manual_run_clears_the_panel(self, tab, qtbot) -> None:  # type: ignore[no-untyped-def]
        with qtbot.waitSignal(tab.runCompleted, timeout=10000):
            tab.run_now()
        panel = tab.kinetics_panel()
        assert panel.table().rowCount() == 0
        assert panel._status.isVisible() or panel._status.text()

    def test_glossary_links_are_forwarded(self, qtbot) -> None:  # type: ignore[no-untyped-def]
        panel = KineticsPanel()
        qtbot.addWidget(panel)
        from PyQt6.QtCore import QUrl

        with qtbot.waitSignal(panel.glossaryRequested, timeout=1000) as blocker:
            panel._on_explanation_link(QUrl("glossary:inverse_dynamics"))
        assert blocker.args == ["inverse_dynamics"]


class TestOverlay:
    def test_checkbox_draws_the_overlay(self, pendulum_tab) -> None:  # type: ignore[no-untyped-def]
        view = pendulum_tab.view()
        run = view.run()
        assert run is not None
        before = len(view._axes.lines)
        view._kinetics_check.setChecked(True)
        view.set_playback_time(run.impact_time_s * 0.5)
        assert len(view._axes.lines) > before
        labels = [line.get_label() for line in view._axes.lines]
        assert any("torque" in str(label) for label in labels)

    def test_overlay_is_inert_for_unsupported_sources(self, tab, qtbot) -> None:  # type: ignore[no-untyped-def]
        with qtbot.waitSignal(tab.runCompleted, timeout=10000):
            tab.run_now()  # manual source
        view = tab.view()
        view._kinetics_check.setChecked(True)
        view.set_playback_time(0.01)
        labels = [str(line.get_label()) for line in view._axes.lines]
        assert not any("torque" in label for label in labels)

    def test_frame_geometry_is_capped_and_labelled(self, pendulum_tab) -> None:  # type: ignore[no-untyped-def]
        series = kinetics_for_run(pendulum_tab.last_run())
        assert series is not None
        frame = overlay_frame(series, series.t.shape[0] // 2)
        assert frame.arcs and frame.arrows
        for label, points in frame.arcs:
            assert "N·m" in label
            radii = np.linalg.norm(points - points.mean(axis=0), axis=1)
            assert radii.max() <= MAX_ARC_RADIUS_M + 1e-9
        for label, _start, vector in frame.arrows:
            assert label.endswith("N")
            assert np.linalg.norm(vector) <= MAX_ARROW_LENGTH_M + 1e-9

    def test_checkbox_carries_hover_guidance(self, tab) -> None:  # type: ignore[no-untyped-def]
        assert tab.view()._kinetics_check.toolTip()
