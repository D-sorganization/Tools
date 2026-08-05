"""PyQt6 GUI smoke tests for the Plots tab and Custom Plot wizard.

Headless-safe. Covers: the Plots tab replacing the Closure Sweep tab in
the main window, built-in add/duplicate/remove list management, plot
rendering against the reference run, the 3-step wizard completing into
a valid PlotSpec, export paths producing well-formed files in tmp, and
hover guidance on every new control.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("PyQt6")
pytest.importorskip("pytestqt")

from PyQt6.QtWidgets import QPushButton, QTabWidget  # noqa: E402

from rate_of_closure.club import get_club  # noqa: E402
from rate_of_closure.model import ImpactScenario  # noqa: E402
from rate_of_closure.plotting import (  # noqa: E402
    BUILTIN_PLOTS,
    PlotSpec,
    spec_to_json,
    write_plot_csv,
    write_plot_json,
)
from rate_of_closure.simulation import SimulationConfig, run_simulation  # noqa: E402
from rate_of_closure.ui.pyqt6.main_window import RateOfClosureMainWindow  # noqa: E402
from rate_of_closure.ui.pyqt6.plot_wizard import PlotWizard  # noqa: E402
from rate_of_closure.ui.pyqt6.plots_tab import PlotsTab  # noqa: E402

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


@pytest.fixture(scope="module")
def reference_run():  # type: ignore[no-untyped-def]
    return run_simulation(
        SimulationConfig(
            scenario=ImpactScenario(clubhead_speed_mph=113.0),
            club=get_club("Driver 10.5°"),
        )
    )


@pytest.fixture
def tab(qtbot, reference_run):  # type: ignore[no-untyped-def]
    widget = PlotsTab()
    qtbot.addWidget(widget)
    widget.set_run(reference_run)
    return widget


class TestPlotsTabInWindow:
    def test_main_window_hosts_the_plots_tab_not_the_sweep(self, qtbot) -> None:  # type: ignore[no-untyped-def]
        window = RateOfClosureMainWindow()
        qtbot.addWidget(window)
        try:
            tabs = window.centralWidget().findChildren(QTabWidget)[0]
            labels = [tabs.tabText(i) for i in range(tabs.count())]
            assert "Plots" in labels
            assert "Closure Sweep" not in labels
            assert window.centralWidget().findChildren(PlotsTab)
        finally:
            window._club_view.stop()
            window._simulation_tab.stop()


class TestPlotList:
    def test_starts_with_the_closure_sweep(self, tab) -> None:  # type: ignore[no-untyped-def]
        assert tab._plot_list.count() == 1
        spec = tab.current_spec()
        assert spec is not None and spec.x_key == "input.omega_shaft_dps"

    def test_add_duplicate_remove(self, tab) -> None:  # type: ignore[no-untyped-def]
        index = tab._builtin_combo.findData("swing_time_series")
        tab._builtin_combo.setCurrentIndex(index)
        tab._on_add_builtin()
        assert tab._plot_list.count() == 2
        tab._on_duplicate()
        assert tab._plot_list.count() == 3
        tab._on_remove()
        assert tab._plot_list.count() == 2

    def test_every_builtin_is_offered(self, tab) -> None:  # type: ignore[no-untyped-def]
        offered = {
            tab._builtin_combo.itemData(i) for i in range(tab._builtin_combo.count())
        }
        assert offered == set(BUILTIN_PLOTS)

    def test_selecting_a_series_plot_renders_it(self, tab) -> None:  # type: ignore[no-untyped-def]
        index = tab._builtin_combo.findData("flight_profile_side")
        tab._builtin_combo.setCurrentIndex(index)
        tab._on_add_builtin()
        tab.refresh()
        data = tab.current_data()
        assert data is not None
        # H6 (#4125): flight distances render in yards by default.
        assert data.x_label == "Downrange Distance [yd]"
        assert tab._figure.axes, "figure must carry rendered axes"


class TestWizard:
    def test_wizard_completes_into_a_line_spec(self, qtbot, reference_run) -> None:  # type: ignore[no-untyped-def]
        wizard = PlotWizard(reference_run)
        qtbot.addWidget(wizard)
        wizard.restart()  # initializes page 1
        wizard._scope_page.buttons["swing"].setChecked(True)
        wizard.next()  # variables page
        page = wizard._variables_page
        assert page.x_combo.count() > 0
        assert page.y_keys(), "a default Y variable must be pre-checked"
        wizard.next()  # style page (also refreshes the preview)
        spec = wizard.build_spec()
        assert spec.kind == "line"
        assert spec.x_key.startswith("swing.")
        assert all(key.startswith("swing.") for key in spec.y_keys)
        assert spec.title

    def test_wizard_sweep_scope_produces_a_sweep_spec(
        self, qtbot, reference_run
    ) -> None:  # type: ignore[no-untyped-def]
        wizard = PlotWizard(reference_run)
        qtbot.addWidget(wizard)
        wizard.restart()
        wizard._scope_page.buttons["sweep"].setChecked(True)
        wizard.next()
        page = wizard._variables_page
        assert str(page.x_combo.currentData()).startswith("input.")
        page.count_spin.setValue(4)
        wizard.next()
        spec = wizard.build_spec()
        assert spec.kind == "sweep"
        assert spec.x_start is not None and spec.x_stop is not None
        assert spec.x_count == 4

    def test_wizard_histogram_scope(self, qtbot, reference_run) -> None:  # type: ignore[no-untyped-def]
        wizard = PlotWizard(reference_run)
        qtbot.addWidget(wizard)
        wizard.restart()
        wizard._scope_page.buttons["histogram"].setChecked(True)
        wizard.next()
        wizard.next()
        spec = wizard.build_spec()
        assert spec.kind == "histogram"
        assert spec.y_keys == ()


class TestExports:
    def test_png_and_svg_export_produce_files(self, tab, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
        index = tab._builtin_combo.findData("flight_profile_top")
        tab._builtin_combo.setCurrentIndex(index)
        tab._on_add_builtin()
        tab.refresh()
        png = tmp_path / "plot.png"
        svg = tmp_path / "plot.svg"
        tab.save_image(str(png))
        tab.save_image(str(svg))
        assert png.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
        assert b"<svg" in svg.read_bytes()

    def test_data_and_definition_exports_round_trip(self, tab, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
        index = tab._builtin_combo.findData("swing_time_series")
        tab._builtin_combo.setCurrentIndex(index)
        tab._on_add_builtin()
        tab.refresh()
        data = tab.current_data()
        spec = tab.current_spec()
        assert data is not None and spec is not None
        csv_path = tmp_path / "data.csv"
        json_path = tmp_path / "data.json"
        def_path = tmp_path / "definition.json"
        write_plot_csv(data, csv_path)
        write_plot_json(data, json_path)
        spec_to_json(spec, def_path)
        assert csv_path.read_text(encoding="utf-8").splitlines()[0]
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        assert payload["format"] == "rate_of_closure.plot_data/1"
        assert (
            PlotSpec.from_json_dict(json.loads(def_path.read_text(encoding="utf-8")))
            == spec
        )


class TestGuidance:
    def test_every_new_control_has_a_tooltip(self, tab, qtbot, reference_run) -> None:  # type: ignore[no-untyped-def]
        assert tab._plot_list.toolTip()
        assert tab._builtin_combo.toolTip()
        for button in tab.findChildren(QPushButton):
            assert button.toolTip(), button.text()
        wizard = PlotWizard(reference_run)
        qtbot.addWidget(wizard)
        wizard.restart()
        for button in wizard._scope_page.buttons.values():
            assert button.toolTip(), button.text()
        page = wizard._variables_page
        for widget in (
            page.x_combo,
            page.y_list,
            page.start_spin,
            page.stop_spin,
            page.count_spin,
        ):
            assert widget.toolTip()
        style = wizard._style_page
        for widget in (
            style.title_edit,
            style.kind_combo,
            style.x_log,
            style.y_log,
        ):
            assert widget.toolTip()
