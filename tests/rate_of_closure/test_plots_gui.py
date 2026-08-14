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

from matplotlib.backend_bases import MouseEvent  # noqa: E402
from PyQt6.QtCore import Qt  # noqa: E402
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

    def test_managed_plots_have_independent_visible_canvases(self, tab) -> None:  # type: ignore[no-untyped-def]
        first_canvas = tab._canvas
        index = tab._builtin_combo.findData("swing_time_series")
        tab._builtin_combo.setCurrentIndex(index)
        tab._on_add_builtin()

        assert len(tab.plot_panes()) == 2
        assert len({id(pane.canvas()) for pane in tab.plot_panes()}) == 2
        assert first_canvas in {pane.canvas() for pane in tab.plot_panes()}

    def test_each_plot_has_autofit_zoom_and_independent_legend(self, tab) -> None:  # type: ignore[no-untyped-def]
        tab._on_duplicate()
        tab.refresh()
        first, second = tab.plot_panes()

        first.zoom_in()
        assert first.zoom_percent() == 125
        assert second.zoom_percent() == 100
        first.set_legend_placement("hidden")
        assert first.legend_placement() == "hidden"
        assert second.legend_placement() == "outside_right"
        first.auto_fit()
        assert first.zoom_percent() == 100

    def test_plot_grid_collapses_to_one_column_when_viewport_is_narrow(
        self, tab, qtbot
    ) -> None:  # type: ignore[no-untyped-def]
        tab._on_duplicate()
        tab.resize(1000, 700)
        tab.show()
        qtbot.waitUntil(lambda: tab._plot_scroll.viewport().width() < 800)
        tab._reflow_panes()

        second_index = tab._plot_grid.indexOf(tab.plot_panes()[1])
        row, column, _row_span, _column_span = tab._plot_grid.getItemPosition(
            second_index
        )
        assert (row, column) == (1, 0)

    def test_long_canvas_titles_wrap_in_the_plot_viewport(self, tab) -> None:  # type: ignore[no-untyped-def]
        tab.refresh()
        title = tab.plot_panes()[0].figure().axes[0].get_title()
        assert "\n" in title

    def test_selection_uses_cached_data_and_add_computes_only_the_new_plot(
        self, tab, qtbot, monkeypatch
    ) -> None:  # type: ignore[no-untyped-def]
        import rate_of_closure.ui.pyqt6.plots_tab as plots_module

        calls = 0
        original = plots_module.compute_plot_data

        def counted(spec, run, should_cancel=None):  # type: ignore[no-untyped-def]
            nonlocal calls
            calls += 1
            return original(spec, run, should_cancel)

        monkeypatch.setattr(plots_module, "compute_plot_data", counted)
        tab.show()
        qtbot.waitUntil(lambda: tab.current_data() is not None, timeout=15_000)
        assert calls == 1
        index = tab._builtin_combo.findData("swing_time_series")
        tab._builtin_combo.setCurrentIndex(index)
        tab._on_add_builtin()
        qtbot.waitUntil(lambda: calls == 2, timeout=15_000)
        qtbot.waitUntil(lambda: tab.current_data() is not None, timeout=15_000)
        assert calls == 2
        tab._plot_list.setCurrentRow(0)
        assert calls == 2
        qtbot.keyClick(tab.plot_panes()[0].canvas(), Qt.Key.Key_Home)
        assert calls == 2

    def test_keyboard_and_pointer_select_exact_series_evidence(
        self, tab, qtbot
    ) -> None:  # type: ignore[no-untyped-def]
        tab.show()
        tab.refresh()
        pane = tab.plot_panes()[0]
        tab.activateWindow()
        pane.canvas().setFocus()
        qtbot.waitUntil(pane.canvas().hasFocus)
        qtbot.keyClick(pane.canvas(), Qt.Key.Key_Home)
        assert pane.selected_evidence() is not None
        assert "source point 1/" in pane.inspection_status()
        assert pane.canvas().hasFocus()

        plan = pane._inspection_plan
        assert plan is not None and plan.kind == "series"
        axes = pane.figure().axes[0]
        x_pixel, y_pixel = axes.transData.transform(
            (plan.x[-1], plan.series[0].values[-1])
        )
        event = MouseEvent(
            "button_press_event", pane.canvas(), x_pixel, y_pixel, button=1
        )
        pane._on_inspection_click(event)
        selected = pane.selected_evidence()
        assert selected is not None and selected.raw_index == plan.raw_count - 1

    def test_new_plot_data_clears_selection_and_histogram_bins_are_derived(
        self, tab, qtbot
    ) -> None:  # type: ignore[no-untyped-def]
        tab.show()
        tab.refresh()
        pane = tab.plot_panes()[0]
        qtbot.keyClick(pane.canvas(), Qt.Key.Key_Home)
        assert pane.selected_evidence() is not None
        tab.set_run(tab.reference_run())
        qtbot.waitUntil(lambda: pane.selected_evidence() is None, timeout=15_000)

        tab.add_spec(
            PlotSpec(kind="histogram", x_key="flight.speed_mps", title="Speed")
        )
        tab.refresh()
        histogram = tab.plot_panes()[-1]
        qtbot.keyClick(histogram.canvas(), Qt.Key.Key_Home)
        assert "Histogram bin 1/" in histogram.inspection_status()

    def test_failed_recompute_retains_prior_data_selection_and_figure(
        self, tab, qtbot, monkeypatch
    ) -> None:  # type: ignore[no-untyped-def]
        import rate_of_closure.ui.pyqt6.plots_tab as plots_module

        tab.show()
        tab.refresh()
        pane = tab.plot_panes()[0]
        qtbot.keyClick(pane.canvas(), Qt.Key.Key_Home)
        prior_data = tab.current_data()
        prior_figure = pane.figure()
        prior_selection = pane.selected_evidence()

        def fail(_spec, _run, _should_cancel=None):  # type: ignore[no-untyped-def]
            raise RuntimeError("planted plot authority failure")

        monkeypatch.setattr(plots_module, "compute_plot_data", fail)
        tab.set_run(tab.reference_run())
        qtbot.waitUntil(
            lambda: "prior accepted plot retained" in tab._status.text(),
            timeout=15_000,
        )
        assert tab.current_data() is prior_data
        assert pane.figure() is prior_figure
        assert pane.selected_evidence() == prior_selection
        assert "prior accepted plot retained" in tab._status.text()

    def test_ninth_plot_is_rejected_before_computation(self, tab, monkeypatch) -> None:  # type: ignore[no-untyped-def]
        import rate_of_closure.ui.pyqt6.plots_tab as plots_module

        calls = 0

        def counted(_spec, _run):  # type: ignore[no-untyped-def]
            nonlocal calls
            calls += 1
            return None

        monkeypatch.setattr(plots_module, "compute_plot_data", counted)
        index = tab._builtin_combo.findData("swing_time_series")
        tab._builtin_combo.setCurrentIndex(index)
        for _ in range(8):
            tab._on_add_builtin()

        assert tab._plot_list.count() == 8
        assert "at most 8 managed plots" in tab._status.text()
        assert calls == 0


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
