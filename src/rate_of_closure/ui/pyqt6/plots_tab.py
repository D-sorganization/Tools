"""Managed built-in and custom plots with reproducible export support."""

from __future__ import annotations

import logging
from collections.abc import Callable

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QResizeEvent, QShowEvent
from PyQt6.QtWidgets import (
    QComboBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from rate_of_closure.club import get_club
from rate_of_closure.model import ImpactScenario
from rate_of_closure.plotting import (
    BUILTIN_PLOTS,
    PlotData,
    PlotSpec,
    builtin_spec,
    compute_plot_data,
)
from rate_of_closure.simulation import SimulationConfig, SimulationRun, run_simulation
from rate_of_closure.ui.pyqt6.plot_canvas_pane import PlotCanvasPane
from rate_of_closure.ui.pyqt6.plot_export_mixin import PlotExportMixin

logger = logging.getLogger(__name__)

__all__ = ["PlotsTab"]

_DEFAULT_CLUB = "Driver 10.5°"
_TWO_COLUMN_VIEWPORT_PX = 800


class PlotsTab(PlotExportMixin, QWidget):
    """Investigative plotting suite tab (plot list left, canvas right)."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._scenario = ImpactScenario(clubhead_speed_mph=113.0)
        self._run: SimulationRun | None = None
        self._data: PlotData | None = None
        self._plot_panes: list[PlotCanvasPane] = []

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.addWidget(self._build_list_box())
        left_layout.addWidget(self._build_export_box())
        left_layout.addStretch(1)
        left.setMinimumWidth(280)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        self._plot_workspace = QWidget()
        self._plot_grid = QGridLayout(self._plot_workspace)
        self._plot_grid.setContentsMargins(4, 4, 4, 4)
        self._plot_grid.setSpacing(8)
        self._plot_scroll = QScrollArea()
        self._plot_scroll.setWidgetResizable(True)
        self._plot_scroll.setWidget(self._plot_workspace)
        right_layout.addWidget(self._plot_scroll, stretch=1)
        self._status = QLabel("")
        self._status.setWordWrap(True)
        right_layout.addWidget(self._status)

        splitter = QSplitter()
        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(splitter)

        self._add_builtin("closure_sweep")
        self._plot_list.setCurrentRow(0)

    def _build_list_box(self) -> QGroupBox:
        box = QGroupBox("Plots")
        layout = QVBoxLayout(box)

        self._plot_list = QListWidget()
        self._plot_list.setToolTip(
            "Your managed plots. Select one to render it on the canvas; "
            "add built-ins or build your own with Custom Plot…"
        )
        self._plot_list.currentRowChanged.connect(self._on_selection_changed)
        layout.addWidget(self._plot_list)

        row = QHBoxLayout()
        self._builtin_combo = QComboBox()
        for name, (label, _factory) in BUILTIN_PLOTS.items():
            self._builtin_combo.addItem(label, userData=name)
        self._builtin_combo.setToolTip(
            "Built-in advanced plots: sweeps, time series, and flight "
            "profiles, each rendered through the same pipeline."
        )
        row.addWidget(self._builtin_combo, stretch=1)
        add_button = QPushButton("Add")
        add_button.setToolTip("Add the selected built-in plot to the list.")
        add_button.clicked.connect(self._on_add_builtin)
        row.addWidget(add_button)
        layout.addLayout(row)

        grid = QGridLayout()
        self._custom_button = QPushButton("Custom Plot…")
        self._custom_button.setToolTip(
            "Open the 3-step wizard: pick a data-source scope, choose X "
            "and Y variables from the catalog, then style with a live "
            "preview."
        )
        self._custom_button.clicked.connect(self._on_custom_plot)
        grid.addWidget(self._custom_button, 0, 0, 1, 2)
        duplicate_button = QPushButton("Duplicate")
        duplicate_button.setToolTip("Copy the selected plot definition.")
        duplicate_button.clicked.connect(self._on_duplicate)
        grid.addWidget(duplicate_button, 1, 0)
        remove_button = QPushButton("Remove")
        remove_button.setToolTip("Delete the selected plot from the list.")
        remove_button.clicked.connect(self._on_remove)
        grid.addWidget(remove_button, 1, 1)
        layout.addLayout(grid)
        return box

    def _build_export_box(self) -> QGroupBox:
        box = QGroupBox("Export")
        grid = QGridLayout(box)
        buttons: tuple[tuple[str, str, Callable[[], None]], ...] = (
            (
                "PNG…",
                "Save the rendered figure as a PNG image.",
                lambda: self._export_image("png"),
            ),
            (
                "SVG…",
                "Save the rendered figure as a scalable SVG image.",
                lambda: self._export_image("svg"),
            ),
            ("Data CSV…", "Save the plotted numbers as CSV.", self._export_csv),
            (
                "Data JSON…",
                "Save the plotted numbers plus the plot definition as JSON.",
                self._export_json,
            ),
            (
                "Save Definition…",
                "Save this plot's definition (.json) so "
                "the investigation can be reloaded and reproduced.",
                self._save_definition,
            ),
            ("Load Definition…", "Load a saved plot.", self._load_definition),
        )
        for index, (text, tip, handler) in enumerate(buttons):
            button = QPushButton(text)
            button.setToolTip(tip)
            button.clicked.connect(handler)
            grid.addWidget(button, index // 2, index % 2)
        return box

    def set_scenario(self, scenario: ImpactScenario) -> None:
        """Adopt the explorer's scenario (rebuilds the lazy run).

        Rendering is deferred while the tab is hidden — sweeps re-run
        the simulation per grid point, far too heavy for every explorer
        keystroke.
        """
        self._scenario = scenario
        self._run = None
        self._data = None
        self._refresh_if_visible()

    def set_run(self, run: SimulationRun) -> None:
        """Adopt a completed simulation run as the reference run."""
        self._run = run
        self._data = None
        self._refresh_if_visible()

    def showEvent(self, event: QShowEvent | None) -> None:  # noqa: N802
        """Render any deferred scenario/run change on first show."""
        super().showEvent(event)
        if self._data is None:
            self.refresh()

    def _refresh_if_visible(self) -> None:
        """Deferred refresh: sweeps re-run the simulation per grid point,
        far too heavy while the tab is hidden; showEvent catches up."""
        self._data = None
        if self.isVisible():
            self.refresh()

    def add_spec(self, spec: PlotSpec, label: str | None = None) -> None:
        """Append a plot definition to the managed list and select it."""
        display_label = label or spec.title or spec.x_key
        item = QListWidgetItem(display_label)
        item.setData(Qt.ItemDataRole.UserRole, spec)
        item.setToolTip(
            f"{spec.kind} plot — X: {spec.x_key}; "
            f"Y: {', '.join(spec.y_keys) or '(distribution)'}"
        )
        self._plot_list.addItem(item)
        pane = PlotCanvasPane(display_label)
        self._plot_panes.append(pane)
        self._reflow_panes()
        self._plot_list.setCurrentItem(item)

    def current_spec(self) -> PlotSpec | None:
        """The selected plot definition, if any."""
        item = self._plot_list.currentItem()
        return item.data(Qt.ItemDataRole.UserRole) if item is not None else None

    def reference_run(self) -> SimulationRun | None:
        """The reference run, building the lazy manual-source run once."""
        if self._run is None:
            try:
                self._run = run_simulation(
                    SimulationConfig(
                        scenario=self._scenario, club=get_club(_DEFAULT_CLUB)
                    )
                )
            except Exception as exc:  # noqa: BLE001 — surfaced in status
                logger.warning("reference run failed: %s", exc)
                self._status.setText(f"Reference run failed: {exc}")
        return self._run

    def plot_panes(self) -> tuple[PlotCanvasPane, ...]:
        """Return every independently controlled visible plot viewport."""
        return tuple(self._plot_panes)

    def refresh(self) -> None:
        """Re-render the selected plot against the reference run."""
        run = self.reference_run()
        if run is None:
            return
        errors: list[str] = []
        self._data = None
        current_row = self._plot_list.currentRow()
        for row, pane in enumerate(self._plot_panes):
            item = self._plot_list.item(row)
            if item is None:
                continue
            spec = item.data(Qt.ItemDataRole.UserRole)
            try:
                data = compute_plot_data(spec, run)
                pane.render_data(data)
                if row == current_row:
                    self._data = data
            except Exception as exc:  # noqa: BLE001 — plotting must not crash
                logger.warning("plot render failed: %s", exc)
                errors.append(f"{item.text()}: {exc}")
        self._sync_selected_pane()
        self._status.setText("; ".join(errors))

    def current_data(self) -> PlotData | None:
        """The data behind the rendered plot (exports read this)."""
        return self._data

    def _add_builtin(self, name: str) -> None:
        label, _factory = BUILTIN_PLOTS[name]
        self.add_spec(builtin_spec(name, self._run), label)

    def _on_add_builtin(self) -> None:
        self._add_builtin(str(self._builtin_combo.currentData()))

    def _on_custom_plot(self) -> None:
        from rate_of_closure.ui.pyqt6.plot_wizard import PlotWizard

        wizard = PlotWizard(self.reference_run(), self)
        if not wizard.exec():
            return
        try:
            spec = wizard.build_spec()
        except Exception as exc:  # noqa: BLE001 — DbC message to the user
            QMessageBox.warning(self, "Custom Plot", str(exc))
            return
        self.add_spec(spec)

    def _on_duplicate(self) -> None:
        item = self._plot_list.currentItem()
        spec = self.current_spec()
        if item is None or spec is None:
            return
        self.add_spec(spec, f"{item.text()} (Copy)")

    def _on_remove(self) -> None:
        row = self._plot_list.currentRow()
        if row >= 0:
            self._plot_list.takeItem(row)
            pane = self._plot_panes.pop(row)
            pane.setParent(None)
            pane.deleteLater()
            self._reflow_panes()
            self._sync_selected_pane()

    def _on_selection_changed(self, _row: int) -> None:
        self._sync_selected_pane()
        self._refresh_if_visible()

    def _sync_selected_pane(self) -> None:
        row = self._plot_list.currentRow()
        if not 0 <= row < len(self._plot_panes):
            self._data = None
            return
        pane = self._plot_panes[row]
        self._figure = pane.figure()
        self._canvas = pane.canvas()
        self._toolbar = pane.toolbar()

    def _reflow_panes(self) -> None:
        column_count = self._plot_column_count()
        while self._plot_grid.count():
            item = self._plot_grid.takeAt(0)
            widget = item.widget() if item is not None else None
            if widget is not None:
                widget.setParent(self._plot_workspace)
        for index, pane in enumerate(self._plot_panes):
            self._plot_grid.addWidget(
                pane,
                index // column_count,
                index % column_count,
            )

    def _plot_column_count(self) -> int:
        """Return a readable responsive column count for the plot viewport."""
        viewport = self._plot_scroll.viewport()
        width = viewport.width() if viewport is not None else 0
        return 2 if width >= _TWO_COLUMN_VIEWPORT_PX else 1

    def resizeEvent(self, event: QResizeEvent | None) -> None:  # noqa: N802
        """Reflow plot panes when the available viewport changes."""
        super().resizeEvent(event)
        self._reflow_panes()
