"""The Plots tab: built-ins, the Custom Plot wizard, and exports.

Replaces (and absorbs) the old Closure Sweep tab (epic #4120 V1). Left:
the managed plot list (add built-in / Custom Plot… / duplicate /
remove) plus export buttons; right: the themed matplotlib canvas with
the standard navigation toolbar. Exports cover the image (PNG / SVG),
the plotted data (CSV / JSON), and the plot definition itself
(save / load ``.json``) so investigations are reproducible.

The tab renders against a reference :class:`SimulationRun`: it adopts
every run completed in the Simulation tab and, until one exists, lazily
builds a manual-source run from the explorer's current scenario.
"""

from __future__ import annotations

import logging

from matplotlib.backends.backend_qtagg import NavigationToolbar2QT
from matplotlib.figure import Figure
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
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
    render_plot,
    spec_from_json,
    spec_to_json,
    write_plot_csv,
    write_plot_json,
)
from rate_of_closure.simulation import SimulationConfig, SimulationRun, run_simulation
from rate_of_closure.ui.pyqt6.figure_canvas import (
    LifecycleSafeFigureCanvas as FigureCanvas,
)

logger = logging.getLogger(__name__)

__all__ = ["PlotsTab"]

#: Fallback club for the lazily built reference run.
_DEFAULT_CLUB = "Driver 10.5°"


class PlotsTab(QWidget):
    """Investigative plotting suite tab (plot list left, canvas right)."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._scenario = ImpactScenario(clubhead_speed_mph=113.0)
        self._run: SimulationRun | None = None
        self._data: PlotData | None = None

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.addWidget(self._build_list_box())
        left_layout.addWidget(self._build_export_box())
        left_layout.addStretch(1)
        left.setMinimumWidth(280)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        self._figure = Figure(figsize=(5.4, 3.6), tight_layout=True)
        self._canvas = FigureCanvas(self._figure)
        self._toolbar = NavigationToolbar2QT(self._canvas, right)
        right_layout.addWidget(self._toolbar)
        right_layout.addWidget(self._canvas, stretch=1)
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

    # ── construction ────────────────────────────────────────────────
    def _build_list_box(self) -> QGroupBox:
        box = QGroupBox("Plots")
        layout = QVBoxLayout(box)

        self._plot_list = QListWidget()
        self._plot_list.setToolTip(
            "Your managed plots. Select one to render it on the canvas; "
            "add built-ins or build your own with Custom Plot…"
        )
        self._plot_list.currentRowChanged.connect(
            lambda _row: self._refresh_if_visible()
        )
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
        buttons: tuple[tuple[str, str, object], ...] = (
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
            (
                "Load Definition…",
                "Load a saved plot definition (.json) into the list.",
                self._load_definition,
            ),
        )
        for index, (text, tip, handler) in enumerate(buttons):
            button = QPushButton(text)
            button.setToolTip(tip)
            button.clicked.connect(handler)  # type: ignore[arg-type]
            grid.addWidget(button, index // 2, index % 2)
        return box

    # ── public API ──────────────────────────────────────────────────
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

    def showEvent(self, event) -> None:  # type: ignore[no-untyped-def]  # noqa: N802
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
        item = QListWidgetItem(label or spec.title or spec.x_key)
        item.setData(Qt.ItemDataRole.UserRole, spec)
        item.setToolTip(
            f"{spec.kind} plot — X: {spec.x_key}; "
            f"Y: {', '.join(spec.y_keys) or '(distribution)'}"
        )
        self._plot_list.addItem(item)
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

    def refresh(self) -> None:
        """Re-render the selected plot against the reference run."""
        spec = self.current_spec()
        if spec is None:
            return
        run = self.reference_run()
        if run is None:
            return
        try:
            self._data = compute_plot_data(spec, run)
            render_plot(self._data, self._figure)
            self._canvas.draw_idle()
            self._status.setText("")
        except Exception as exc:  # noqa: BLE001 — plotting must not crash
            logger.warning("plot render failed: %s", exc)
            self._status.setText(f"Plot failed: {exc}")

    def current_data(self) -> PlotData | None:
        """The data behind the rendered plot (exports read this)."""
        return self._data

    # ── behaviour ───────────────────────────────────────────────────
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

    # ── exports ─────────────────────────────────────────────────────
    def _ready_for_export(self) -> bool:
        if self._data is None:
            self.refresh()
        if self._data is None:
            QMessageBox.information(
                self, "Export", "Nothing to export yet — select a plot first."
            )
            return False
        return True

    def _export_image(self, fmt: str) -> None:
        if not self._ready_for_export():
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            f"Export {fmt.upper()}",
            f"plot.{fmt}",
            f"{fmt.upper()} image (*.{fmt})",
        )
        if path:
            self.save_image(path)

    def save_image(self, path: str) -> None:
        """Save the rendered figure (format inferred from the suffix)."""
        self._figure.savefig(path)

    def _export_csv(self) -> None:
        if not self._ready_for_export():
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Data CSV", "plot_data.csv", "CSV (*.csv)"
        )
        if path:
            assert self._data is not None
            write_plot_csv(self._data, path)

    def _export_json(self) -> None:
        if not self._ready_for_export():
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Data JSON", "plot_data.json", "JSON (*.json)"
        )
        if path:
            assert self._data is not None
            write_plot_json(self._data, path)

    def _save_definition(self) -> None:
        spec = self.current_spec()
        if spec is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Plot Definition", "plot_definition.json", "JSON (*.json)"
        )
        if path:
            spec_to_json(spec, path)

    def _load_definition(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Plot Definition", "", "JSON (*.json)"
        )
        if not path:
            return
        try:
            self.add_spec(spec_from_json(path))
        except Exception as exc:  # noqa: BLE001 — bad files reported nicely
            QMessageBox.warning(self, "Load Plot Definition", str(exc))
