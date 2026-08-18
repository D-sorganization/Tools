"""Custom Plot wizard: scope -> variables -> style, with live preview.

Three guided steps (epic #4120 V1):

1. **Scope** — what to plot over: the swing time series, the flight
   trajectory, a sweep of one input across re-run simulations, or a
   histogram of a sampled variable.
2. **Variables** — X and Y (multi-select) chosen from the data catalog,
   grouped by category and filtered to what the scope supports; sweep
   scope adds the swept range (start / stop / points).
3. **Style** — title, kind (line / scatter), log-axis flags, and a live
   preview rendered through the same pipeline as the final plot.

The wizard's product is a validated
:class:`~rate_of_closure.plotting.spec.PlotSpec`.
"""

from __future__ import annotations

import dataclasses
import logging

from matplotlib.figure import Figure
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QAbstractSpinBox,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QRadioButton,
    QSpinBox,
    QVBoxLayout,
    QWizard,
    QWizardPage,
)

from rate_of_closure.plotting.catalog import CATALOG, variables_by_category
from rate_of_closure.plotting.render import compute_plot_data, render_plot
from rate_of_closure.plotting.spec import PlotSpec
from rate_of_closure.simulation.session import SimulationRun
from rate_of_closure.ui.pyqt6.figure_canvas import (
    LifecycleSafeFigureCanvas as FigureCanvas,
)

logger = logging.getLogger(__name__)

__all__ = ["SCOPES", "PlotWizard"]

#: scope id -> (Title Case label, tooltip). Order is presentation order.
SCOPES: dict[str, tuple[str, str]] = {
    "swing": (
        "Swing Time Series",
        "Plot per-sample swing variables (clubhead position, speed, "
        "angular speed) against each other over the sampled swing.",
    ),
    "flight": (
        "Flight Trajectory",
        "Plot per-sample ball-flight variables (position, height, "
        "lateral, speed) against each other over the integrated flight.",
    ),
    "sweep": (
        "Sweep an Input",
        "Vary one input variable across a range, re-running the full "
        "swing → impact → flight simulation at every grid point, and "
        "plot scalar outputs (Impact / Launch / Metric) against it.",
    ),
    "histogram": (
        "Histogram",
        "Bin the distribution of one per-sample variable (swing or "
        "flight series) into a histogram.",
    ),
}

#: scope id -> categories offered for X.
_X_CATEGORIES: dict[str, tuple[str, ...]] = {
    "swing": ("Swing Sample",),
    "flight": ("Flight",),
    "sweep": ("Input",),
    "histogram": ("Swing Sample", "Kinetics", "Flight"),
}

#: scope id -> categories offered for Y (empty = no Y step content).
#: Kinetics series share the swing sample grid (#4125 H2), so they are
#: offered as Y variables of the swing scope.
_Y_CATEGORIES: dict[str, tuple[str, ...]] = {
    "swing": ("Swing Sample", "Kinetics"),
    "flight": ("Flight",),
    "sweep": ("Impact", "Launch", "Metric"),
    "histogram": (),
}

#: Input keys that can be swept, with their default ranges.
_SWEEP_DEFAULTS: dict[str, tuple[float, float]] = {
    "input.clubhead_speed_mph": (80.0, 130.0),
    "input.omega_plane_dps": (1000.0, 3000.0),
    "input.omega_shaft_dps": (0.0, 4000.0),
    "input.lie_angle_deg": (45.0, 90.0),
    "input.com_to_face_mm": (25.0, 50.0),
    "input.impact_offset_toe_mm": (-20.0, 20.0),
    "input.impact_offset_high_mm": (-10.0, 10.0),
    "input.contact_duration_us": (300.0, 600.0),
    "input.plane_yaw_deg": (-20.0, 20.0),
    "input.plane_side_tilt_deg": (-80.0, -10.0),
    "input.plane_forward_tilt_deg": (-20.0, 20.0),
    "input.impact_time_s": (0.006, 0.054),
}


class _ScopePage(QWizardPage):
    """Step 1: pick the data-source scope."""

    def __init__(self) -> None:
        super().__init__()
        self.setTitle("Data Source")
        self.setSubTitle("What should this plot draw its data from?")
        layout = QVBoxLayout(self)
        self.buttons: dict[str, QRadioButton] = {}
        for scope, (label, tip) in SCOPES.items():
            button = QRadioButton(label)
            button.setToolTip(tip)
            self.buttons[scope] = button
            layout.addWidget(button)
        self.buttons["swing"].setChecked(True)

    def scope(self) -> str:
        """The selected scope id."""
        return next(s for s, b in self.buttons.items() if b.isChecked())


class _VariablesPage(QWizardPage):
    """Step 2: pick X and Y variables from the catalog."""

    def __init__(self, scope_page: _ScopePage) -> None:
        super().__init__()
        self._scope_page = scope_page
        self.setTitle("Variables")
        self.setSubTitle(
            "Pick the X variable and one or more Y variables, grouped by "
            "catalog category."
        )
        form = QFormLayout(self)
        self.x_combo = QComboBox()
        self.x_combo.setToolTip(
            "The horizontal-axis variable. Sweeps vary this input across "
            "the range below; series scopes read it off the run."
        )
        self.x_combo.currentIndexChanged.connect(self._on_x_changed)
        form.addRow("X Variable", self.x_combo)

        self.y_list = QListWidget()
        self.y_list.setToolTip(
            "Check one or more variables to draw against X; each becomes "
            "its own themed series with a legend entry."
        )
        self.y_list.setMinimumHeight(160)
        form.addRow("Y Variables", self.y_list)

        self.start_spin = QDoubleSpinBox()
        self.stop_spin = QDoubleSpinBox()
        self.count_spin = QSpinBox()
        for spin in (self.start_spin, self.stop_spin):
            spin.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
            spin.setDecimals(4)
            spin.setRange(-1e6, 1e6)
        self.count_spin.setRange(2, 501)
        self.count_spin.setValue(25)
        self.start_spin.setToolTip("First value of the swept input range.")
        self.stop_spin.setToolTip("Last value of the swept input range.")
        self.count_spin.setToolTip(
            "Number of sweep grid points; each point re-runs the full "
            "simulation, so more points take longer."
        )
        self._range_rows: list[tuple[QLabel, object]] = []
        for text, widget in (
            ("Sweep Start", self.start_spin),
            ("Sweep Stop", self.stop_spin),
            ("Sweep Points", self.count_spin),
        ):
            label = QLabel(text)
            form.addRow(label, widget)
            self._range_rows.append((label, widget))

    def initializePage(self) -> None:  # noqa: N802 — Qt override
        scope = self._scope_page.scope()
        self.x_combo.blockSignals(True)
        self.x_combo.clear()
        for category in _X_CATEGORIES[scope]:
            for spec in variables_by_category(category):
                self.x_combo.addItem(
                    f"{category} — {spec.axis_label}", userData=spec.key
                )
        self.x_combo.blockSignals(False)
        self.y_list.clear()
        for category in _Y_CATEGORIES[scope]:
            for spec in variables_by_category(category):
                item = QListWidgetItem(f"{category} — {spec.axis_label}")
                item.setData(Qt.ItemDataRole.UserRole, spec.key)
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                item.setCheckState(Qt.CheckState.Unchecked)
                self.y_list.addItem(item)
        first = self.y_list.item(0)
        if first is not None:
            first.setCheckState(Qt.CheckState.Checked)
        show_range = scope == "sweep"
        for label, widget in self._range_rows:
            label.setVisible(show_range)
            widget.setVisible(show_range)  # type: ignore[attr-defined]
        self.y_list.setVisible(scope != "histogram")
        self._on_x_changed()

    def _on_x_changed(self, *_args: object) -> None:
        key = self.x_key()
        if key in _SWEEP_DEFAULTS:
            start, stop = _SWEEP_DEFAULTS[key]
            self.start_spin.setValue(start)
            self.stop_spin.setValue(stop)

    def x_key(self) -> str:
        """The selected X catalog key."""
        return str(self.x_combo.currentData())

    def y_keys(self) -> tuple[str, ...]:
        """The checked Y catalog keys."""
        return tuple(
            str(item.data(Qt.ItemDataRole.UserRole))
            for item in (self.y_list.item(i) for i in range(self.y_list.count()))
            if item is not None and item.checkState() == Qt.CheckState.Checked
        )


class _StylePage(QWizardPage):
    """Step 3: title / kind / log flags with a live preview."""

    def __init__(self, wizard: PlotWizard) -> None:
        super().__init__()
        self._wizard = wizard
        self.setTitle("Style and Preview")
        self.setSubTitle("Name the plot, pick its kind, and preview it live.")
        layout = QVBoxLayout(self)
        form = QFormLayout()
        self.title_edit = QLineEdit()
        self.title_edit.setToolTip("Plot title shown above the axes.")
        self.title_edit.textChanged.connect(self._refresh_preview)
        form.addRow("Title", self.title_edit)
        self.kind_combo = QComboBox()
        self.kind_combo.setToolTip(
            "Line joins the samples in order; Scatter draws one marker "
            "per sample. Sweeps and histograms fix their own kind."
        )
        self.kind_combo.currentIndexChanged.connect(self._refresh_preview)
        form.addRow("Kind", self.kind_combo)
        self.x_log = QCheckBox("Log X Axis")
        self.x_log.setToolTip("Use a logarithmic horizontal axis.")
        self.y_log = QCheckBox("Log Y Axis")
        self.y_log.setToolTip("Use a logarithmic vertical axis.")
        self.x_log.toggled.connect(self._refresh_preview)
        self.y_log.toggled.connect(self._refresh_preview)
        row = QHBoxLayout()
        row.addWidget(self.x_log)
        row.addWidget(self.y_log)
        form.addRow("Axes", row)
        layout.addLayout(form)
        self._figure = Figure(figsize=(4.2, 2.6), tight_layout=True)
        self._canvas = FigureCanvas(self._figure)
        self._canvas.setToolTip("Live preview rendered through the plot pipeline.")
        layout.addWidget(self._canvas, stretch=1)
        self._status = QLabel("")
        self._status.setWordWrap(True)
        layout.addWidget(self._status)

    def initializePage(self) -> None:  # noqa: N802 — Qt override
        scope = self._wizard.scope()
        self.kind_combo.blockSignals(True)
        self.kind_combo.clear()
        if scope == "sweep":
            self.kind_combo.addItem("Sweep", userData="sweep")
        elif scope == "histogram":
            self.kind_combo.addItem("Histogram", userData="histogram")
        else:
            self.kind_combo.addItem("Line", userData="line")
            self.kind_combo.addItem("Scatter", userData="scatter")
        self.kind_combo.blockSignals(False)
        if not self.title_edit.text():
            self.title_edit.setText(
                f"Custom Plot — {CATALOG[self._wizard.x_key()].label}"
            )
        self._refresh_preview()

    def _refresh_preview(self, *_args: object) -> None:
        run = self._wizard.reference_run
        try:
            spec = self._wizard.build_spec()
        except Exception as exc:  # noqa: BLE001 — DbC message to the user
            self._status.setText(f"Incomplete definition: {exc}")
            return
        if run is None:
            self._status.setText("No reference run available for the preview.")
            return
        try:
            preview = spec
            if spec.kind == "sweep" and spec.x_count > 7:
                preview = dataclasses.replace(spec, x_count=7)
            render_plot(compute_plot_data(preview, run), self._figure)
            self._canvas.draw_idle()
            self._status.setText(
                "Preview uses a coarse grid for sweeps; the final plot "
                "renders the full range."
            )
        except Exception as exc:  # noqa: BLE001 — preview must never crash
            logger.debug("preview failed: %s", exc)
            self._status.setText(f"Preview unavailable: {exc}")


class PlotWizard(QWizard):
    """The 3-step Custom Plot wizard producing a validated PlotSpec."""

    def __init__(self, reference_run: SimulationRun | None, parent=None) -> None:  # type: ignore[no-untyped-def]
        super().__init__(parent)
        self.setWindowTitle("Custom Plot")
        self.reference_run = reference_run
        self._scope_page = _ScopePage()
        self._variables_page = _VariablesPage(self._scope_page)
        self._style_page = _StylePage(self)
        self.addPage(self._scope_page)
        self.addPage(self._variables_page)
        self.addPage(self._style_page)

    def scope(self) -> str:
        """The selected scope id."""
        return self._scope_page.scope()

    def x_key(self) -> str:
        """The selected X catalog key."""
        return self._variables_page.x_key()

    def build_spec(self) -> PlotSpec:
        """The PlotSpec described by the current wizard state.

        Raises:
            PreconditionError: If the selection is incomplete/invalid.
        """
        scope = self.scope()
        page = self._variables_page
        kind = str(self._style_page.kind_combo.currentData() or "line")
        x_key = page.x_key()
        title = self._style_page.title_edit.text()
        x_log = self._style_page.x_log.isChecked()
        y_log = self._style_page.y_log.isChecked()
        if scope == "histogram":
            return PlotSpec(
                kind="histogram",
                x_key=x_key,
                title=title,
                x_log=x_log,
                y_log=y_log,
            )
        if scope == "sweep":
            return PlotSpec(
                kind="sweep",
                x_key=x_key,
                y_keys=page.y_keys(),
                title=title,
                x_log=x_log,
                y_log=y_log,
                x_start=page.start_spin.value(),
                x_stop=page.stop_spin.value(),
                x_count=page.count_spin.value(),
            )
        return PlotSpec(
            kind=kind,
            x_key=x_key,
            y_keys=page.y_keys(),
            title=title,
            x_log=x_log,
            y_log=y_log,
        )
