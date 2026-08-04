"""Rate of Closure Impact Explorer — PyQt6 main window.

Layout: controls on the left (scenario inputs + presets), clickable
results and an explanation panel below them, and a tab stack on the
right: the animated 3D clubhead (with playback speed and fixed/moving
display modes), the closure sweep, and the Derivation & Traceability
tab that typesets the whole calculation with live numbers.

The window consumes complete :class:`~rate_of_closure.model.ImpactScenario`
objects from the controls panel and hands them to the model and views —
it never reaches into widgets or model internals (LoD).
"""

from __future__ import annotations

import logging
import math

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QScrollArea,
    QSplitter,
    QStatusBar,
    QTabWidget,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from rate_of_closure.derivation import (
    METRIC_EXPLANATIONS,
    RESULT_EXPLANATIONS,
)
from rate_of_closure.model import ImpactScenario, closure_metrics, solve
from rate_of_closure.ui.pyqt6.club_view import Club3DView, SweepView
from rate_of_closure.ui.pyqt6.controls_panel import ControlsPanel
from rate_of_closure.ui.pyqt6.derivation_view import DerivationView
from rate_of_closure.units import convert_from_canonical

logger = logging.getLogger(__name__)

__all__ = ["RateOfClosureMainWindow"]

# ── Theme integration (optional — graceful fallback) ───────────────
try:
    from shared.python.theme.integration import ThemedWindowMixin

    _THEME_AVAILABLE = True
except ImportError:  # standalone / vendored use
    _THEME_AVAILABLE = False

    class ThemedWindowMixin:  # type: ignore[no-redef]
        """No-op stand-in when the shared theme package is unavailable."""

        def setup_theme_support(self, settings_app: str = "") -> None:
            """Match the themed mixin's interface; do nothing."""


#: (result field, Title Case label) in display order. Every field must
#: have an entry in RESULT_EXPLANATIONS — enforced by the test suite.
_RESULT_ROWS: tuple[tuple[str, str], ...] = (
    ("path_deviation_deg", "Impact-Point Path vs Reference"),
    ("aoa_deviation_deg", "Attack-Angle Change"),
    ("tangential_speed_mph", "Rotation-Induced Velocity"),
    ("speed_delta_mph", "Delivered Speed Change"),
    ("closure_rate_dps", "Closure Rate (CCV)"),
    ("normalized_closure_deg_per_ft", "Normalized Closure"),
    ("closure_during_contact_deg", "Face Closure During Contact"),
    ("loft_gain_during_contact_deg", "Dynamic Loft Gained During Contact"),
)

#: (metric field, Title Case label) for the Common Closure Metrics box.
_METRIC_ROWS: tuple[tuple[str, str], ...] = (
    ("ccv_dps", "Club Closure Velocity (CCV)"),
    ("closure_deg_per_ft", "Closure per Foot of Travel"),
    ("closure_deg_per_inch", "Closure per Inch of Travel"),
    ("closure_deg_per_ms", "Closure per Millisecond"),
    ("r_isa_ft", "Distance to Screw Axis (R_ISA)"),
    ("r_isa_m", "Distance to Screw Axis (Metric)"),
    ("time_to_square_from_1deg_open_ms", "Time to Square From 1° Open"),
    ("toe_heel_speed_delta_mph", "Toe vs Heel Speed Difference"),
)

#: Fixed unit suffix per row; rows keyed in _QUANTITY_ROWS follow the
#: user's selected display unit instead.
_UNITS: dict[str, str] = {
    "path_deviation_deg": "°",
    "aoa_deviation_deg": "°",
    "normalized_closure_deg_per_ft": " °/ft",
    "closure_during_contact_deg": "°",
    "loft_gain_during_contact_deg": "°",
    "closure_deg_per_ft": " °/ft",
    "closure_deg_per_inch": " °/in",
    "closure_deg_per_ms": " °/ms",
    "r_isa_ft": " ft",
    "r_isa_m": " m",
    "time_to_square_from_1deg_open_ms": " ms",
}

#: result/metric field -> the units drop-down quantity it follows.
_QUANTITY_ROWS: dict[str, str] = {
    "tangential_speed_mph": "speed",
    "speed_delta_mph": "speed",
    "toe_heel_speed_delta_mph": "speed",
    "closure_rate_dps": "rotation",
    "ccv_dps": "rotation",
}


class _ResultRow(QFrame):
    """A clickable result box: label left, live value right.

    Clicking (or keyboard-activating) the row emits ``clicked`` with the
    result field name so the window can show the explanation.
    """

    clicked = pyqtSignal(str)

    def __init__(self, field: str, label: str) -> None:
        super().__init__()
        self._field = field
        self.setObjectName("resultRow")
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFocusPolicy(Qt.FocusPolicy.TabFocus)
        self.setToolTip("Click for the explanation and derivation trace")
        self.setAccessibleName(label)

        row = QHBoxLayout(self)
        row.setContentsMargins(10, 6, 10, 6)
        name = QLabel(label)
        row.addWidget(name)
        row.addStretch(1)
        self.value_label = QLabel("—")
        font = self.value_label.font()
        font.setBold(True)
        self.value_label.setFont(font)
        row.addWidget(self.value_label)

    def mousePressEvent(self, event) -> None:  # type: ignore[no-untyped-def]  # noqa: N802
        """Emit the field name; keep default frame behaviour."""
        self.clicked.emit(self._field)
        super().mousePressEvent(event)

    def keyPressEvent(self, event) -> None:  # type: ignore[no-untyped-def]  # noqa: N802
        """Space/Return activate the row, matching button conventions."""
        if event.key() in (Qt.Key.Key_Space, Qt.Key.Key_Return):
            self.clicked.emit(self._field)
            return
        super().keyPressEvent(event)


class RateOfClosureMainWindow(ThemedWindowMixin, QMainWindow):
    """Interactive explorer for rotation-induced impact-point deviations."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Rate of Closure Impact Explorer")
        self.setMinimumSize(1240, 800)

        self._controls = ControlsPanel()
        self._rows: dict[str, _ResultRow] = {}
        self._club_view = Club3DView()
        self._sweep_view = SweepView()
        self._derivation_view = DerivationView()

        left_content = QWidget()
        left_layout = QVBoxLayout(left_content)
        left_layout.addWidget(self._controls)
        left_layout.addWidget(
            self._build_rows_box("Impact-Point Deviation", _RESULT_ROWS)
        )
        left_layout.addWidget(
            self._build_rows_box("Common Closure Metrics", _METRIC_ROWS)
        )
        left_layout.addWidget(self._build_explanation_box())
        left_layout.addStretch(1)
        left = QScrollArea()
        left.setWidgetResizable(True)
        left.setFrameShape(QFrame.Shape.NoFrame)
        left.setWidget(left_content)
        left.setMinimumWidth(390)

        tabs = QTabWidget()
        tabs.addTab(self._club_view, "3D Clubhead")
        tabs.addTab(self._sweep_view, "Closure Sweep")
        tabs.addTab(self._derivation_view, "Derivation && Traceability")

        splitter = QSplitter()
        splitter.addWidget(left)
        splitter.addWidget(tabs)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        self.setCentralWidget(splitter)
        self.setStatusBar(QStatusBar())
        self.setStyleSheet(
            "QFrame#resultRow { border-radius: 6px; }"
            "QFrame#resultRow:hover { border: 1px solid palette(highlight); }"
        )

        self._controls.scenarioChanged.connect(self._on_scenario)
        # Theming is applied by the shared launcher (setup_themed_app),
        # which also owns the single Theme menu — calling
        # setup_theme_support() here as well would add a duplicate.
        self._on_scenario(self._controls.scenario())
        self._show_explanation(_RESULT_ROWS[0][0])

    # ── construction ────────────────────────────────────────────────
    def _build_rows_box(
        self, title: str, rows: tuple[tuple[str, str], ...]
    ) -> QGroupBox:
        box = QGroupBox(title)
        layout = QVBoxLayout(box)
        layout.setSpacing(4)
        for field, label in rows:
            row = _ResultRow(field, label)
            row.clicked.connect(self._show_explanation)
            self._rows[field] = row
            layout.addWidget(row)
        return box

    def _build_explanation_box(self) -> QGroupBox:
        box = QGroupBox("What This Number Means")
        layout = QVBoxLayout(box)
        self._explanation = QTextBrowser()
        self._explanation.setOpenExternalLinks(False)
        self._explanation.setMinimumHeight(110)
        self._explanation.setMaximumHeight(170)
        layout.addWidget(self._explanation)
        return box

    # ── behaviour ───────────────────────────────────────────────────
    def _show_explanation(self, field: str) -> None:
        labels = dict(_RESULT_ROWS) | dict(_METRIC_ROWS)
        text = RESULT_EXPLANATIONS.get(field) or METRIC_EXPLANATIONS.get(field, "")
        self._explanation.setHtml(f"<b>{labels[field]}</b><br/>{text}")

    def _format_row(self, field: str, value: float) -> str:
        """Format one row's value in the user's selected display unit."""
        if not math.isfinite(value):
            return "∞ (not closing)"
        quantity = _QUANTITY_ROWS.get(field)
        if quantity is None:
            return f"{value:+.2f}{_UNITS[field]}"
        unit = self._controls.unit_for(quantity)
        displayed = convert_from_canonical(quantity, unit, value)
        return f"{displayed:+.2f} {unit}"

    def _on_scenario(self, scenario: ImpactScenario) -> None:
        result = solve(scenario)
        metrics = closure_metrics(scenario)
        for field, _ in _RESULT_ROWS:
            self._rows[field].value_label.setText(
                self._format_row(field, getattr(result, field))
            )
        for field, _ in _METRIC_ROWS:
            self._rows[field].value_label.setText(
                self._format_row(field, getattr(metrics, field))
            )
        self._club_view.set_scenario(scenario)
        self._sweep_view.set_scenario(scenario)
        self._derivation_view.set_scenario(scenario)
        status_bar = self.statusBar()
        if status_bar is None:  # pragma: no cover - Qt always provides one here
            return
        status_bar.showMessage(
            f"Reference {result.reference_speed_mph:.1f} mph — impact point "
            f"path {result.path_deviation_deg:+.2f}° "
            f"({'left' if result.path_deviation_deg < 0 else 'right'}), "
            f"AoA {result.aoa_deviation_deg:+.2f}°, "
            f"CCV {result.closure_rate_dps:.0f} °/s "
            f"({result.normalized_closure_deg_per_ft:.1f} °/ft)"
        )

    def closeEvent(self, event) -> None:  # type: ignore[no-untyped-def]  # noqa: N802
        """Stop the animation timer before the window goes away."""
        self._club_view.stop()
        super().closeEvent(event)
