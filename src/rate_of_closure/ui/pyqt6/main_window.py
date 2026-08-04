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

from rate_of_closure.derivation import RESULT_EXPLANATIONS
from rate_of_closure.model import ImpactScenario, solve
from rate_of_closure.ui.pyqt6.club_view import Club3DView, SweepView
from rate_of_closure.ui.pyqt6.controls_panel import ControlsPanel
from rate_of_closure.ui.pyqt6.derivation_view import DerivationView

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

_UNITS: dict[str, str] = {
    "path_deviation_deg": "°",
    "aoa_deviation_deg": "°",
    "tangential_speed_mph": " mph",
    "speed_delta_mph": " mph",
    "closure_rate_dps": " °/s",
    "normalized_closure_deg_per_ft": " °/ft",
    "closure_during_contact_deg": "°",
    "loft_gain_during_contact_deg": "°",
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
        left_layout.addWidget(self._build_results_box())
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
        if _THEME_AVAILABLE:
            self.setup_theme_support(settings_app="RateOfClosure")
        self._on_scenario(self._controls.scenario())
        self._show_explanation(_RESULT_ROWS[0][0])

    # ── construction ────────────────────────────────────────────────
    def _build_results_box(self) -> QGroupBox:
        box = QGroupBox("Impact-Point Deviation")
        layout = QVBoxLayout(box)
        layout.setSpacing(4)
        for field, label in _RESULT_ROWS:
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
        label = dict(_RESULT_ROWS)[field]
        text = RESULT_EXPLANATIONS.get(field, "")
        self._explanation.setHtml(f"<b>{label}</b><br/>{text}")

    def _on_scenario(self, scenario: ImpactScenario) -> None:
        result = solve(scenario)
        for field, _ in _RESULT_ROWS:
            value = getattr(result, field)
            self._rows[field].value_label.setText(f"{value:+.2f}{_UNITS[field]}")
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
