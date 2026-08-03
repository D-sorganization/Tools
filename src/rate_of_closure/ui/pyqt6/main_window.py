"""Rate of Closure Impact Explorer — PyQt6 main window.

Layout: controls on the left (scenario inputs + presets), results and the
animated 3D clubhead on the right, with a sweep tab showing how the
impact-point path deviation grows with rate of closure.

The window consumes complete :class:`~rate_of_closure.model.ImpactScenario`
objects from the controls panel and hands them to the model and views —
it never reaches into widgets or model internals (LoD).
"""

from __future__ import annotations

import logging

from PyQt6.QtWidgets import (
    QFormLayout,
    QGroupBox,
    QLabel,
    QMainWindow,
    QSplitter,
    QStatusBar,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from rate_of_closure.model import ImpactScenario, solve
from rate_of_closure.ui.pyqt6.club_view import Club3DView, SweepView
from rate_of_closure.ui.pyqt6.controls_panel import ControlsPanel

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


_RESULT_ROWS: tuple[tuple[str, str], ...] = (
    ("path_deviation_deg", "Impact-point path vs reference"),
    ("aoa_deviation_deg", "Attack-angle change"),
    ("tangential_speed_mph", "Rotation-induced velocity"),
    ("speed_delta_mph", "Delivered speed change"),
    ("closure_during_contact_deg", "Face closure during contact"),
    ("loft_gain_during_contact_deg", "Dynamic loft gained in contact"),
)

_UNITS: dict[str, str] = {
    "path_deviation_deg": "°",
    "aoa_deviation_deg": "°",
    "tangential_speed_mph": " mph",
    "speed_delta_mph": " mph",
    "closure_during_contact_deg": "°",
    "loft_gain_during_contact_deg": "°",
}


class RateOfClosureMainWindow(ThemedWindowMixin, QMainWindow):
    """Interactive explorer for rotation-induced impact-point deviations."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Rate of Closure Impact Explorer")
        self.setMinimumSize(1200, 780)

        self._controls = ControlsPanel()
        self._results_labels: dict[str, QLabel] = {}
        self._club_view = Club3DView()
        self._sweep_view = SweepView()

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.addWidget(self._controls)
        left_layout.addWidget(self._build_results_box())
        left_layout.addStretch(1)

        tabs = QTabWidget()
        tabs.addTab(self._club_view, "3D Clubhead")
        tabs.addTab(self._sweep_view, "Closure Sweep")

        splitter = QSplitter()
        splitter.addWidget(left)
        splitter.addWidget(tabs)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        self.setCentralWidget(splitter)
        self.setStatusBar(QStatusBar())

        self._controls.scenarioChanged.connect(self._on_scenario)
        if _THEME_AVAILABLE:
            self.setup_theme_support(settings_app="RateOfClosure")
        self._on_scenario(self._controls.scenario())

    # ── construction ────────────────────────────────────────────────
    def _build_results_box(self) -> QGroupBox:
        box = QGroupBox("Impact-Point Deviation")
        form = QFormLayout(box)
        for name, label in _RESULT_ROWS:
            value = QLabel("—")
            self._results_labels[name] = value
            form.addRow(f"{label}:", value)
        return box

    # ── behaviour ───────────────────────────────────────────────────
    def _on_scenario(self, scenario: ImpactScenario) -> None:
        result = solve(scenario)
        for name, _ in _RESULT_ROWS:
            value = getattr(result, name)
            self._results_labels[name].setText(f"{value:+.2f}{_UNITS[name]}")
        self._club_view.set_scenario(scenario)
        self._sweep_view.set_scenario(scenario)
        status_bar = self.statusBar()
        if status_bar is None:  # pragma: no cover - Qt always provides one here
            return
        status_bar.showMessage(
            f"Reference {result.reference_speed_mph:.1f} mph — impact point "
            f"path {result.path_deviation_deg:+.2f}° "
            f"({'left' if result.path_deviation_deg < 0 else 'right'}), "
            f"AoA {result.aoa_deviation_deg:+.2f}°"
        )

    def closeEvent(self, event) -> None:  # type: ignore[no-untyped-def]  # noqa: N802
        """Stop the animation timer before the window goes away."""
        self._club_view.stop()
        super().closeEvent(event)
