#!/usr/bin/env python3
"""Standalone launcher for Signal Toolkit + Polynomial Generator.

Provides an integrated two-tab GUI for signal analysis and polynomial function
generation.  Intended for standalone diagnostics, troubleshooting, and
validation without requiring the full UpstreamDrift application context.

Tabs (matching UpstreamDrift tab ordering):
    1. Polynomial Generator  — visual polynomial curve design
    2. Signal Toolkit        — signal analysis, filtering, and fitting

Usage::

    python launch_signal_toolkit.py

The launcher uses the shared components from ``src/shared/python/signal_toolkit``
without modifying them, so it is safe to run alongside other projects that depend
on that package.
"""

from __future__ import annotations

import sys

from _bootstrap import bootstrap

bootstrap(__file__)

import numpy as np  # noqa: E402
from PyQt6.QtGui import QAction, QKeySequence  # noqa: E402
from PyQt6.QtWidgets import (  # noqa: E402
    QApplication,
    QMainWindow,
    QMenu,
    QMessageBox,
    QTabWidget,
)
from signal_toolkit.core import SignalGenerator  # noqa: E402
from signal_toolkit.polynomial_generator import PolynomialGeneratorWidget  # noqa: E402
from signal_toolkit.widget import SignalToolkitWidget  # noqa: E402

# ---------------------------------------------------------------------------
# Unified dark stylesheet applied at the window level so both child widgets
# render consistently (both are instantiated with use_builtin_theme=False).
# ---------------------------------------------------------------------------
_DARK_STYLESHEET = """
QMainWindow, QWidget {
    background-color: #2b2b2b;
    color: #ffffff;
    font-family: 'Segoe UI', Arial, sans-serif;
}
QMenuBar {
    background-color: #1e1e1e;
    color: #cccccc;
}
QMenuBar::item:selected {
    background-color: #3d3d3d;
}
QMenu {
    background-color: #2b2b2b;
    border: 1px solid #555;
    color: #cccccc;
}
QMenu::item:selected {
    background-color: #0078d4;
    color: #ffffff;
}
QStatusBar {
    background-color: #1e1e1e;
    color: #aaaaaa;
}
QGroupBox {
    border: 1px solid #444;
    border-radius: 6px;
    margin-top: 12px;
    padding-top: 10px;
    font-weight: bold;
    color: #e0e0e0;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 5px;
}
QPushButton {
    background-color: #3d3d3d;
    border: 1px solid #555;
    border-radius: 4px;
    padding: 6px 12px;
    color: #fff;
}
QPushButton:hover {
    background-color: #4d4d4d;
    border: 1px solid #666;
}
QPushButton:pressed {
    background-color: #2b2b2b;
}
QPushButton:disabled {
    background-color: #333;
    color: #666;
}
QPushButton#fitBtn {
    background-color: #2e7d32;
    color: white;
    font-weight: bold;
    border: none;
    padding: 10px;
}
QPushButton#fitBtn:hover {
    background-color: #388e3c;
}
QComboBox, QDoubleSpinBox, QSpinBox, QLineEdit {
    background-color: #1e1e1e;
    border: 1px solid #555;
    border-radius: 4px;
    padding: 4px;
    color: #fff;
}
QComboBox::drop-down {
    border: none;
}
QTextEdit {
    background-color: #1e1e1e;
    border: 1px solid #555;
    border-radius: 4px;
    color: #fff;
}
QTabWidget::pane {
    border: 1px solid #444;
    background: #2b2b2b;
}
QTabBar::tab {
    background: #333;
    color: #ccc;
    padding: 8px 16px;
    border-top-left-radius: 4px;
    border-top-right-radius: 4px;
}
QTabBar::tab:selected {
    background: #0078d4;
    color: white;
}
QTabBar::tab:hover:!selected {
    background: #444;
}
QSlider::groove:horizontal {
    height: 6px;
    background: #444;
    border-radius: 3px;
}
QSlider::handle:horizontal {
    width: 16px;
    margin: -5px 0;
    background: #0078d4;
    border-radius: 8px;
}
QScrollArea {
    border: none;
}
QLabel {
    color: #cccccc;
}
QRadioButton {
    color: #e0e0e0;
}
QSplitter::handle {
    background-color: #555;
}
"""


class SignalToolkitLauncher(QMainWindow):
    """Combined Polynomial Generator + Signal Toolkit application window.

    Polynomial Generator occupies the first tab (matching UpstreamDrift layout)
    so polynomial functions can be visually designed and immediately forwarded to
    the Signal Toolkit for further analysis.
    """

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Signal Toolkit — Standalone Launcher")
        self.setMinimumSize(1200, 800)
        self.setStyleSheet(_DARK_STYLESHEET)

        # Central tab widget
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        # Tab 1 — Polynomial Generator (first, matching UpstreamDrift ordering)
        self.poly_gen = PolynomialGeneratorWidget(self, use_builtin_theme=False)
        self.tabs.addTab(self.poly_gen, "Polynomial Generator")

        # Tab 2 — Signal Toolkit
        self.toolkit = SignalToolkitWidget(self, use_builtin_theme=False)
        self.tabs.addTab(self.toolkit, "Signal Toolkit")

        # Route polynomial fits into the toolkit for immediate inspection
        self.poly_gen.polynomial_generated.connect(self._on_poly_generated)

        self._build_menus()
        sb = self.statusBar()
        if sb is not None:
            sb.showMessage("Ready")

    # ------------------------------------------------------------------
    # Signal routing
    # ------------------------------------------------------------------

    def _on_poly_generated(self, joint_name: str, coeffs: list) -> None:
        """Convert polynomial coefficients to a Signal and load into Toolkit.

        Args:
            joint_name: Name of the joint / channel from the polynomial widget.
            coeffs: Polynomial coefficients in highest-degree-first order
                (as returned by ``numpy.polyfit``).
        """
        # Use the toolkit's current time axis when available so the scales match.
        if self.toolkit.current_signal is not None:
            t = self.toolkit.current_signal.time
        else:
            t = np.linspace(0, 10, 1000)

        # np.polyfit returns highest-degree-first; SignalGenerator expects lowest-first.
        signal = SignalGenerator.polynomial(t, list(reversed(coeffs)))
        signal.name = f"Polynomial ({joint_name})"
        self.toolkit.load_external_signal(signal)

        sb = self.statusBar()
        if sb is not None:
            sb.showMessage(f"Polynomial ({joint_name}) sent to Signal Toolkit", 5000)

    # ------------------------------------------------------------------
    # Menu bar
    # ------------------------------------------------------------------

    def _build_menus(self) -> None:
        """Create the application menu bar."""
        menubar = self.menuBar()
        assert menubar is not None

        # --- File ---
        file_menu = QMenu("&File", self)

        go_poly = QAction("&Polynomial Generator\tCtrl+1", self)
        go_poly.setShortcut(QKeySequence("Ctrl+1"))
        go_poly.setStatusTip("Switch to Polynomial Generator tab")
        go_poly.triggered.connect(lambda: self.tabs.setCurrentIndex(0))
        file_menu.addAction(go_poly)

        go_toolkit = QAction("&Signal Toolkit\tCtrl+2", self)
        go_toolkit.setShortcut(QKeySequence("Ctrl+2"))
        go_toolkit.setStatusTip("Switch to Signal Toolkit tab")
        go_toolkit.triggered.connect(lambda: self.tabs.setCurrentIndex(1))
        file_menu.addAction(go_toolkit)

        file_menu.addSeparator()

        exit_action = QAction("E&xit", self)
        exit_action.setShortcut(QKeySequence("Ctrl+Q"))
        exit_action.setStatusTip("Close the launcher")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        menubar.addMenu(file_menu)

        # --- Help ---
        help_menu = QMenu("&Help", self)

        about_action = QAction("&About", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)

        menubar.addMenu(help_menu)

    def _show_about(self) -> None:
        """Display an about dialog."""
        QMessageBox.about(
            self,
            "About Signal Toolkit Launcher",
            "Signal Toolkit — Standalone Launcher\n\n"
            "Integrated interface for:\n"
            "  Tab 1 · Polynomial Generator — visual curve design\n"
            "  Tab 2 · Signal Toolkit       — analysis, filtering, fitting\n\n"
            "Polynomial functions designed in Tab 1 are automatically forwarded\n"
            "to the Signal Toolkit for inspection.\n\n"
            "Part of the D-sorganization Tools repository.\n"
            "Uses shared components from src/shared/python/signal_toolkit.",
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> int:
    """Run the standalone launcher."""
    app = QApplication(sys.argv)
    app.setApplicationName("Signal Toolkit Launcher")
    app.setOrganizationName("D-sorganization")

    window = SignalToolkitLauncher()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
