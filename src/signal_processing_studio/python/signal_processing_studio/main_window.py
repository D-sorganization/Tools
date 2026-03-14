"""Signal Processing Studio - Unified signal processing application.

Hosts Function Generator, Signal Toolkit, and Polynomial Generator
as tabs within a single QMainWindow with shared theme support.
"""

from __future__ import annotations

import sys

from PyQt6.QtGui import QAction, QKeySequence
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QMenu,
    QMessageBox,
    QTabWidget,
)
from signal_toolkit.polynomial_generator import PolynomialGeneratorWidget
from signal_toolkit.widget import SignalToolkitWidget

from .signal_bus import SignalBus

# Theme integration (optional - graceful degradation)
try:
    from shared.python.theme.integration import ThemedWindowMixin

    HAS_THEME = True
except ImportError:
    HAS_THEME = False

# Function Generator import (may need path setup)
try:
    from function_generator.ui.pyqt6.main_window import FunctionGeneratorWidget

    HAS_FUNC_GEN = True
except ImportError:
    HAS_FUNC_GEN = False


def _get_base_classes() -> tuple:
    """Build base class tuple dynamically based on available dependencies."""
    if HAS_THEME:
        return (ThemedWindowMixin, QMainWindow)
    return (QMainWindow,)


class SignalProcessingStudio(*_get_base_classes()):  # type: ignore[misc]
    """Unified signal processing application."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Signal Processing Studio")
        self.setMinimumSize(1300, 850)

        # Central tab widget
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        # Create widgets (builtin themes disabled - host provides styling)
        if HAS_FUNC_GEN:
            self.func_gen = FunctionGeneratorWidget(self, use_builtin_theme=False)
            self.tabs.addTab(self.func_gen, "Function Generator")
        else:
            self.func_gen = None

        self.toolkit = SignalToolkitWidget(self, use_builtin_theme=False)
        self.tabs.addTab(self.toolkit, "Signal Toolkit")

        self.poly_gen = PolynomialGeneratorWidget(self, use_builtin_theme=False)
        self.tabs.addTab(self.poly_gen, "Polynomial Generator")

        # Wire cross-widget communication
        self.signal_bus: SignalBus | None = None
        if self.func_gen is not None:
            self.signal_bus = SignalBus(
                func_gen=self.func_gen,
                toolkit=self.toolkit,
                poly_gen=self.poly_gen,
                status_callback=self._update_status,
            )
        else:
            self.signal_bus = None
            # Still wire polynomial -> toolkit
            self.poly_gen.polynomial_generated.connect(self._on_poly_fallback)

        # Create menus
        self._create_menus()

        # Theme support
        if HAS_THEME:
            self.setup_theme_support(
                settings_app="SignalProcessingStudio",
                show_custom_options=True,
            )

        # Status bar
        self.statusBar().showMessage("Ready")

    def _create_menus(self) -> None:
        """Create the application menu bar."""
        menubar = self.menuBar()

        # File menu
        file_menu = QMenu("&File", self)

        if self.func_gen is not None:
            send_action = QAction("Send to &Toolkit", self)
            send_action.setShortcut(QKeySequence("Ctrl+T"))
            send_action.setStatusTip("Send current signal to Signal Toolkit tab")
            send_action.triggered.connect(self._send_to_toolkit)
            file_menu.addAction(send_action)
            file_menu.addSeparator()

        exit_action = QAction("E&xit", self)
        exit_action.setShortcut(QKeySequence("Ctrl+Q"))
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        menubar.addMenu(file_menu)

        # Help menu
        help_menu = QMenu("&Help", self)
        about_action = QAction("&About", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)
        menubar.addMenu(help_menu)

    def _send_to_toolkit(self) -> None:
        """Send current Function Generator signal to Toolkit and switch tab."""
        if self.signal_bus is not None:
            self.signal_bus.send_current_to_toolkit()
            # Switch to the toolkit tab
            toolkit_idx = self.tabs.indexOf(self.toolkit)
            if toolkit_idx >= 0:
                self.tabs.setCurrentIndex(toolkit_idx)

    def _on_poly_fallback(self, joint_name: str, coeffs: list) -> None:
        """Fallback when Function Generator is unavailable."""
        assert joint_name is not None, "joint_name must be provided"
        import numpy as np
        from signal_toolkit.core import SignalGenerator

        if self.toolkit.current_signal is not None:
            t = self.toolkit.current_signal.time
        else:
            t = np.linspace(0, 10, 1000)

        signal = SignalGenerator.polynomial(t, list(reversed(coeffs)))
        signal.name = f"Polynomial ({joint_name})"
        self.toolkit.load_external_signal(signal)
        self._update_status(f"Polynomial from {joint_name} sent to Toolkit")

    def _update_status(self, message: str) -> None:
        """Update the status bar message."""
        self.statusBar().showMessage(message, 5000)

    def _show_about(self) -> None:
        """Show the about dialog."""
        QMessageBox.about(
            self,
            "About Signal Processing Studio",
            "Signal Processing Studio v1.0\n\n"
            "Unified interface for:\n"
            "  - Function Generator (waveform creation)\n"
            "  - Signal Toolkit (analysis, filtering, fitting)\n"
            "  - Polynomial Generator (visual curve design)\n\n"
            "Part of the Tools repository.",
        )


def main() -> int:
    """Run as standalone application."""
    app = QApplication(sys.argv)
    app.setApplicationName("Signal Processing Studio")
    app.setOrganizationName("D-sorganization")

    window = SignalProcessingStudio()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
