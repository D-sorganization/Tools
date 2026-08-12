# mypy: ignore-errors
"""Entry point for running the application.

Usage:
    python -m double_pendulum_golf
"""

import logging
import sys
from pathlib import Path

from PyQt6.QtCore import QEvent, QObject, Qt
from PyQt6.QtGui import QIcon, QKeySequence, QShortcut, QWheelEvent
from PyQt6.QtWidgets import QApplication, QComboBox, QDoubleSpinBox, QSpinBox

from .gui import MainWindow
from .gui.diagnostics import get_tracker

logger = logging.getLogger(__name__)


class _WheelBlockFilter(QObject):
    """Global event filter: blocks wheel on value-inputs, Ctrl+Wheel zooms fonts.

    - Plain wheel on QComboBox/QSpinBox/QDoubleSpinBox: blocked (#1193)
    - Ctrl+Wheel anywhere: scales application font by ±1pt (#1147)

    Font bounds and persistence are owned by ``MainWindow`` — this
    filter delegates to it so the user cannot zoom past the supported
    range and the value survives across launches via QSettings.
    """

    def eventFilter(self, obj: QObject | None, event: QEvent | None) -> bool:  # noqa: N802
        if event is not None and event.type() == QEvent.Type.Wheel:
            wheel: QWheelEvent = event  # type: ignore[assignment]
            # Ctrl+Wheel → font zoom (delegated to MainWindow for bounds + persist)
            if wheel.modifiers() & Qt.KeyboardModifier.ControlModifier:
                from .gui.main_window import MainWindow

                delta = wheel.angleDelta().y()
                step = 1 if delta > 0 else -1
                MainWindow.adjust_global_font_zoom(step)
                event.accept()
                return True
            # Plain wheel on value-input widgets → blocked
            if isinstance(obj, (QComboBox, QDoubleSpinBox, QSpinBox)):
                event.ignore()
                return True  # Block the event
        return False

    def reset_font(self) -> None:
        """Reset font to default size (Ctrl+0)."""
        from .gui.main_window import MainWindow

        MainWindow.reset_global_font_zoom()


_LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
_LOG_DIR = Path.home() / ".pendulum_simulator"

# Icon path (relative to this package)
_ICON_PATH = Path(__file__).parent / "resources" / "pendulum_icon.png"


def _configure_logging() -> None:
    """Set up console + file logging.

    Postcondition: root logger has at least one handler.
    """
    _LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = _LOG_DIR / "debug.log"

    logging.basicConfig(
        level=logging.INFO,
        format=_LOG_FORMAT,
        handlers=[
            logging.StreamHandler(sys.stderr),
            logging.FileHandler(log_file, encoding="utf-8"),
        ],
    )
    logging.getLogger(__name__).info("Logging configured — file: %s", log_file)


__version__ = "0.1.0"


def get_dockable_ui(parent: object | None = None) -> MainWindow:
    """Return the canonical main window for an embedding host.

    The optional parent is applied only when it is a Qt widget. Standalone
    launch remains owned by :func:`main`.
    """
    from PyQt6.QtWidgets import QWidget

    window = MainWindow()
    if isinstance(parent, QWidget):
        window.setParent(parent)
    return window


def main() -> None:
    # Handle --version flag before any GUI initialization
    if "--version" in sys.argv:
        # Version goes to stdout (not the logger) so `--version` works
        # without configuring logging and is captured by simple shells/CI.
        print(f"pendulum-simulator {__version__}")  # noqa: T201
        sys.exit(0)

    _configure_logging()

    # Initialize diagnostics tracker early — installs global exception hook
    tracker = get_tracker()
    tracker.record(
        "app_lifecycle",
        "Application starting",
        severity="info",
    )

    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    # Block mouse wheel on combo/spin boxes globally (#1193)
    _wheel_filter = _WheelBlockFilter(app)
    app.installEventFilter(_wheel_filter)

    # Set application icon (taskbar favicon)
    if _ICON_PATH.exists():
        icon = QIcon(str(_ICON_PATH))
        app.setWindowIcon(icon)
        logging.getLogger(__name__).info("App icon set from %s", _ICON_PATH)
    else:
        logging.getLogger(__name__).warning("App icon not found at %s", _ICON_PATH)
        tracker.record(
            "ui",
            f"App icon not found at {_ICON_PATH}",
            severity="warning",
        )

    window = MainWindow()
    window.show()

    # Add Ctrl+0 shortcut to reset font size (#1147)
    _reset_shortcut = QShortcut(QKeySequence("Ctrl+0"), window)
    _reset_shortcut.activated.connect(_wheel_filter.reset_font)

    sys.exit(app.exec())


if __name__ == "__main__":  # pragma: no cover
    main()
