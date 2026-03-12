"""Entry point for running the application.

Usage:
    python -m double_pendulum_golf
"""

import logging
import sys
from pathlib import Path

from PyQt6.QtCore import QEvent, QObject
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import QApplication, QComboBox, QDoubleSpinBox, QSpinBox

from .gui import MainWindow
from .gui.diagnostics import get_tracker


class _WheelBlockFilter(QObject):
    """Global event filter that blocks mouse wheel on value-input widgets.

    Prevents accidental value changes when scrolling the controls panel.
    Closes #1193.
    """

    def eventFilter(self, obj: QObject, event: QEvent) -> bool:  # noqa: N802
        if event.type() == QEvent.Type.Wheel:
            if isinstance(obj, (QComboBox, QDoubleSpinBox, QSpinBox)):
                event.ignore()
                return True  # Block the event
        return False


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


def main() -> None:
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

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
