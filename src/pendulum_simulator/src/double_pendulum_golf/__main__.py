"""Entry point for running the application.

Usage:
    python -m double_pendulum_golf
"""

import logging
import sys
from pathlib import Path

from PyQt6.QtWidgets import QApplication

from .gui import MainWindow

_LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
_LOG_DIR = Path.home() / ".pendulum_simulator"


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

    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
