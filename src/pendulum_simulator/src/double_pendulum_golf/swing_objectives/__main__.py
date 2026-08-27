"""Entry point for the Swing Objective Lab.

Usage::

    python -m double_pendulum_golf.swing_objectives

Closes #4772.
"""

from __future__ import annotations

import logging
import sys

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    """Launch the Swing Objective Lab window.

    Args:
        argv: Command-line arguments; defaults to ``sys.argv``.

    Returns:
        Process exit status.
    """
    from PyQt6.QtWidgets import QApplication

    from double_pendulum_golf.gui.swing_objective_lab import SwingObjectiveLabWindow

    logging.basicConfig(level=logging.INFO)
    app = QApplication(argv if argv is not None else sys.argv)
    window = SwingObjectiveLabWindow()
    window.resize(1280, 800)
    window.show()
    return int(app.exec())


if __name__ == "__main__":
    raise SystemExit(main())
