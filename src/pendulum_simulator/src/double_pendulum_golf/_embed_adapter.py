"""Provider-safe embedding adapter for the pendulum workbench."""

from __future__ import annotations

from typing import Any


def get_dockable_ui(parent: Any = None) -> Any:
    """Create the canonical PyQt6 window without starting an event loop."""
    from PyQt6.QtWidgets import QWidget

    from double_pendulum_golf.gui.main_window import MainWindow

    window = MainWindow()
    if isinstance(parent, QWidget):
        window.setParent(parent)
    return window
