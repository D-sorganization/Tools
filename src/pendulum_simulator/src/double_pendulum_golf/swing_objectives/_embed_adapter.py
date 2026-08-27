"""Provider-safe embedding adapter for the Swing Objective Lab tile.

The launcher docks this surface inside its own window, so the adapter must
return a widget without ever starting an event loop.

Closes #4772.
"""

from __future__ import annotations

from typing import Any


def get_dockable_ui(parent: Any = None) -> Any:
    """Create the Swing Objective Lab surface without starting an event loop.

    Args:
        parent: Optional Qt parent supplied by the launcher.

    Returns:
        A ``QWidget`` hosting the comparison surface.
    """
    from PyQt6.QtWidgets import QWidget

    from double_pendulum_golf.gui.swing_objective_lab import SwingObjectiveLabWidget

    widget = SwingObjectiveLabWidget()
    if isinstance(parent, QWidget):
        widget.setParent(parent)
    return widget
