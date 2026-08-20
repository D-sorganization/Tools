"""Small error presenter shared by performance workspace actions."""

from __future__ import annotations

from collections.abc import Callable

from PyQt6.QtWidgets import QMessageBox, QWidget


def present_value_error(
    parent: QWidget, title: str, action: Callable[[], object]
) -> None:
    """Run one calculation and present its fail-closed reason."""

    try:
        action()
    except ValueError as error:
        QMessageBox.warning(parent, title, str(error))


__all__ = ["present_value_error"]
