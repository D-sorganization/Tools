"""Shared operator guards for plant-affecting HMI actions.

The Control/Routing/Inspector tabs already gate PLC writes behind an Admin role
check plus a modal confirmation. The E-stop *clear* — the least reversible
action in the system — had neither (issue #4021). These helpers are the single
definition of both gates so every call site stays consistent.
"""

from __future__ import annotations

from typing import Any

from PyQt6.QtWidgets import QMessageBox, QWidget

__all__ = ["confirm_action", "require_admin"]

ADMIN_ROLE = "Admin"
ACCESS_DENIED_TITLE = "Access Denied"


def require_admin(
    parent: QWidget | None,
    role: str,
    action: str,
    message_box: Any = QMessageBox,
) -> bool:
    """Return whether ``role`` may perform ``action``.

    Shows an "Access Denied" dialog and returns ``False`` for non-Admins.

    Args:
        parent: Dialog parent widget.
        role: The currently selected role, e.g. ``"Operator"``.
        action: Verb phrase completing "Only Admin users can ...".
        message_box: Injection seam for tests; defaults to ``QMessageBox``.

    Raises:
        TypeError: If ``role`` or ``action`` is not a string.
    """
    if not isinstance(role, str):
        raise TypeError(f"role must be a str, got {type(role).__name__}")
    if not isinstance(action, str):
        raise TypeError(f"action must be a str, got {type(action).__name__}")

    if role != ADMIN_ROLE:
        message_box.critical(
            parent,
            ACCESS_DENIED_TITLE,
            f"Only Admin users can {action}.",
        )
        return False
    return True


def confirm_action(
    parent: QWidget | None,
    title: str,
    text: str,
    message_box: Any = QMessageBox,
) -> bool:
    """Return whether the operator confirmed a modal Yes/No prompt.

    Defaults to *No* so a stray touch on a Raspberry Pi touchscreen cannot
    dismiss the dialog into the affirmative.

    Raises:
        TypeError: If ``title`` or ``text`` is not a string.
    """
    if not isinstance(title, str):
        raise TypeError(f"title must be a str, got {type(title).__name__}")
    if not isinstance(text, str):
        raise TypeError(f"text must be a str, got {type(text).__name__}")

    answer = message_box.question(
        parent,
        title,
        text,
        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        QMessageBox.StandardButton.No,
    )
    return bool(answer == QMessageBox.StandardButton.Yes)
