"""Shared clickable result-row widget for the PyQt6 explorer.

Used by the main window's deviation/metric boxes and the Simulation
tab's launch-number box: label on the left, live bold value on the
right, click (or Space/Return) emits the row's field name so the host
can show the matching explanation text.
"""

from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import QFrame, QHBoxLayout, QLabel

__all__ = ["ResultRow"]


class ResultRow(QFrame):
    """A clickable result box: label left, live value right."""

    clicked = pyqtSignal(str)

    def __init__(self, field: str, label: str) -> None:
        super().__init__()
        self._field = field
        self.setObjectName("resultRow")
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFocusPolicy(Qt.FocusPolicy.TabFocus)
        self.setToolTip("Click for the explanation and derivation trace")
        self.setAccessibleName(label)

        row = QHBoxLayout(self)
        row.setContentsMargins(10, 6, 10, 6)
        name = QLabel(label)
        row.addWidget(name)
        row.addStretch(1)
        self.value_label = QLabel("—")
        font = self.value_label.font()
        font.setBold(True)
        self.value_label.setFont(font)
        row.addWidget(self.value_label)

    def mousePressEvent(self, event) -> None:  # type: ignore[no-untyped-def]  # noqa: N802
        """Emit the field name; keep default frame behaviour."""
        self.clicked.emit(self._field)
        super().mousePressEvent(event)

    def keyPressEvent(self, event) -> None:  # type: ignore[no-untyped-def]  # noqa: N802
        """Space/Return activate the row, matching button conventions."""
        if event.key() in (Qt.Key.Key_Space, Qt.Key.Key_Return):
            self.clicked.emit(self._field)
            return
        super().keyPressEvent(event)
