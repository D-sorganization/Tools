"""Shared clickable result-row widget for the PyQt6 explorer.

Used by the main window's deviation/metric boxes and the Simulation
tab's launch-number box: label on the left, live bold value on the
right, click (or Space/Return) emits the row's field name so the host
can show the matching explanation text.
"""

from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QPalette
from PyQt6.QtWidgets import QFrame, QHBoxLayout, QLabel

__all__ = ["ResultRow", "explanation_html", "selection_stylesheet"]


def explanation_html(label: str, text: str, field: str) -> str:
    """Explanation-panel HTML: prominent name header + glossary link.

    The selected row's NAME leads as a heading (#4120 V4 — no ambiguity
    about which number is described), and a ``glossary:<term>`` link
    jumps to the Glossary tab, pre-selecting the matching term when the
    field maps onto one (:data:`rate_of_closure.glossary.FIELD_TO_TERM`).
    """
    from rate_of_closure.glossary import FIELD_TO_TERM

    term = FIELD_TO_TERM.get(field, "")
    return (
        f'<h3 style="margin:0 0 4px 0">{label}</h3>'
        f'<p style="margin:0">{text}</p>'
        f'<p style="margin:6px 0 0 0"><a href="glossary:{term}">'
        "Glossary</a></p>"
    )


def selection_stylesheet(palette: QPalette) -> str:
    """Row styling incl. the persistent selected state (#4120 V4).

    The selected background is the theme's own highlight color at low
    alpha — derived from the live palette, never hard-coded — so the
    selection reads correctly in every theme.
    """
    highlight = palette.color(QPalette.ColorRole.Highlight)
    tint = f"rgba({highlight.red()}, {highlight.green()}, {highlight.blue()}, 44)"
    return (
        "QFrame#resultRow { border-radius: 6px; }"
        "QFrame#resultRow:hover { border: 1px solid palette(highlight); }"
        'QFrame#resultRow[selected="true"] { '
        f"background-color: {tint}; "
        "border: 1px solid palette(highlight); }"
    )


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
        self.setProperty("selected", False)

        row = QHBoxLayout(self)
        row.setContentsMargins(10, 6, 10, 6)
        name = QLabel(label)
        # Small-window robustness (#4120): long labels may be clipped by
        # the layout, so the full text always rides on the tooltip, and
        # the value keeps a readable minimum width.
        name.setToolTip(label)
        row.addWidget(name)
        row.addStretch(1)
        self.value_label = QLabel("—")
        font = self.value_label.font()
        font.setBold(True)
        self.value_label.setFont(font)
        self.value_label.setMinimumWidth(64)
        row.addWidget(self.value_label)

    @property
    def field(self) -> str:
        """The result field this row displays."""
        return self._field

    def is_selected(self) -> bool:
        """Whether the row currently shows the persistent selected state."""
        return bool(self.property("selected"))

    def set_selected(self, selected: bool) -> None:
        """Apply/clear the persistent selected highlight (#4120 V4)."""
        if bool(self.property("selected")) == selected:
            return
        self.setProperty("selected", selected)
        style = self.style()
        if style is not None:  # repolish so the dynamic property restyles
            style.unpolish(self)
            style.polish(self)

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
