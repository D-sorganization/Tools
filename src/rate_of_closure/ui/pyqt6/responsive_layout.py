"""Width-aware Qt containers for compact, scrollable control rails."""

from __future__ import annotations

from PyQt6.QtCore import QEvent
from PyQt6.QtWidgets import QGroupBox, QLayout, QWidget

__all__ = ["HeightForWidthGroupBox"]


class HeightForWidthGroupBox(QGroupBox):
    """A group box that reserves the height its wrapped layout requires.

    Qt layouts use a widget's ordinary size hint when distributing vertical
    space.  A wrapped ``QFormLayout`` can require substantially more height at
    a narrow width, which otherwise lets child editors collapse even inside a
    scroll area.  This container promotes the complete widget height-for-width
    result (including themed group title and frame chrome) to a real minimum
    height whenever geometry or style metrics change.
    """

    def __init__(
        self,
        title: str = "",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(title, parent)

    def event(self, event: QEvent | None) -> bool:
        handled = super().event(event)
        if event is not None and event.type() in (
            QEvent.Type.ApplicationFontChange,
            QEvent.Type.FontChange,
            QEvent.Type.LayoutRequest,
            QEvent.Type.Resize,
            QEvent.Type.Show,
            QEvent.Type.StyleChange,
        ):
            self._reserve_wrapped_height()
        return bool(handled)

    def _reserve_wrapped_height(self) -> None:
        layout: QLayout | None = self.layout()
        if layout is None or not layout.hasHeightForWidth():
            return
        widget_height = self.heightForWidth(self.width())
        required = max(
            widget_height if widget_height >= 0 else 0,
            layout.heightForWidth(self.width()),
            layout.minimumSize().height(),
        )
        if required != self.minimumHeight():
            self.setMinimumHeight(required)
