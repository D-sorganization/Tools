"""Non-displacing accessible state overlay around a persistent visual widget."""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QLabel, QVBoxLayout, QWidget

from rate_of_closure.variation_visual_state import (
    AnnouncementRole,
    VariationVisualOrigin,
    VariationVisualState,
)


class VisualStateFrame(QWidget):
    """Keep one visual mounted while announcing its execution evidence state."""

    def __init__(self, content: QWidget, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._content = content
        self._state_strip = QLabel()
        self._state_strip.setObjectName("variationVisualStateStrip")
        self._state_strip.setWordWrap(True)
        self._state_strip.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self._state_strip.setAlignment(Qt.AlignmentFlag.AlignLeft)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        layout.addWidget(self._state_strip)
        layout.addWidget(content)

    @property
    def content(self) -> QWidget:
        """Return the stable mounted visual content."""
        return self._content

    def set_state(self, state: VariationVisualState, announcement: str) -> None:
        """Apply semantic state without resizing or replacing the visual."""
        if not isinstance(state, VariationVisualState):
            raise TypeError("state must be a VariationVisualState")
        retained = state.visual_origin is VariationVisualOrigin.PRIOR_ACCEPTED
        label = (
            "Variation visualization — prior accepted result retained"
            if retained
            else "Variation visualization — current accepted result"
            if state.visual_origin is VariationVisualOrigin.CURRENT_ACCEPTED
            else "Variation visualization — analysis preview"
        )
        self.setAccessibleName(label)
        self.setProperty("visualPhase", state.phase.value)
        self.setProperty("visualOrigin", state.visual_origin.value)
        show_overlay = state.phase.value not in ("empty", "result")
        text = (
            f"Prior accepted result retained. {announcement}"
            if retained
            else announcement
        )
        self._state_strip.setText(text if show_overlay else "")
        self._state_strip.setAccessibleName("Variation visualization state")
        self._state_strip.setAccessibleDescription(text if show_overlay else label)
        self._state_strip.setProperty(
            "announcementRole",
            "alert" if state.announcement_role is AnnouncementRole.ALERT else "status",
        )
        self._state_strip.setVisible(show_overlay)


__all__ = ["VisualStateFrame"]
