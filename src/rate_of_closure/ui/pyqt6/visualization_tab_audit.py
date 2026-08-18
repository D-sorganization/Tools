"""Geometry utilities for deterministic PyQt visualization-tab audits."""

from __future__ import annotations

from PyQt6.QtCore import QPoint, QRect
from PyQt6.QtWidgets import (
    QAbstractButton,
    QAbstractItemView,
    QAbstractSlider,
    QAbstractSpinBox,
    QComboBox,
    QLineEdit,
    QPlainTextEdit,
    QTextEdit,
    QWidget,
)


def resolve_visual_widget(tab: QWidget, locator: str) -> QWidget:
    """Resolve one strict ``attr:`` chain or descendant ``type:`` locator."""
    prefix, separator, value = locator.partition(":")
    if separator != ":" or not value:
        raise ValueError("visual locator must have a nonempty prefix and value")
    if prefix == "attr":
        current: object = tab
        for name in value.split("."):
            current = getattr(current, name)
        if not isinstance(current, QWidget):
            raise TypeError(f"visual locator is not a QWidget: {locator}")
        return current
    if prefix == "type":
        matches = [
            child
            for child in tab.findChildren(QWidget)
            if type(child).__name__ == value
        ]
        if len(matches) != 1:
            raise ValueError(f"visual locator must resolve exactly once: {locator}")
        return matches[0]
    raise ValueError(f"unsupported visual locator prefix: {prefix}")


def mapped_rect(widget: QWidget, ancestor: QWidget) -> QRect:
    """Map a widget's local rectangle into one ancestor's coordinates."""
    return QRect(widget.mapTo(ancestor, QPoint(0, 0)), widget.size())


def visible_intersection(widget: QWidget, tab: QWidget) -> QRect:
    """Return the unobscured landmark region clipped through every ancestor."""
    visible = widget.visibleRegion().boundingRect()
    if visible.isEmpty():
        return QRect()
    mapped = QRect(widget.mapTo(tab, visible.topLeft()), visible.size())
    return mapped.intersected(tab.rect())


def interactive_overlaps(root: QWidget) -> tuple[str, ...]:
    """Return positive-area overlaps among visible sibling-like controls."""
    kinds = (
        QAbstractButton,
        QAbstractItemView,
        QAbstractSlider,
        QComboBox,
        QAbstractSpinBox,
        QLineEdit,
        QPlainTextEdit,
        QTextEdit,
    )
    controls = [
        widget
        for widget in root.findChildren(QWidget)
        if widget.isVisible()
        and isinstance(widget, kinds)
        and not visible_intersection(widget, root).isEmpty()
    ]
    conflicts: list[str] = []
    for index, left in enumerate(controls):
        left_rect = visible_intersection(left, root)
        for right in controls[index + 1 :]:
            if left.isAncestorOf(right) or right.isAncestorOf(left):
                continue
            overlap = left_rect.intersected(visible_intersection(right, root))
            if overlap.width() > 1 and overlap.height() > 1:
                left_name = left.accessibleName() or type(left).__name__
                right_name = right.accessibleName() or type(right).__name__
                conflicts.append(f"{left_name} <> {right_name}")
    return tuple(conflicts)


__all__ = [
    "interactive_overlaps",
    "mapped_rect",
    "resolve_visual_widget",
    "visible_intersection",
]
