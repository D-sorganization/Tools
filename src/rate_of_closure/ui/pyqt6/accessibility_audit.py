"""Deterministic semantic-control audit for the PyQt visualization shell."""

from __future__ import annotations

from dataclasses import dataclass

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QAbstractButton,
    QAbstractItemView,
    QAbstractSlider,
    QAbstractSpinBox,
    QComboBox,
    QLabel,
    QLineEdit,
    QWidget,
)

_SEMANTIC_CONTROL_TYPES = (
    QAbstractButton,
    QAbstractItemView,
    QAbstractSlider,
    QAbstractSpinBox,
    QComboBox,
    QLineEdit,
    FigureCanvasQTAgg,
)


@dataclass(frozen=True)
class AccessibilityFinding:
    """One missing or malformed accessible-control label."""

    widget_type: str
    object_name: str
    issue: str


@dataclass(frozen=True)
class AccessibilityAuditResult:
    """Bounded audit result for one currently visible tab."""

    control_count: int
    findings: tuple[AccessibilityFinding, ...]


def _button_text(widget: QWidget) -> str:
    if isinstance(widget, QAbstractButton):
        return str(widget.text()).replace("&", "").strip()
    return ""


def _buddy_text(root: QWidget, widget: QWidget) -> str:
    for label in root.findChildren(QLabel):
        if label.buddy() is widget:
            return str(label.text()).replace("&", "").strip()
    return ""


def accessible_control_name(root: QWidget, widget: QWidget) -> str:
    """Resolve the bounded name exposed by one semantic control."""

    explicit = str(widget.accessibleName()).strip()
    if explicit:
        return explicit
    button = _button_text(widget)
    if button:
        return button
    if isinstance(widget, QLineEdit):
        placeholder = str(widget.placeholderText()).strip()
        if placeholder:
            return placeholder
    return _buddy_text(root, widget)


def audit_visible_focusable_controls(root: QWidget) -> AccessibilityAuditResult:
    """Return semantic-control failures for the currently visible page."""

    findings: list[AccessibilityFinding] = []
    control_count = 0
    for widget in root.findChildren(QWidget):
        if not isinstance(widget, _SEMANTIC_CONTROL_TYPES):
            continue
        if isinstance(widget, QLineEdit) and isinstance(
            widget.parentWidget(), QAbstractSpinBox
        ):
            continue
        if (
            not widget.isVisible()
            or not widget.isEnabled()
            or widget.focusPolicy() == Qt.FocusPolicy.NoFocus
        ):
            continue
        control_count += 1
        name = accessible_control_name(root, widget)
        if not name:
            issue = "missing accessible name"
        elif len(name) > 512:
            issue = "accessible name exceeds 512 characters"
        else:
            continue
        findings.append(
            AccessibilityFinding(type(widget).__name__, widget.objectName(), issue)
        )
    return AccessibilityAuditResult(control_count, tuple(findings))


__all__ = [
    "AccessibilityFinding",
    "AccessibilityAuditResult",
    "accessible_control_name",
    "audit_visible_focusable_controls",
]
