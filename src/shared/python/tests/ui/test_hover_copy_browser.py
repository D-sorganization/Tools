"""Tests for the HoverCopyTextBrowser widget."""

from __future__ import annotations

from typing import Any

from PyQt6.QtCore import QEvent, QPointF, QTimer
from PyQt6.QtGui import QEnterEvent
from PyQt6.QtWidgets import QApplication

from shared.python.ui.hover_copy_browser import HoverCopyTextBrowser


def test_hover_copy_browser_initialization(qtbot: Any) -> None:
    """Test that HoverCopyTextBrowser initializes with a hidden copy button."""
    widget = HoverCopyTextBrowser()
    qtbot.addWidget(widget)

    assert widget.copy_btn is not None
    assert widget.copy_btn.isHidden()
    assert widget.copy_btn.toolTip() == "Copy to clipboard"


def test_hover_copy_browser_copy(qtbot: Any) -> None:
    """Test that copy_all_text successfully copies to the clipboard."""
    widget = HoverCopyTextBrowser()
    qtbot.addWidget(widget)
    widget.setPlainText("Hello, World!")

    widget.copy_all_text()

    clipboard = QApplication.clipboard()
    assert clipboard is not None
    assert clipboard.text() == "Hello, World!"
    assert widget.copy_btn.toolTip() == "Copied!"

    # Stop pending restore timers so they do not fire into later tests on the worker.
    for timer in widget.copy_btn.findChildren(QTimer):
        timer.stop()


def test_hover_copy_browser_hover(qtbot: Any) -> None:
    """Test hover enter/leave events show and hide the copy button."""
    widget = HoverCopyTextBrowser()
    qtbot.addWidget(widget)
    widget.resize(200, 100)

    # Simulate enter event
    enter_event = QEnterEvent(
        QPointF(50.0, 50.0),
        QPointF(50.0, 50.0),
        QPointF(50.0, 50.0),
    )
    QApplication.sendEvent(widget, enter_event)
    assert not widget.copy_btn.isHidden()

    # Simulate leave event
    ev_leave = QEvent(QEvent.Type.Leave)
    QApplication.sendEvent(widget, ev_leave)

    assert widget.copy_btn.isHidden()
