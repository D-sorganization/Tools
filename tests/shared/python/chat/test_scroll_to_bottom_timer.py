"""MessagingMixin._scroll_to_bottom timer lifetime (Tools #5667 follow-up).

The deferred scroll used a bare ``QTimer.singleShot(10, lambda: ...)`` that
touched ``self._scroll_area`` after the dock could already be deleted.
Checked structurally on a minimal host -- the event loop is never spun.
"""

from __future__ import annotations

import pytest

pytest.importorskip("PyQt6.QtCore", reason="PyQt6 required")

from PyQt6 import sip  # noqa: E402
from PyQt6.QtCore import QTimer  # noqa: E402
from PyQt6.QtWidgets import QApplication, QLabel, QScrollArea  # noqa: E402

from shared.python.chat._qt.messaging import MessagingMixin  # noqa: E402


class _Host(MessagingMixin):
    def __init__(self) -> None:
        self._scroll_area = QScrollArea()
        tall = QLabel("x\n" * 400)
        self._scroll_area.setWidget(tall)
        self._scroll_area.resize(100, 50)


def test_scroll_timer_dies_with_scroll_area(qapp: QApplication) -> None:
    host = _Host()
    host._scroll_to_bottom()

    timers = host._scroll_area.findChildren(QTimer)
    scroll_timers = [t for t in timers if t.isSingleShot() and t.isActive()]
    assert len(scroll_timers) == 1

    sip.delete(host._scroll_area)
    assert sip.isdeleted(scroll_timers[0])


def test_scroll_timer_scrolls_to_maximum(qapp: QApplication) -> None:
    host = _Host()
    try:
        host._scroll_area.show()
        scrollbar = host._scroll_area.verticalScrollBar()
        assert scrollbar is not None
        scrollbar.setRange(0, 500)
        scrollbar.setValue(0)
        host._scroll_to_bottom()

        timers = host._scroll_area.findChildren(QTimer)
        [t for t in timers if t.isSingleShot() and t.isActive()][0].timeout.emit()

        assert scrollbar.value() == scrollbar.maximum()
    finally:
        sip.delete(host._scroll_area)
