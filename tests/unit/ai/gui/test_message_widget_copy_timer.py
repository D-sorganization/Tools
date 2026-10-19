"""MessageWidget copy-feedback timer lifetime (Tools #5667 follow-up).

The "Copied!" reset used a bare ``QTimer.singleShot`` lambda that outlived
the widget and raised ``wrapped C/C++ object ... has been deleted`` in a later
test's event loop. Checked structurally -- the event loop is never spun.
"""

from __future__ import annotations

import pytest

pytest.importorskip("PyQt6.QtCore", reason="PyQt6 required")

from unittest.mock import patch  # noqa: E402

from PyQt6 import sip  # noqa: E402
from PyQt6.QtCore import QTimer  # noqa: E402
from PyQt6.QtWidgets import QApplication  # noqa: E402

from shared.python.ai.gui.assistant_widgets import MessageWidget  # noqa: E402

_COPY = "shared.python.ai.gui.chat_export.copy_message_to_clipboard"


def _click_copy(widget: MessageWidget) -> None:
    with patch(_COPY):
        widget._on_copy_clicked()


def test_copy_reset_timer_dies_with_widget(qapp: QApplication) -> None:
    widget = MessageWidget("assistant", "hello")
    _click_copy(widget)

    timers = widget._copy_btn.findChildren(QTimer)
    assert len(timers) == 1
    reset_timer = timers[0]
    assert reset_timer.isActive()
    assert reset_timer.isSingleShot()

    sip.delete(widget)
    assert sip.isdeleted(reset_timer)


def test_copy_reset_timer_restores_label(qapp: QApplication) -> None:
    widget = MessageWidget("assistant", "hello")
    try:
        original = widget._copy_btn.text()
        _click_copy(widget)
        assert widget._copy_btn.text() == "Copied!"

        widget._copy_btn.findChildren(QTimer)[0].timeout.emit()

        assert widget._copy_btn.text() == original
    finally:
        sip.delete(widget)
