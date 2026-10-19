"""DataProcessorWidget status-reset timer (Tools #5667 follow-up).

``_set_status`` used a bare ``QTimer.singleShot(5000, lambda: ...)``: it
outlived the widget (``wrapped C/C++ object ... has been deleted`` in a later
test's event loop) and stacked one reset per call, so an older reset could
blank a newer status early. Now one label-owned timer is restarted per call.
Checked structurally -- the event loop is never spun.
"""

from __future__ import annotations

import pytest

pytest.importorskip("PyQt6.QtCore", reason="PyQt6 required")

from PyQt6 import sip  # noqa: E402
from PyQt6.QtCore import QTimer  # noqa: E402
from PyQt6.QtWidgets import QApplication  # noqa: E402
from sidekick.ui.widgets.data_processor_widget import (  # noqa: E402
    DataProcessorWidget,
)


def _status_timers(widget: DataProcessorWidget) -> list[QTimer]:
    return widget.status_label.findChildren(QTimer)


def test_status_reset_timer_dies_with_widget(qapp: QApplication) -> None:
    widget = DataProcessorWidget()
    widget._set_status("Loaded 5 rows")

    timers = _status_timers(widget)
    assert len(timers) == 1
    reset_timer = timers[0]
    assert reset_timer.isActive()
    assert reset_timer.isSingleShot()

    sip.delete(widget)
    assert sip.isdeleted(reset_timer)


def test_newer_status_restarts_the_one_reset_timer(qapp: QApplication) -> None:
    widget = DataProcessorWidget()
    try:
        widget._set_status("first")
        first = _status_timers(widget)
        widget._set_status("second")

        assert _status_timers(widget) == first
        assert len(first) == 1
        assert widget.status_label.text() == "second"
    finally:
        sip.delete(widget)


def test_status_reset_timer_restores_ready(qapp: QApplication) -> None:
    widget = DataProcessorWidget()
    try:
        widget._set_status("Saved")
        _status_timers(widget)[0].timeout.emit()

        assert widget.status_label.text() == "Ready"
    finally:
        sip.delete(widget)
