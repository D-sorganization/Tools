"""Lifetime regressions for UnitConverterWidget signal slots (#5256 / Tools)."""

from __future__ import annotations

import gc
import weakref

from PyQt6 import sip
from PyQt6.QtCore import QTimer
from sidekick.ui.widgets.unit_converter_widget import UnitConverterWidget


def test_unit_converter_freed_by_refcount_without_cyclic_gc(qapp) -> None:
    """Signal slots must not capture self; refcount alone must free the widget."""
    gc.disable()
    try:
        widget = UnitConverterWidget()
        widget_ref = weakref.ref(widget)
        del widget
        assert widget_ref() is None
    finally:
        gc.enable()


def test_copy_feedback_timer_dies_with_widget(qapp) -> None:
    """The copy-button reset timer must be owned by, and die with, the button.

    A bare ``QTimer.singleShot`` lambda outlived the widget and raised
    ``RuntimeError: wrapped C/C++ object of type QPushButton has been deleted``
    inside the Qt event loop of whichever test ran a second later. Checked
    structurally (no event-loop spin) so leaked timers from unrelated tests on
    the same worker cannot fire inside this test.
    """
    widget = UnitConverterWidget()
    row = widget.recent_widgets[0]
    row.to_value.setText("42")
    widget._copy_result(row.index)

    timers = row.copy_btn.findChildren(QTimer)
    assert len(timers) == 1
    reset_timer = timers[0]
    assert reset_timer.isActive()
    assert reset_timer.isSingleShot()

    sip.delete(widget)
    assert sip.isdeleted(reset_timer)
