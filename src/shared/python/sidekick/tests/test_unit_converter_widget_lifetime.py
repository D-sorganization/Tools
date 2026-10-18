"""Lifetime regressions for UnitConverterWidget signal slots (#5256 / Tools)."""

from __future__ import annotations

import gc
import weakref

from PyQt6 import sip
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


def test_copy_feedback_timer_dies_with_widget(qtbot) -> None:
    """The copy-button reset timer must not fire after the widget is deleted.

    A bare ``QTimer.singleShot`` lambda outlived the widget and raised
    ``RuntimeError: wrapped C/C++ object of type QPushButton has been deleted``
    inside the Qt event loop of whichever test ran a second later.
    """
    widget = UnitConverterWidget()
    row = widget.recent_widgets[0]
    row.to_value.setText("42")
    widget._copy_result(row.index)

    sip.delete(widget)
    # Outlast the 1 s reset interval; pytest-qt fails the test if the timer
    # callback raises inside the event loop.
    qtbot.wait(1200)
