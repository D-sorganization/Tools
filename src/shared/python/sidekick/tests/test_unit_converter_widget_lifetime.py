"""Lifetime regressions for UnitConverterWidget signal slots (#5256 / Tools)."""

from __future__ import annotations

import gc
import weakref

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
