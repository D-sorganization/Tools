"""Shared teardown for the chat dock tests (Tools #5884).

Dock fixtures here build parentless ``ChatDockWidget`` instances and never
close them. Signal connections keep each dock in a reference cycle, so it is
destroyed by Python's cyclic GC at an arbitrary later moment -- possibly in
the middle of another test's deferred ``scroll_now``, which then writes to a
freed C++ scroll bar and segfaults the xdist worker. Closing every leftover
dock after each test stops its timers (``closeEvent``) and deletes it
deterministically.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

try:
    from PyQt6 import sip
    from PyQt6.QtWidgets import QApplication
except ImportError:  # pragma: no cover - Qt-less environments
    QApplication = None  # type: ignore[assignment,misc]


def _leftover_docks() -> list[object]:
    """Top-level chat docks, matched by duck type.

    Some tests import the dock through the ``src.`` alias, which yields a
    distinct class object, so ``isinstance`` would miss them.
    """
    return [
        widget
        for widget in QApplication.topLevelWidgets()
        if hasattr(widget, "_stop_scroll_timer") and not sip.isdeleted(widget)
    ]


@pytest.fixture(autouse=True)
def _close_leftover_chat_docks() -> Iterator[None]:
    yield
    if QApplication is None or QApplication.instance() is None:
        return
    for dock in _leftover_docks():
        dock.close()
        sip.delete(dock)
