"""Chat docks must not outlive their test with a pending deferred scroll (#5884).

A parentless ``ChatDockWidget`` left to Python's cyclic GC can be destroyed in
the middle of a later test's event processing; if that happens inside the
dock's deferred ``scroll_now`` the C++-owned scroll bar is freed under it and
the xdist worker segfaults. The package conftest closes leftover docks after
every test. The two tests below run in order within this module (loadscope).
"""

from __future__ import annotations

import pytest

pytest.importorskip("PyQt6.QtWidgets", reason="PyQt6 required")

from PyQt6.QtCore import QTimer  # noqa: E402
from PyQt6.QtWidgets import QApplication  # noqa: E402

from shared.python.chat._chat_dock_widget_qt import (  # noqa: E402
    ChatConnectionConfig,
    ChatDockWidget,
)


def _docks_with_pending_scroll() -> list[object]:
    pending = []
    for widget in QApplication.topLevelWidgets():
        scroll_area = getattr(widget, "_scroll_area", None)
        if scroll_area is None or not hasattr(widget, "_stop_scroll_timer"):
            continue
        if any(timer.isActive() for timer in scroll_area.findChildren(QTimer)):
            pending.append(widget)
    return pending


def test_leaves_a_dock_with_a_pending_scroll(qapp: QApplication) -> None:
    dock = ChatDockWidget(
        connection=ChatConnectionConfig(
            app_context="teardown-test", app_name="teardown_test", session_id="t-1"
        )
    )
    dock._populate_history([{"role": "user", "content": "hi"}])

    assert dock in _docks_with_pending_scroll()


def test_no_dock_from_an_earlier_test_still_has_a_pending_scroll(
    qapp: QApplication,
) -> None:
    assert _docks_with_pending_scroll() == []
