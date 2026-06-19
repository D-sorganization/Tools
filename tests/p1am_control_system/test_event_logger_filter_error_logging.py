"""Regression test for #3670.

``EventLogViewerWidget.update_event_types_combobox`` previously swallowed all
exceptions from the SQLite event database with a bare ``except Exception: pass``,
silently leaving the filter combobox empty with no diagnostic. It must now log
the failure via the module logger so a degrading DB is observable.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

from p1am_control_system.desktop import event_logger


class _FakeCombo:
    """Minimal QComboBox stand-in for headless testing."""

    def __init__(self) -> None:
        self._items: list[str] = []
        self._index = 0

    def currentText(self) -> str:
        return self._items[self._index] if self._items else ""

    def clear(self) -> None:
        self._items = []
        self._index = 0

    def addItem(self, text: str) -> None:
        self._items.append(text)

    def findText(self, text: str) -> int:
        return self._items.index(text) if text in self._items else -1

    def setCurrentIndex(self, idx: int) -> None:
        self._index = idx


def test_db_failure_is_logged_not_swallowed(caplog) -> None:
    def _boom() -> list[str]:
        raise RuntimeError("corrupt db")

    widget = SimpleNamespace(
        event_type_combo=_FakeCombo(),
        logger=SimpleNamespace(get_unique_event_types=_boom),
    )

    with caplog.at_level(logging.ERROR, logger=event_logger.__name__):
        event_logger.EventLogViewerWidget.update_event_types_combobox(widget)

    assert any("event-type filter" in rec.getMessage() for rec in caplog.records), (
        "DB failure must be logged"
    )
    # Combobox still has the default 'All' entry and did not raise.
    assert widget.event_type_combo._items == ["All"]
