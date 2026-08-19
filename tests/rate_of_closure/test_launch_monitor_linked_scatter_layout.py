"""Readable linked-scatter status geometry and retained-data disclosure."""

from __future__ import annotations

import pytest

pytest.importorskip("PyQt6")
pytest.importorskip("pytestqt")

from PyQt6.QtCore import Qt  # noqa: E402

from rate_of_closure.ui.pyqt6.launch_monitor_analytics_tab import (  # noqa: E402
    LaunchMonitorAnalyticsTab,
)

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


def test_linked_scatter_status_is_compact_without_losing_policy(
    qtbot,  # type: ignore[no-untyped-def]
) -> None:
    tab = LaunchMonitorAnalyticsTab()
    qtbot.addWidget(tab)
    tab.resize(1050, 680)
    tab.show()
    qtbot.wait(0)

    status = tab.preview_status.text()
    assert status.startswith("Displayed 120 of 120 finite pairs")
    assert "Ranges:" in status
    assert status.count(" to ") == 2
    assert "No retained source row selected" in status
    assert len(status) < 220
    assert "All retained rows remain exportable" in (
        tab.preview_status.accessibleDescription()
    )
    assert tab.preview.height() >= 260
    assert tab.preview_status.height() < tab.preview.height()
    assert tab.preview_status.geometry().top() - tab.preview.geometry().bottom() >= 12

    tab.preview.setFocus()
    qtbot.keyPress(tab.preview, Qt.Key.Key_End)
    selected = tab.preview_status.text()
    assert "Retained row index 119 (zero-based)" in selected
    assert "shot demo-120" in selected
