"""Small-window layout regression test (#4120 V2 defect fix).

Reported defect: entry boxes and value labels became unreadable at
small window sizes. The window now supports 1024x700 (scrolling control
columns, minimum entry widths, tooltip-backed labels); this headless
test resizes the main window to that floor, walks every tab (including
nested display sub-tabs), and asserts that every visible QLineEdit /
QDoubleSpinBox keeps a readable width and that no visible entry widget
collapses to zero height.
"""

from __future__ import annotations

import pytest

pytest.importorskip("PyQt6")
pytest.importorskip("pytestqt")

from PyQt6.QtWidgets import (  # noqa: E402
    QAbstractSpinBox,
    QLineEdit,
    QTabWidget,
)

from rate_of_closure.ui.pyqt6.main_window import RateOfClosureMainWindow  # noqa: E402

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]

#: Verified window floor (width, height) — matches gui_registration.
SMALL_WINDOW = (1024, 700)

#: Readable minimum width [px] for a typed entry at the window floor.
MIN_ENTRY_WIDTH_PX = 64


@pytest.fixture
def small_window(qtbot):  # type: ignore[no-untyped-def]
    window = RateOfClosureMainWindow()
    qtbot.addWidget(window)
    window.resize(*SMALL_WINDOW)
    window.show()
    qtbot.waitExposed(window)
    yield window
    window._club_view.stop()
    window._simulation_tab.stop()


def _visible_entries(window):  # type: ignore[no-untyped-def]
    """All visible entry widgets (spin boxes and their line edits)."""
    entries = [w for w in window.findChildren(QAbstractSpinBox) if w.isVisible()]
    entries += [w for w in window.findChildren(QLineEdit) if w.isVisible()]
    return entries


def _walk_tabs(window, qtbot):  # type: ignore[no-untyped-def]
    """Yield after activating every page of every (nested) tab widget."""
    for tabs in window.findChildren(QTabWidget):
        for index in range(tabs.count()):
            tabs.setCurrentIndex(index)
            qtbot.wait(10)
            yield f"{type(tabs.widget(index)).__name__}[{index}]"


class TestSmallWindowLayout:
    def test_window_minimum_matches_the_supported_floor(self, small_window) -> None:  # type: ignore[no-untyped-def]
        assert small_window.minimumWidth() <= SMALL_WINDOW[0]
        assert small_window.minimumHeight() <= SMALL_WINDOW[1]
        assert small_window.width() == SMALL_WINDOW[0]
        assert small_window.height() == SMALL_WINDOW[1]

    def test_every_visible_entry_stays_readable_on_every_tab(
        self, small_window, qtbot
    ) -> None:  # type: ignore[no-untyped-def]
        checked = 0
        for page in _walk_tabs(small_window, qtbot):
            for entry in _visible_entries(small_window):
                assert entry.width() >= MIN_ENTRY_WIDTH_PX, (
                    f"{type(entry).__name__} is {entry.width()}px wide on "
                    f"{page} — unreadable below {MIN_ENTRY_WIDTH_PX}px"
                )
                assert (
                    entry.height() > 0
                ), f"{type(entry).__name__} collapsed to zero height on {page}"
                checked += 1
        assert checked > 20, "the walk must actually visit entry widgets"

    def test_no_visible_zero_height_widgets_in_control_columns(
        self, small_window, qtbot
    ) -> None:  # type: ignore[no-untyped-def]
        from PyQt6.QtWidgets import QComboBox, QLabel, QPushButton

        for page in _walk_tabs(small_window, qtbot):
            for cls in (QComboBox, QPushButton, QLabel):
                for widget in small_window.findChildren(cls):
                    if widget.isVisible():
                        assert widget.height() > 0, (
                            f"{type(widget).__name__} "
                            f"({widget.objectName() or widget}) has zero "
                            f"height on {page}"
                        )
