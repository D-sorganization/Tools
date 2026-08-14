"""Hover-hint completeness sweep (#4120 V4).

Headless walk over every (nested) tab of the main window asserting
that every interactive widget — buttons, checkboxes, combos, sliders,
spin boxes, line edits — carries a tooltip (its own or an ancestor's,
matching Qt's tooltip inheritance for composite controls).
"""

from __future__ import annotations

import pytest

pytest.importorskip("PyQt6")
pytest.importorskip("pytestqt")

from PyQt6.QtWidgets import (  # noqa: E402
    QAbstractButton,
    QAbstractSpinBox,
    QComboBox,
    QLineEdit,
    QSlider,
    QTabWidget,
    QWidget,
)

from rate_of_closure.ui.pyqt6.main_window import (  # noqa: E402
    RateOfClosureMainWindow,
)

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]

_INTERACTIVE = (QAbstractButton, QComboBox, QSlider, QAbstractSpinBox, QLineEdit)


def _effective_tooltip(widget: QWidget) -> str:
    """The widget's tooltip, falling back through its ancestors."""
    node: QWidget | None = widget
    while node is not None:
        tip = node.toolTip()
        if tip:
            return tip
        node = node.parentWidget()
    return ""


def _is_internal(widget: QWidget) -> bool:
    """Skip Qt-internal children of composite widgets (spin/combo/scroll)."""
    if widget.objectName().startswith("qt_") or widget.objectName() in (
        "ScrollLeftButton",
        "ScrollRightButton",
    ):
        return True  # Qt chrome (tab-bar scrollers, table corner button)
    parent = widget.parentWidget()
    while parent is not None:
        if isinstance(parent, (QAbstractSpinBox, QComboBox, QSlider)):
            return True
        name = type(parent).__name__
        if name in ("QScrollBar", "QCalendarWidget", "NavigationToolbar2QT"):
            return True
        parent = parent.parentWidget()
    if type(widget).__name__ == "QScrollBar":
        return True
    return False


@pytest.fixture
def window(qtbot):  # type: ignore[no-untyped-def]
    win = RateOfClosureMainWindow()
    qtbot.addWidget(win)
    yield win
    if win._help_dialog is not None:
        win._help_dialog.close()
    win._club_view.stop()
    win._simulation_tab.stop()
    win._variation_tab.stop()


class TestTooltipCompleteness:
    def test_every_interactive_widget_has_a_tooltip(self, window) -> None:  # type: ignore[no-untyped-def]
        missing: list[str] = []
        for tab_index in range(window._tabs.count()):
            window._tabs.setCurrentIndex(tab_index)
            tab_name = window._tabs.tabText(tab_index)
            page = window._tabs.widget(tab_index)
            # Visit nested tab stacks too (Simulation viewers, results).
            for nested in page.findChildren(QTabWidget):
                for nested_index in range(nested.count()):
                    nested.setCurrentIndex(nested_index)
            for widget in page.findChildren(QWidget):
                if not isinstance(widget, _INTERACTIVE):
                    continue
                if _is_internal(widget):
                    continue
                if not _effective_tooltip(widget):
                    label = (
                        widget.text()
                        if isinstance(widget, QAbstractButton)
                        else widget.objectName()
                    )
                    missing.append(f"{tab_name}: {type(widget).__name__}({label!r})")
        assert missing == [], "widgets without hover hints:\n" + "\n".join(
            sorted(set(missing))
        )

    def test_left_panel_controls_have_tooltips(self, window) -> None:  # type: ignore[no-untyped-def]
        missing = []
        for widget in window._controls.findChildren(QWidget):
            if not isinstance(widget, _INTERACTIVE) or _is_internal(widget):
                continue
            if not _effective_tooltip(widget):
                missing.append(type(widget).__name__)
        assert missing == []
