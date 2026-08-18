"""Adversarial geometry checks for visualization landmark audits."""

from __future__ import annotations

import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtWidgets import (
    QLabel,
    QPlainTextEdit,
    QScrollArea,
    QSlider,
    QTableWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from rate_of_closure.ui.pyqt6.visualization_tab_audit import (
    interactive_overlaps,
    visible_intersection,
)

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


def test_visible_intersection_respects_scroll_viewport_clipping(qtbot) -> None:
    """A below-fold landmark must not pass merely because it is inside the tab."""
    tab = QWidget()
    scroll = QScrollArea(tab)
    scroll.setGeometry(0, 0, 320, 100)
    content = QWidget()
    layout = QVBoxLayout(content)
    layout.addWidget(QLabel("spacer"))
    layout.addSpacing(180)
    landmark = QLabel("below viewport")
    landmark.setFixedHeight(100)
    layout.addWidget(landmark)
    scroll.setWidget(content)
    tab.resize(500, 500)
    tab.show()
    qtbot.addWidget(tab)

    assert visible_intersection(landmark, tab).isEmpty()


def test_interactive_overlap_covers_item_slider_and_text_edit_families(qtbot) -> None:
    root = QWidget()
    root.resize(500, 500)
    widgets = [QTableWidget(root), QSlider(root), QTextEdit(root), QPlainTextEdit(root)]
    for widget in widgets:
        widget.setGeometry(20, 20, 220, 120)
        widget.show()
    root.show()
    qtbot.addWidget(root)

    conflicts = interactive_overlaps(root)
    assert conflicts
    assert any("QTableWidget" in conflict for conflict in conflicts)
    assert any("QSlider" in conflict for conflict in conflicts)
    assert any("QTextEdit" in conflict for conflict in conflicts)
    assert any("QPlainTextEdit" in conflict for conflict in conflicts)
