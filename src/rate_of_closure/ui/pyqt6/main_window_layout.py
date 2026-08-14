"""Focused layout builders for the Rate of Closure PyQt shell."""

from __future__ import annotations

from collections.abc import Callable, Sequence

from PyQt6.QtCore import Qt, QUrl
from PyQt6.QtWidgets import (
    QFrame,
    QGroupBox,
    QScrollArea,
    QSizePolicy,
    QTabWidget,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from rate_of_closure.ui.pyqt6.controls_panel import ControlsPanel
from rate_of_closure.ui.pyqt6.main_window_contracts import (
    _METRIC_ROWS,
    _RESULT_ROWS,
)
from rate_of_closure.ui.pyqt6.result_row import ResultRow

PrimaryTabSpec = tuple[str, QWidget, str]


class ResultsSidebar(QScrollArea):
    """Scrollable controls, result rows, and selected-result explanation."""

    def __init__(
        self,
        controls: ControlsPanel,
        show_explanation: Callable[[str], None],
        follow_explanation_link: Callable[[QUrl], None],
    ) -> None:
        super().__init__()
        self.rows: dict[str, ResultRow] = {}
        content = QWidget()
        content.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        layout = QVBoxLayout(content)
        layout.addWidget(controls)
        layout.addWidget(
            self._build_rows_box(
                "Impact-Point Deviation", _RESULT_ROWS, show_explanation
            )
        )
        layout.addWidget(
            self._build_rows_box(
                "Common Closure Metrics", _METRIC_ROWS, show_explanation
            )
        )
        explanation_box, self.explanation = self._build_explanation(
            follow_explanation_link
        )
        layout.addWidget(explanation_box)
        layout.addStretch(1)
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setWidget(content)
        self.setMinimumWidth(320)

    def _build_rows_box(
        self,
        title: str,
        rows: tuple[tuple[str, str], ...],
        show_explanation: Callable[[str], None],
    ) -> QGroupBox:
        """Create one labeled collection of clickable result rows."""
        box = QGroupBox(title)
        layout = QVBoxLayout(box)
        layout.setSpacing(4)
        for field, label in rows:
            row = ResultRow(field, label)
            row.clicked.connect(show_explanation)
            self.rows[field] = row
            layout.addWidget(row)
        return box

    @staticmethod
    def _build_explanation(
        follow_link: Callable[[QUrl], None],
    ) -> tuple[QGroupBox, QTextBrowser]:
        """Create the bounded rich-text result explanation browser."""
        box = QGroupBox("What This Number Means")
        layout = QVBoxLayout(box)
        explanation = QTextBrowser(box)
        explanation.setOpenExternalLinks(False)
        explanation.setOpenLinks(False)
        explanation.setToolTip(
            "Explanation of the selected result row; the Glossary link "
            "jumps to the matching term."
        )
        explanation.anchorClicked.connect(follow_link)
        explanation.setMinimumHeight(110)
        explanation.setMaximumHeight(170)
        layout.addWidget(explanation)
        return box, explanation


def create_primary_tabs(specs: Sequence[PrimaryTabSpec]) -> QTabWidget:
    """Create the movable, persistable primary workspace tab collection."""
    tabs = QTabWidget()
    tab_bar = tabs.tabBar()
    if tab_bar is None:  # pragma: no cover - Qt always creates its tab bar
        raise RuntimeError("Primary tab bar was not created")
    for module_id, widget, label in specs:
        index = tabs.addTab(widget, label)
        tab_bar.setTabData(index, module_id)
    tabs.setMovable(True)
    tab_bar.setElideMode(Qt.TextElideMode.ElideNone)
    tabs.setUsesScrollButtons(True)
    tab_bar.setToolTip(
        "Drag tabs to reorder the workspace. Tab order and the active "
        "view are saved for the next session."
    )
    return tabs


__all__ = ["PrimaryTabSpec", "ResultsSidebar", "create_primary_tabs"]
