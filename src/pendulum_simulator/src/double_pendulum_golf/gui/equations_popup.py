"""
LaTeX-quality math popup for the Pendulum Simulator.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from PyQt6.QtWidgets import QDialog, QWidget

from .equations_data import EquationTopic, _TOPICS

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def show_equations_popup(parent: QWidget | None, topic: EquationTopic) -> QDialog:
    """Show a non-modal equations popup.

    Pre: topic is a valid EquationTopic.
    Post: returns the QDialog instance (caller may discard).
    """
    from PyQt6.QtCore import Qt
    from PyQt6.QtWidgets import (
        QApplication,
        QDialog,
        QPushButton,
        QTextBrowser,
        QVBoxLayout,
    )

    if topic not in _TOPICS:
        raise ValueError(f"Unknown topic: {topic}")
    title, html = _TOPICS[topic]

    dlg = QDialog(parent)
    dlg.setWindowTitle(title)
    dlg.setMinimumSize(720, 600)
    dlg.setStyleSheet("QDialog { background: #1a1a28; }")

    layout = QVBoxLayout(dlg)
    layout.setContentsMargins(0, 0, 0, 0)

    browser = QTextBrowser()
    browser.setOpenExternalLinks(True)
    browser.setHtml(html)
    browser.setStyleSheet("QTextBrowser { background: #1a1a28; border: none; }")
    layout.addWidget(browser)

    copy_btn = QPushButton("Copy to Clipboard")

    def _copy_text() -> None:
        cb = QApplication.clipboard()
        if cb is not None:
            cb.setText(browser.toPlainText())

    copy_btn.clicked.connect(_copy_text)
    layout.addWidget(copy_btn)

    dlg.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
    dlg.show()
    logger.info("Opened equations popup: %s", title)
    return dlg
