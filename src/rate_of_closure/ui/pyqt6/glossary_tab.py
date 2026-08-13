"""Glossary tab — searchable term list + definition pane (#4120 V4).

Renders :data:`rate_of_closure.glossary.GLOSSARY`: a search box filters
the term list live, selecting a term shows its sourced definition, and
:meth:`GlossaryTab.select_term` lets explanation panels jump straight
to the matching term via their 'Glossary' links.
"""

from __future__ import annotations

import logging

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QSplitter,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from rate_of_closure.glossary import GLOSSARY, search_terms

logger = logging.getLogger(__name__)

__all__ = ["GlossaryTab"]


class GlossaryTab(QWidget):
    """Searchable glossary of every term used across the app."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        intro = QLabel(
            "Every technical term used in the app, with sourced "
            "definitions. Type to filter; click a term for its "
            "definition. Explanation panels link here."
        )
        intro.setWordWrap(True)

        self._search = QLineEdit()
        self._search.setPlaceholderText("Search terms and definitions…")
        self._search.setClearButtonEnabled(True)
        self._search.setToolTip(
            "Filter the glossary: matches the term name and the "
            "definition text, case-insensitive."
        )
        self._search.textChanged.connect(self._refilter)

        self._list = QListWidget()
        self._list.setToolTip("Glossary terms — click one to read its definition")
        self._list.currentItemChanged.connect(self._on_current_item)

        self._definition = QTextBrowser()
        self._definition.setOpenExternalLinks(False)
        self._definition.setToolTip("The selected term's sourced definition")

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.addWidget(self._search)
        left_layout.addWidget(self._list, stretch=1)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(left)
        splitter.addWidget(self._definition)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        layout = QVBoxLayout(self)
        layout.addWidget(intro)
        layout.addWidget(splitter, stretch=1)

        self._refilter("")
        if self._list.count() > 0:
            self._list.setCurrentRow(0)

    # ── public API ──────────────────────────────────────────────────
    def select_term(self, key: str) -> None:
        """Clear the filter and pre-select ``key`` (explanation links)."""
        if key not in GLOSSARY:
            logger.warning("unknown glossary term requested: %s", key)
            return
        self._search.clear()  # ensure the term is present in the list
        for index in range(self._list.count()):
            item = self._list.item(index)
            if item is not None and item.data(Qt.ItemDataRole.UserRole) == key:
                self._list.setCurrentItem(item)
                return

    def current_term(self) -> str | None:
        """The selected glossary key, if any (used by tests)."""
        item = self._list.currentItem()
        if item is None:
            return None
        return str(item.data(Qt.ItemDataRole.UserRole))

    # ── internals ──────────────────────────────────────────────────
    def _refilter(self, query: str) -> None:
        selected = self.current_term()
        self._list.blockSignals(True)
        self._list.clear()
        for key in search_terms(query):
            item = QListWidgetItem(GLOSSARY[key].term)
            item.setData(Qt.ItemDataRole.UserRole, key)
            item.setToolTip(GLOSSARY[key].definition)
            self._list.addItem(item)
            if key == selected:
                self._list.setCurrentItem(item)
        self._list.blockSignals(False)
        current = self._list.currentItem()
        self._on_current_item(current, None)

    def _on_current_item(
        self, current: QListWidgetItem | None, _previous: QListWidgetItem | None
    ) -> None:
        if current is None:
            self._definition.setHtml(
                "<i>No matching term — clear the search to see the full glossary.</i>"
            )
            return
        key = str(current.data(Qt.ItemDataRole.UserRole))
        entry = GLOSSARY[key]
        self._definition.setHtml(
            f"<h3 style='margin-top:0'>{entry.term}</h3><p>{entry.definition}</p>"
        )
