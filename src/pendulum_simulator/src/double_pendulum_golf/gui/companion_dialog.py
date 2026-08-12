"""Reader-oriented experiment and glossary guide for the desktop app."""

from __future__ import annotations

from collections.abc import Iterable

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QSplitter,
    QTabWidget,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from ..companion_catalog import (
    Experiment,
    GlossaryTerm,
    load_companion_catalog,
    search_glossary,
)


def _bullets(items: Iterable[str]) -> str:
    return "\n".join(f"• {item}" for item in items)


def _experiment_text(experiment: Experiment) -> str:
    return "\n\n".join(
        (
            f"Purpose\n{experiment.purpose}",
            f"Hypothesis\n{experiment.hypothesis}",
            f"What Would Challenge This Result?\n{experiment.falsifier}",
            f"Workflow\n{_bullets(experiment.workflow)}",
            f"Tips\n{_bullets(experiment.tips)}",
            f"Observables\n{_bullets(experiment.observables)}",
            f"Limitations\n{_bullets(experiment.limitations)}",
        )
    )


def _term_text(term: GlossaryTerm) -> str:
    return "\n\n".join(
        (
            term.term,
            term.definition,
            f"Plain Language\n{term.plain_language}",
            f"Units\n{term.units}",
            f"Interpretation Caution\n{term.caution}",
        )
    )


class CompanionGuideDialog(QDialog):
    """Searchable guide shared conceptually with the React companion."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._catalog = load_companion_catalog()
        self._visible_terms: tuple[GlossaryTerm, ...] = self._catalog.glossary
        self.setWindowTitle("Proximal–Distal Companion Guide")
        self.setMinimumSize(820, 620)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)

        layout = QVBoxLayout(self)
        introduction = QLabel(
            "Run bounded experiments, inspect the declared observables, and use "
            "the falsifier and limitations before interpreting a result."
        )
        introduction.setWordWrap(True)
        layout.addWidget(introduction)

        tabs = QTabWidget()
        tabs.setAccessibleName("Companion Guide Sections")
        tabs.addTab(self._build_experiments_tab(), "Guided Experiments")
        tabs.addTab(self._build_glossary_tab(), "Glossary")
        layout.addWidget(tabs)

    def _build_experiments_tab(self) -> QWidget:
        tab = QWidget()
        layout = QHBoxLayout(tab)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        self.experiment_list = QListWidget()
        self.experiment_list.setAccessibleName("Guided Experiments")
        for experiment in self._catalog.experiments:
            self.experiment_list.addItem(experiment.title)
        self.experiment_details = QTextBrowser()
        self.experiment_details.setAccessibleName("Experiment Explanation")
        splitter.addWidget(self.experiment_list)
        splitter.addWidget(self.experiment_details)
        splitter.setStretchFactor(1, 2)
        layout.addWidget(splitter)
        self.experiment_list.currentRowChanged.connect(self._show_experiment)
        self.experiment_list.setCurrentRow(0)
        return tab

    def _build_glossary_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        self.glossary_search = QLineEdit()
        self.glossary_search.setPlaceholderText("Search terms and definitions…")
        self.glossary_search.setAccessibleName("Search the Glossary")
        layout.addWidget(self.glossary_search)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        self.glossary_list = QListWidget()
        self.glossary_list.setAccessibleName("Glossary Terms")
        self.glossary_details = QTextBrowser()
        self.glossary_details.setAccessibleName("Glossary Definition")
        splitter.addWidget(self.glossary_list)
        splitter.addWidget(self.glossary_details)
        splitter.setStretchFactor(1, 2)
        layout.addWidget(splitter)
        self.glossary_search.textChanged.connect(self._filter_glossary)
        self.glossary_list.currentRowChanged.connect(self._show_term)
        self._populate_glossary(self._catalog.glossary)
        return tab

    def _show_experiment(self, row: int) -> None:
        if 0 <= row < len(self._catalog.experiments):
            experiment = self._catalog.experiments[row]
            self.experiment_details.setPlainText(_experiment_text(experiment))

    def _filter_glossary(self, query: str) -> None:
        self._populate_glossary(search_glossary(self._catalog, query))

    def _populate_glossary(self, terms: tuple[GlossaryTerm, ...]) -> None:
        self._visible_terms = terms
        self.glossary_list.clear()
        self.glossary_list.addItems([term.term for term in terms])
        if terms:
            self.glossary_list.setCurrentRow(0)
        else:
            self.glossary_details.setPlainText("No glossary terms match this search.")

    def _show_term(self, row: int) -> None:
        if 0 <= row < len(self._visible_terms):
            self.glossary_details.setPlainText(_term_text(self._visible_terms[row]))


def show_companion_guide(parent: QWidget | None = None) -> CompanionGuideDialog:
    """Create and show a non-modal companion guide."""
    dialog = CompanionGuideDialog(parent)
    dialog.show()
    return dialog
