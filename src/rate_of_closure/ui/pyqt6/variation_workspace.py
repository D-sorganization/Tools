"""Sibling workflow container for variation analysis surfaces."""

from __future__ import annotations

from PyQt6.QtWidgets import QTabWidget, QVBoxLayout, QWidget

from rate_of_closure.ui.pyqt6.durable_ensemble_tab import DurableEnsembleTab
from rate_of_closure.ui.pyqt6.morris_tab import MorrisScreeningTab
from rate_of_closure.ui.pyqt6.variation_tab import VariationTab


class VariationWorkspace(QWidget):
    """Keep materialized, durable, and screening workflows explicit."""

    def __init__(
        self,
        monte_carlo: VariationTab,
        morris: MorrisScreeningTab,
        durable: DurableEnsembleTab | None = None,
        parent: QWidget | None = None,
    ) -> None:
        if not isinstance(monte_carlo, VariationTab):
            raise TypeError("monte_carlo must be a VariationTab")
        if not isinstance(morris, MorrisScreeningTab):
            raise TypeError("morris must be a MorrisScreeningTab")
        durable = durable or DurableEnsembleTab(None, monte_carlo.build_plan)
        if not isinstance(durable, DurableEnsembleTab):
            raise TypeError("durable must be a DurableEnsembleTab")
        super().__init__(parent)
        self._landing = monte_carlo._landing
        self._tabs = QTabWidget()
        self._tabs.setAccessibleName("Variation workflow")
        self._tabs.addTab(monte_carlo, "Monte Carlo & Dispersion")
        self._tabs.addTab(durable, "Durable Ensemble Analysis")
        self._tabs.addTab(morris, "Morris Screening")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._tabs)

    def tabs(self) -> QTabWidget:
        """Expose the stable sibling-tab navigation surface."""
        return self._tabs


__all__ = ["VariationWorkspace"]
