"""Outcome, perturbation-source, and phase controls for swing ensembles."""

from __future__ import annotations

from typing import cast

import numpy as np
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import QComboBox, QFormLayout, QSlider, QWidget

from rate_of_closure.variation.plot_data import EnsemblePlotDataset
from rate_of_closure.variation.simulation_types import TrialEvaluationStatus


class ArcFilterControls(QWidget):
    """Compact selectors returning deterministic canonical trial indices."""

    changed = pyqtSignal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._trial_count = 0
        self._outcome = QComboBox()
        self._source = QComboBox()
        self._band = QComboBox()
        self._phase = QSlider()
        self._phase.setOrientation(Qt.Orientation.Horizontal)
        self._phase.setRange(5, 100)
        self._phase.setValue(100)
        for control, tooltip in (
            (self._outcome, "Show all trials or one available outcome cohort."),
            (
                self._source,
                "Select the sampled perturbation source used for quantile filtering.",
            ),
            (self._band, "Restrict the perturbation to a populated rank band."),
            (self._phase, "Show the leading portion of the common-time swing trace."),
        ):
            control.setToolTip(tooltip)
        form = QFormLayout(self)
        form.setContentsMargins(0, 0, 0, 0)
        form.addRow("Outcome Cohort", self._outcome)
        form.addRow("Perturbation Source", self._source)
        form.addRow("Source Quantile Band", self._band)
        form.addRow("Displayed Phase End", self._phase)
        self._outcome.currentIndexChanged.connect(self.changed)
        self._source.currentIndexChanged.connect(self._source_changed)
        self._band.currentIndexChanged.connect(self.changed)
        self._phase.valueChanged.connect(self.changed)

    def set_dataset(self, dataset: EnsemblePlotDataset) -> None:
        """Populate only cohorts and quantile bands that contain trials."""
        self._trial_count = len(dataset.cohorts)
        self._outcome.blockSignals(True)
        self._outcome.clear()
        self._outcome.addItem("All Outcomes", None)
        for status in TrialEvaluationStatus:
            if status in dataset.cohorts:
                self._outcome.addItem(status.value.replace("_", " ").title(), status)
        self._outcome.blockSignals(False)
        self._source.blockSignals(True)
        self._source.clear()
        self._source.addItem("All Sources", None)
        for name in dataset.result.variation.input_names:
            self._source.addItem(name, name)
        self._source.blockSignals(False)
        self._source_changed()

    def trial_indices(self, dataset: EnsemblePlotDataset) -> np.ndarray:
        """Return stable trial indices satisfying every active filter."""
        indices: np.ndarray = np.arange(len(dataset.cohorts), dtype=int)
        outcome = self._outcome.currentData()
        if outcome is not None:
            indices = indices[
                np.fromiter(
                    (dataset.cohorts[i] is outcome for i in indices), dtype=bool
                )
            ]
        source = self._source.currentData()
        band = self._band.currentData()
        if source is None or band is None:
            return indices
        column = dataset.result.variation.input_names.index(str(source))
        ranked = sorted(
            indices,
            key=lambda i: (dataset.result.variation.inputs[i, column], i),
        )
        group_count, group_index = band
        groups = [
            group
            for group in np.array_split(np.asarray(ranked, dtype=int), group_count)
            if group.size
        ]
        selected: np.ndarray = np.asarray(
            groups[min(group_index, len(groups) - 1)], dtype=int
        )
        return selected

    def sample_count(self, total: int) -> int:
        """Return a non-empty leading common-time sample count."""
        return max(1, int(np.ceil(total * self._phase.value() / 100.0)))

    def _source_changed(self) -> None:
        has_source = self._source.currentData() is not None
        self._band.blockSignals(True)
        self._band.clear()
        self._band.addItem("All Values", None)
        if has_source:
            labels = (
                ("Lower Half", "Upper Half")
                if self._trial_count < 3
                else ("Lower Third", "Middle Third", "Upper Third")
            )
            for index, label in enumerate(labels):
                self._band.addItem(label, (len(labels), index))
        self._band.setEnabled(has_source)
        self._band.blockSignals(False)
        self.changed.emit()

    @property
    def phase_percent(self) -> int:
        """Return the displayed leading phase percentage."""
        return cast("int", self._phase.value())

    @property
    def outcome_filter(self) -> str | None:
        """Return the stable selected outcome value."""
        value = self._outcome.currentData()
        return None if value is None else str(value.value)

    @property
    def perturbation_source_key(self) -> str | None:
        """Return the stable selected input key."""
        value = self._source.currentData()
        return None if value is None else str(value)

    @property
    def perturbation_band(self) -> str | None:
        """Return the visible rank-band label."""
        return None if self._band.currentData() is None else self._band.currentText()


__all__ = ["ArcFilterControls"]
