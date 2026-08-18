"""Validated Morris design editors for the PyQt workflow."""

from __future__ import annotations

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QFormLayout, QGroupBox, QSpinBox


class MorrisDesignControls(QGroupBox):
    """Bounded authority design controls with coupled sample requirements."""

    changed = pyqtSignal()

    def __init__(self) -> None:
        super().__init__("Morris Design")
        form = QFormLayout(self)
        self.trajectories = self._editor(2, 5_000, 12, "Trajectories")
        self.levels = self._editor(4, 10_000, 4, "Even grid levels")
        self.levels.setSingleStep(2)
        self.seed = self._editor(0, 2**31 - 1, 0, "Random seed")
        self.minimum_effects = self._editor(2, 5_000, 4, "Minimum valid effects")
        self.workers = self._editor(1, 32, 1, "Authority workers")
        self.trajectories.valueChanged.connect(self.minimum_effects.setMaximum)
        self.minimum_effects.setMaximum(self.trajectories.value())
        for editor in (
            self.trajectories,
            self.levels,
            self.seed,
            self.minimum_effects,
            self.workers,
        ):
            editor.valueChanged.connect(self.changed)
        for label, editor in (
            ("Trajectories", self.trajectories),
            ("Levels", self.levels),
            ("Seed", self.seed),
            ("Minimum Effects", self.minimum_effects),
            ("Workers", self.workers),
        ):
            form.addRow(label, editor)

    @staticmethod
    def _editor(
        minimum: int, maximum: int, value: int, accessible_name: str
    ) -> QSpinBox:
        editor = QSpinBox()
        editor.setRange(minimum, maximum)
        editor.setValue(value)
        editor.setAccessibleName(accessible_name)
        editor.setToolTip(f"{accessible_name} used by the authority-validated design.")
        return editor

    def set_editable(self, editable: bool) -> None:
        """Enable or disable the complete design atomically."""
        for editor in (
            self.trajectories,
            self.levels,
            self.seed,
            self.minimum_effects,
            self.workers,
        ):
            editor.setEnabled(editable)


__all__ = ["MorrisDesignControls"]
