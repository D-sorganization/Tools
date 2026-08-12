"""One registry-presented editable Morris factor row."""

from __future__ import annotations

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QWidget,
)

from rate_of_closure.application.morris.presentation import MorrisFactorRow
from rate_of_closure.application.morris.request_document import MorrisFactorDraft

_BOUND_LIMIT = 1_000_000_000.0


class MorrisFactorEditor(QWidget):
    """Editable view of one immutable shared factor presentation."""

    changed = pyqtSignal()

    def __init__(self, factor: MorrisFactorRow, parent: QWidget | None = None) -> None:
        if not isinstance(factor, MorrisFactorRow):
            raise TypeError("factor must be a MorrisFactorRow")
        super().__init__(parent)
        self.variable_key = factor.variable_key
        self.enabled = QCheckBox()
        self.enabled.setChecked(factor.enabled and factor.applicable)
        self.enabled.setEnabled(factor.applicable)
        self.enabled.setAccessibleName(f"Enable {factor.label}")
        self.label = QLabel(factor.label)
        self.label.setMinimumWidth(185)
        self.lower_editor = self._bound_editor(
            factor.lower, f"{factor.label} lower bound"
        )
        self.upper_editor = self._bound_editor(
            factor.upper, f"{factor.label} upper bound"
        )
        self.unit = QLabel(factor.unit or "—")
        self.unit.setMinimumWidth(55)
        guidance = factor.guidance
        if factor.applicability:
            guidance = f"{guidance}\nApplies when: {factor.applicability}"
        for widget in (
            self.enabled,
            self.label,
            self.lower_editor,
            self.upper_editor,
            self.unit,
        ):
            widget.setToolTip(guidance)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.enabled)
        layout.addWidget(self.label, stretch=1)
        layout.addWidget(self.lower_editor)
        layout.addWidget(self.upper_editor)
        layout.addWidget(self.unit)
        self.enabled.toggled.connect(self._set_editors_enabled)
        self.enabled.toggled.connect(self.changed)
        self.lower_editor.valueChanged.connect(self.changed)
        self.upper_editor.valueChanged.connect(self.changed)
        self._set_editors_enabled(self.enabled.isChecked())

    @staticmethod
    def _bound_editor(value: float, accessible_name: str) -> QDoubleSpinBox:
        editor = QDoubleSpinBox()
        editor.setRange(-_BOUND_LIMIT, _BOUND_LIMIT)
        editor.setDecimals(6)
        editor.setValue(value)
        editor.setKeyboardTracking(False)
        editor.setAccessibleName(accessible_name)
        return editor

    def _set_editors_enabled(self, enabled: bool) -> None:
        self.lower_editor.setEnabled(enabled)
        self.upper_editor.setEnabled(enabled)

    def draft(self) -> MorrisFactorDraft:
        """Return the exact current editor state."""
        return MorrisFactorDraft(
            self.variable_key,
            self.enabled.isChecked(),
            self.lower_editor.value(),
            self.upper_editor.value(),
        )


__all__ = ["MorrisFactorEditor"]
