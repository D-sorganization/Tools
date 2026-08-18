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
from rate_of_closure.application.morris.workspace import MorrisWorkspaceFactorDraft
from rate_of_closure.application.morris.workspace_validation import (
    INVALID_BOUNDS_MESSAGE,
)

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
        self._workspace_lower_text = str(factor.lower)
        self._workspace_upper_text = str(factor.upper)
        self._workspace_validation_error: str | None = None
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
        self.lower_editor.valueChanged.connect(self._lower_changed)
        self.upper_editor.valueChanged.connect(self._upper_changed)
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

    def _lower_changed(self) -> None:
        self._workspace_lower_text = self.lower_editor.cleanText()
        self._workspace_validation_error = self._current_validation_error()
        self.changed.emit()

    def _upper_changed(self) -> None:
        self._workspace_upper_text = self.upper_editor.cleanText()
        self._workspace_validation_error = self._current_validation_error()
        self.changed.emit()

    def _current_validation_error(self) -> str | None:
        return (
            None
            if self.lower_editor.value() < self.upper_editor.value()
            else INVALID_BOUNDS_MESSAGE
        )

    def draft(self) -> MorrisFactorDraft:
        """Return the exact current editor state."""
        if self.enabled.isChecked() and self._workspace_validation_error is not None:
            raise ValueError(
                f"{self.variable_key} cannot be enabled until its bounds are valid"
            )
        return MorrisFactorDraft(
            self.variable_key,
            self.enabled.isChecked(),
            self.lower_editor.value(),
            self.upper_editor.value(),
        )

    def workspace_draft(self) -> MorrisWorkspaceFactorDraft:
        """Return exact editor text plus its current validation state."""
        lower = self._workspace_lower_text
        upper = self._workspace_upper_text
        return MorrisWorkspaceFactorDraft(
            self.variable_key,
            self.enabled.isChecked(),
            lower,
            upper,
            self._workspace_validation_error,
        )

    def load_workspace_draft(self, draft: MorrisWorkspaceFactorDraft) -> None:
        """Restore a validated draft without changing its canonical identity."""
        if draft.variable_key != self.variable_key:
            raise ValueError("workspace factor does not match this editor")
        self.enabled.setChecked(draft.enabled)
        if draft.validation_error is None:
            self.lower_editor.setValue(float(draft.lower))
            self.upper_editor.setValue(float(draft.upper))
        else:
            lower_line = self.lower_editor.lineEdit()
            upper_line = self.upper_editor.lineEdit()
            assert lower_line is not None and upper_line is not None
            lower_line.setText(draft.lower)
            upper_line.setText(draft.upper)
        self._workspace_lower_text = draft.lower
        self._workspace_upper_text = draft.upper
        self._workspace_validation_error = draft.validation_error


__all__ = ["MorrisFactorEditor"]
