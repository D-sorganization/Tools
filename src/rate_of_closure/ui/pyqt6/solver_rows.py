"""Row widgets for the Solver panel (split out for the 500-LOC budget).

:class:`GoalRow` (enable + target + weight) and :class:`VariableRow`
(Optimize-with-bounds / Fix radio pair) — extracted verbatim from
``solver_panel.py`` when the H7b target panel joined it (#4125).
"""

from __future__ import annotations

from PyQt6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QRadioButton,
    QWidget,
)

from rate_of_closure.ui.pyqt6.solver_specs import GoalSpec, VariableSpec
from rate_of_closure.ui.pyqt6.target_panel import make_spin

__all__ = ["GoalRow", "VariableRow"]


class GoalRow(QWidget):
    """One goal quantity: enable checkbox + target + weight entries."""

    def __init__(self, spec: GoalSpec) -> None:
        super().__init__()
        self.spec = spec
        self.enabled = QCheckBox(spec.label)
        self.enabled.setToolTip(spec.guidance)
        self.target = make_spin(
            spec.spin_range[0], spec.spin_range[1], spec.default_target, 1, spec.unit
        )
        self.target.setToolTip(spec.guidance)
        self.weight = make_spin(0.01, 100.0, 1.0, 2, "")
        self.weight.setToolTip(
            "Suggested range: 0.1-10 relative weight (1 default); larger "
            "weights make the optimizer trade other goals away to hit "
            "this one. Source: shared swing_sim solver tuning "
            "documentation (launch-monitor-resolution residual scales)."
        )
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.enabled, stretch=1)
        layout.addWidget(self.target)
        layout.addWidget(QLabel("w"))
        layout.addWidget(self.weight)
        for widget in (self.target, self.weight):
            widget.setEnabled(False)
        self.enabled.toggled.connect(self.target.setEnabled)
        self.enabled.toggled.connect(self.weight.setEnabled)


class VariableRow(QWidget):
    """One variable: radio Optimize (min/max bounds) | Fix (value)."""

    def __init__(self, spec: VariableSpec) -> None:
        super().__init__()
        self.spec = spec
        lo, hi = spec.spin_range
        self.optimize = QRadioButton("Optimize")
        self.fix = QRadioButton("Fix")
        self._group = QButtonGroup(self)
        self._group.addButton(self.optimize)
        self._group.addButton(self.fix)
        # NOTE: not named "lower"/"raise_" — those are QWidget methods.
        self.low = make_spin(lo, hi, spec.default_bounds[0], spec.decimals, spec.unit)
        self.high = make_spin(lo, hi, spec.default_bounds[1], spec.decimals, spec.unit)
        self.fixed_value = make_spin(
            lo, hi, spec.default_value, spec.decimals, spec.unit
        )
        label = QLabel(spec.label)
        for widget in (label, self.optimize, self.fix, self.low, self.high):
            widget.setToolTip(spec.guidance)
        self.fixed_value.setToolTip(spec.guidance)

        layout = QGridLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(label, 0, 0, 1, 4)
        layout.addWidget(self.optimize, 1, 0)
        layout.addWidget(self.low, 1, 1)
        layout.addWidget(self.high, 1, 2)
        layout.addWidget(self.fix, 1, 3)
        layout.addWidget(self.fixed_value, 1, 4)
        self.optimize.toggled.connect(self._sync_enabled)
        self.fix.setChecked(True)
        self._sync_enabled()

    def _sync_enabled(self, *_args: object) -> None:
        free = self.optimize.isChecked()
        self.low.setEnabled(free)
        self.high.setEnabled(free)
        self.fixed_value.setEnabled(not free)
