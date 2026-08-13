"""Noise-row editor widget for the Variation tab (#4120 V3).

One row per varied variable: registry-driven variable picker, sampling
distribution, unit-aware scale, and optional truncation (clipping)
bounds. Split out of :mod:`.variation_tab` to keep both modules inside
the 500-line budget.
"""

from __future__ import annotations

from PyQt6.QtWidgets import (
    QAbstractSpinBox,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QPushButton,
    QWidget,
)

from rate_of_closure.ui.pyqt6.variation_results import short_label
from shared.python.swing_sim.variation import (
    DISTRIBUTIONS,
    NoiseSpec,
    keys_for_mode,
    variable_registry,
)

__all__ = ["NoiseRow", "make_spin"]


def make_spin(lo: float, hi: float, value: float, decimals: int) -> QDoubleSpinBox:
    """A no-arrow, typed QDoubleSpinBox in the app's input style."""
    spin = QDoubleSpinBox()
    spin.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
    spin.setKeyboardTracking(False)
    spin.setDecimals(decimals)
    spin.setRange(lo, hi)
    spin.setValue(value)
    return spin


class NoiseRow(QWidget):
    """One noise spec: variable, distribution, scale, optional clipping."""

    def __init__(self, mode: str, on_remove) -> None:  # type: ignore[no-untyped-def]
        super().__init__()
        self._loaded_spec: NoiseSpec | None = None
        self._loaded_editor_state: tuple[object, ...] | None = None
        self.variable = QComboBox()
        self.variable.setToolTip(
            "Which registry variable varies run to run. Grouped by "
            "namespaced category (shared across the desktop and web tools)."
        )
        self.distribution = QComboBox()
        self.distribution.addItems(list(DISTRIBUTIONS))
        self.distribution.setToolTip(
            "Sampling distribution about the base value: normal (scale = "
            "standard deviation), uniform or triangular (scale = half-width, "
            "triangular peaks at the base value)."
        )
        self.scale = make_spin(1e-6, 1e6, 1.0, 4)
        self.clip = QCheckBox("Clip")
        self.clip.setToolTip(
            "Truncate samples into [min, max] by clipping — samples never "
            "leave the bounds and the draw count stays deterministic."
        )
        self.clip_low = make_spin(-1e6, 1e6, -5.0, 3)
        self.clip_high = make_spin(-1e6, 1e6, 5.0, 3)
        for widget in (self.clip_low, self.clip_high):
            widget.setEnabled(False)
            widget.setToolTip("Truncation bound (absolute value, same unit).")
        self.clip.toggled.connect(self.clip_low.setEnabled)
        self.clip.toggled.connect(self.clip_high.setEnabled)
        remove = QPushButton("✕")
        remove.setFixedWidth(28)
        remove.setToolTip("Remove this noise row.")
        remove.clicked.connect(lambda: on_remove(self))

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.variable, stretch=3)
        layout.addWidget(self.distribution, stretch=1)
        layout.addWidget(self.scale, stretch=1)
        layout.addWidget(self.clip)
        layout.addWidget(self.clip_low, stretch=1)
        layout.addWidget(self.clip_high, stretch=1)
        layout.addWidget(remove)

        self.variable.currentIndexChanged.connect(self._on_variable_changed)
        self.set_mode(mode)

    def set_mode(self, mode: str) -> None:
        """Repopulate the variable picker for a pipeline mode."""
        self._loaded_spec = None
        self._loaded_editor_state = None
        current = self.key()
        self.variable.blockSignals(True)
        self.variable.clear()
        for key in keys_for_mode(mode):
            self.variable.addItem(short_label(key), key)
        self.variable.blockSignals(False)
        index = self.variable.findData(current)
        self.variable.setCurrentIndex(max(index, 0))
        self._on_variable_changed()

    def key(self) -> str | None:
        """The selected registry key (None while empty)."""
        data = self.variable.currentData()
        return None if data is None else str(data)

    def _on_variable_changed(self, *_args: object) -> None:
        key = self.key()
        definition = None if key is None else variable_registry().get(key)
        if definition is None:
            return
        suffix = f" {definition.unit}" if definition.unit else ""
        for widget in (self.scale, self.clip_low, self.clip_high):
            widget.setSuffix(suffix)
        self.scale.setValue(definition.typical_scale)
        self.scale.setToolTip(
            f"Noise scale in {definition.unit or 'unitless'} about the base "
            f"value. {definition.guidance}"
        )
        self.variable.setToolTip(f"{definition.key} — {definition.guidance}")
        half = 5.0 * definition.typical_scale
        self.clip_low.setValue(definition.default - half)
        self.clip_high.setValue(definition.default + half)

    def to_spec(self) -> NoiseSpec:
        """Build the DbC-validated NoiseSpec described by this row."""
        key = self.key()
        if key is None:
            raise ValueError("noise row has no variable selected")
        state = self._editor_state()
        loaded = self._loaded_spec
        if loaded is not None and state == self._loaded_editor_state:
            return loaded
        lower, upper = self._edited_bounds(loaded, state)
        return NoiseSpec(
            variable_key=key,
            distribution=self.distribution.currentText(),
            scale=self.scale.value(),
            lower=lower,
            upper=upper,
            spec_id=None if loaded is None else loaded.spec_id,
            time_window_s=None if loaded is None else loaded.time_window_s,
            point_ids=() if loaded is None else loaded.point_ids,
        )

    def load_spec(self, spec: NoiseSpec) -> None:
        """Drive the editors from a NoiseSpec (plan load)."""
        index = self.variable.findData(spec.variable_key)
        if index < 0:
            raise ValueError(
                f"noise variable is not representable in this mode: {spec.variable_key}"
            )
        self.variable.setCurrentIndex(index)
        self.distribution.setCurrentText(spec.distribution)
        self.scale.setValue(spec.scale)
        self.clip.setChecked(spec.lower is not None or spec.upper is not None)
        if spec.lower is not None:
            self.clip_low.setValue(spec.lower)
        if spec.upper is not None:
            self.clip_high.setValue(spec.upper)
        self._loaded_spec = spec
        self._loaded_editor_state = self._editor_state()

    def accepts_numeric_range(self, spec: NoiseSpec) -> bool:
        """Return whether the row controls can display a spec without clamping."""
        values: list[tuple[QDoubleSpinBox, float]] = [(self.scale, spec.scale)]
        if spec.lower is not None:
            values.append((self.clip_low, spec.lower))
        if spec.upper is not None:
            values.append((self.clip_high, spec.upper))
        return all(spin.minimum() <= value <= spin.maximum() for spin, value in values)

    def _editor_state(self) -> tuple[object, ...]:
        """Return the visible fields used to detect intentional edits."""
        return (
            self.key(),
            self.distribution.currentText(),
            self.scale.value(),
            self.clip.isChecked(),
            self.clip_low.value(),
            self.clip_high.value(),
        )

    def _edited_bounds(
        self, loaded: NoiseSpec | None, state: tuple[object, ...]
    ) -> tuple[float | None, float | None]:
        """Merge edited bounds while retaining unrepresented one-sided authority."""
        if not self.clip.isChecked():
            return None, None
        lower: float | None = self.clip_low.value()
        upper: float | None = self.clip_high.value()
        prior = self._loaded_editor_state
        if loaded is not None and prior is not None:
            if loaded.lower is None and state[4] == prior[4]:
                lower = None
            if loaded.upper is None and state[5] == prior[5]:
                upper = None
        return lower, upper
