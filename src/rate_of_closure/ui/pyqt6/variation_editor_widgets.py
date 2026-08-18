"""Small shared editor factories for the PyQt variation workflow."""

from PyQt6.QtWidgets import QAbstractSpinBox, QDoubleSpinBox

__all__ = ["make_spin"]


def make_spin(lo: float, hi: float, value: float, decimals: int) -> QDoubleSpinBox:
    """Return a no-arrow, typed numeric editor in the app input style."""
    spin = QDoubleSpinBox()
    spin.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
    spin.setKeyboardTracking(False)
    spin.setDecimals(decimals)
    spin.setRange(lo, hi)
    spin.setValue(value)
    return spin
