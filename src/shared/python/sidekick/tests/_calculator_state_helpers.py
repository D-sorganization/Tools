"""Shared test doubles for the CalculatorStateMixin test modules."""

from __future__ import annotations

from PyQt6.QtWidgets import QWidget
from sidekick.ui.mixins.calculator_state_mixin import CalculatorStateMixin


class MockCalculator(QWidget, CalculatorStateMixin):
    def __init__(self, name=None) -> None:
        QWidget.__init__(self)
        CalculatorStateMixin.__init__(self, name)
