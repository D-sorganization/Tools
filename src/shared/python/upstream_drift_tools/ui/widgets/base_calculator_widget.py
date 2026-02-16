"""Base class for all calculator widgets in the fleet."""

from __future__ import annotations

import logging
from typing import Any

from PyQt6.QtWidgets import QMainWindow, QMessageBox, QVBoxLayout, QWidget

from ..mixins.calculator_state_mixin import CalculatorStateMixin

logger = logging.getLogger(__name__)


class BaseCalculatorWidget(QMainWindow, CalculatorStateMixin):
    """Base class for calculator windows providing state management and common UI patterns."""

    def __init__(
        self,
        calculator_name: str,
        window_title: str | None = None,
        min_size: tuple[int, int] = (1000, 700),
        parent: QWidget | None = None,
    ) -> None:
        QMainWindow.__init__(self, parent)
        CalculatorStateMixin.__init__(self, calculator_name)

        self.setWindowTitle(window_title or calculator_name)
        self.setMinimumSize(*min_size)

        # Main layout will be set by subclasses in their _setup_ui method
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)

    def show_error(self, title: str, message: str) -> None:
        """Display an error message box."""
        QMessageBox.critical(self, title, message)

    def show_info(self, title: str, message: str) -> None:
        """Display an information message box."""
        QMessageBox.information(self, title, message)

    def closeEvent(self, event: Any) -> None:
        """Handle window close event with state saving."""
        self.handle_close_event(event)

    def get_calculator_specific_state(self) -> dict[str, Any]:
        """Override to save custom UI state."""
        return {}

    def set_calculator_specific_state(self, state: dict[str, Any]) -> None:
        """Override to restore custom UI state."""
