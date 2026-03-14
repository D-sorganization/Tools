"""Persistence mixin for the ElectrodeAdvisorWidget.

Contains save_state, load_state, show_context_menu, copy_results,
and setup_state_management.
"""

from __future__ import annotations

import logging
from datetime import datetime

from PyQt6.QtCore import QPoint

logger = logging.getLogger(__name__)


class PersistenceMixin:
    """Mixin providing state persistence and clipboard for ElectrodeAdvisorWidget."""

    def setup_state_management(self) -> None:
        """Register widgets for state management (stub for future extensibility)"""

    def save_state(self) -> None:
        """Save current state to persistent storage"""
        try:
            state_data = self.get_current_state()  # type: ignore[attr-defined]

            from integrated_process_simulator.utilities.state_manager import (
                StateManager,
            )

            state_manager = StateManager()
            filename = f"{self.calculator_name}_state.json"  # type: ignore[attr-defined]
            state_manager.save_state(filename, state_data)
            logger.info("Electrode Advisor state saved to %s", filename)
        except ImportError as e:
            logger.warning("Warning: Could not save state: %s", e)

    def load_state(self) -> None:
        """Load state from persistent storage"""
        try:
            from integrated_process_simulator.utilities.state_manager import (
                StateManager,
            )

            state_manager = StateManager()
            filename = f"{self.calculator_name}_state.json"  # type: ignore[attr-defined]

            state_data = state_manager.load_state(filename)
            if state_data is not None:
                success = self.restore_state(state_data)  # type: ignore[attr-defined]
                if success:
                    logger.info("Electrode Advisor state loaded from %s", filename)
                else:
                    logger.error("Failed to restore state from %s", filename)
            else:
                logger.info("No saved state found for %s", filename)
        except ImportError as e:
            logger.warning("Warning: Could not load state: %s", e)

    def show_context_menu(self, position: QPoint) -> None:
        """Show context menu for manual state management"""
        assert position is not None, "position must be provided"
        from PyQt6.QtWidgets import QMenu

        menu = QMenu(self)  # type: ignore[arg-type]

        save_action = menu.addAction("Save State")
        if save_action:
            save_action.triggered.connect(lambda checked: self.save_state())

        load_action = menu.addAction("Load State")
        if load_action:
            load_action.triggered.connect(lambda checked: self.load_state())

        menu.addSeparator()

        copy_action = menu.addAction("Copy Results")
        if copy_action:
            copy_action.triggered.connect(lambda checked: self.copy_results())

        menu.exec(self.mapToGlobal(position))  # type: ignore[attr-defined]

    def copy_results(self) -> None:
        """Copy current results to clipboard"""
        try:
            from PyQt6.QtWidgets import QApplication

            if hasattr(self, "calculation_results") and self.calculation_results:  # type: ignore[attr-defined]
                results_text = "Electrode Advisor Results\n"
                results_text += (
                    f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
                )

                if "resistances" in self.calculation_results:  # type: ignore[attr-defined]
                    results_text += "Resistances:\n"
                    for path, resistance in self.calculation_results[  # type: ignore[attr-defined]
                        "resistances"
                    ].items():
                        results_text += f"  {path}: {resistance:.3f} Ω\n"

                if "actual_currents" in self.calculation_results:  # type: ignore[attr-defined]
                    results_text += "\nCurrents:\n"
                    for path, current in self.calculation_results[  # type: ignore[attr-defined]
                        "actual_currents"
                    ].items():
                        results_text += f"  {path}: {current:.3f} A\n"

                clipboard = QApplication.clipboard()
                if clipboard:
                    clipboard.setText(results_text)
                logger.info("Results copied to clipboard")
            else:
                logger.info("No results available to copy.")
        except ImportError as e:
            logger.warning("Warning: Could not copy results: %s", e)
