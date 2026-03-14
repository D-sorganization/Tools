"""IOMixin -- state management, export, and clipboard operations."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from PyQt6.QtCore import QPoint, QTimer
from PyQt6.QtWidgets import QDoubleSpinBox

logger = logging.getLogger(__name__)


class IOMixin:
    """Mixin providing state save/load, export, and clipboard operations."""

    # -- Attributes provided by the host class (declared for mypy) --
    bath_diameter_input: Any
    bath_temp_input: Any
    calculation_results: Any
    calculator_name: Any
    config: Any
    depth_inputs: Any
    electrode_diameter_combo: Any
    electrode_fig: Any
    glass_layer_height_input: Any
    mapToGlobal: Any
    metal_layer_height_input: Any
    phase_inputs: Any
    _calculate_system: Any
    _update_status: Any

    def setup_state_management(self) -> None:
        """Register widgets for state management (stub for future extensibility)"""

    def save_state(self) -> None:
        """Save current state to persistent storage"""
        try:
            state_data = self.get_current_state()
            # Save to file using the calculator name as identifier

            from integrated_process_simulator.utilities.state_manager import (
                StateManager,
            )

            state_manager = StateManager()
            filename = f"{self.calculator_name}_state.json"
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
            filename = f"{self.calculator_name}_state.json"

            # Check if state file exists
            state_data = state_manager.load_state(filename)
            if state_data is not None:
                success = self.restore_state(state_data)
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

        menu = QMenu(self)  # type: ignore[call-overload]

        # State management actions
        save_action = menu.addAction("Save State")
        if save_action:
            save_action.triggered.connect(lambda checked: self.save_state())

        load_action = menu.addAction("Load State")
        if load_action:
            load_action.triggered.connect(lambda checked: self.load_state())

        menu.addSeparator()

        # Copy actions
        copy_action = menu.addAction("Copy Results")
        if copy_action:
            copy_action.triggered.connect(lambda checked: self.copy_results())

        menu.exec(self.mapToGlobal(position))

    def copy_results(self) -> None:
        """Copy current results to clipboard"""
        try:
            from PyQt6.QtWidgets import QApplication

            # Get current calculation results
            if hasattr(self, "calculation_results") and self.calculation_results:
                results_text = "Electrode Advisor Results\n"
                results_text += (
                    f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
                )

                # Add resistance data
                if "resistances" in self.calculation_results:
                    results_text += "Resistances:\n"
                    for path, resistance in self.calculation_results[
                        "resistances"
                    ].items():
                        results_text += f"  {path}: {resistance:.3f} Ω\n"

                # Add current data
                if "actual_currents" in self.calculation_results:
                    results_text += "\nCurrents:\n"
                    for path, current in self.calculation_results[
                        "actual_currents"
                    ].items():
                        results_text += f"  {path}: {current:.3f} A\n"

                # Copy to clipboard
                clipboard = QApplication.clipboard()
                if clipboard:
                    clipboard.setText(results_text)
                logger.info("Results copied to clipboard")
            else:
                logger.info("No results available to copy.")
        except ImportError as e:
            logger.warning("Warning: Could not copy results: %s", e)

    def get_current_state(self) -> dict[str, Any]:
        """Get current state for state management"""
        try:
            return {
                "calculator_name": "ElectrodeAdvisor",
                "timestamp": datetime.now().isoformat(),
                "config": {
                    "bath_radius": self.bath_diameter_input.value() / 2.0,
                    "electrode_radius": float(
                        self.electrode_diameter_combo.currentText()
                    )
                    / 2.0,
                    "metal_height": self.config.metal_layer_height,
                    "glass_height": self.config.glass_depth,
                    "bath_temperature": self.config.bath_temperature_base,
                    "vertical_spreading": self.config.vertical_spreading_factor,
                    "horizontal_spreading": self.config.horizontal_spreading_factor,
                },
                "electrode_depths": {
                    "E1": (
                        getattr(self, "depth_inputs", [None, None, None])[0].value()
                        if hasattr(self, "depth_inputs") and len(self.depth_inputs) > 0
                        else 0
                    ),
                    "E2": (
                        getattr(self, "depth_inputs", [None, None, None])[1].value()
                        if hasattr(self, "depth_inputs") and len(self.depth_inputs) > 1
                        else 0
                    ),
                    "E3": (
                        getattr(self, "depth_inputs", [None, None, None])[2].value()
                        if hasattr(self, "depth_inputs") and len(self.depth_inputs) > 2
                        else 0
                    ),
                },
                "electrical_measurements": {
                    "1-2": {
                        "current": (
                            getattr(self, "phase_inputs", {})
                            .get("1-2", {})
                            .get("current", None)
                            .value()
                            if hasattr(self, "phase_inputs")
                            and "1-2" in self.phase_inputs
                            else 0
                        ),
                        "voltage": (
                            getattr(self, "phase_inputs", {})
                            .get("1-2", {})
                            .get("voltage", None)
                            .value()
                            if hasattr(self, "phase_inputs")
                            and "1-2" in self.phase_inputs
                            else 0
                        ),
                    },
                    "2-3": {
                        "current": (
                            getattr(self, "phase_inputs", {})
                            .get("2-3", {})
                            .get("current", None)
                            .value()
                            if hasattr(self, "phase_inputs")
                            and "2-3" in self.phase_inputs
                            else 0
                        ),
                        "voltage": (
                            getattr(self, "phase_inputs", {})
                            .get("2-3", {})
                            .get("voltage", None)
                            .value()
                            if hasattr(self, "phase_inputs")
                            and "2-3" in self.phase_inputs
                            else 0
                        ),
                    },
                    "3-1": {
                        "current": (
                            getattr(self, "phase_inputs", {})
                            .get("3-1", {})
                            .get("current", None)
                            .value()
                            if hasattr(self, "phase_inputs")
                            and "3-1" in self.phase_inputs
                            else 0
                        ),
                        "voltage": (
                            getattr(self, "phase_inputs", {})
                            .get("3-1", {})
                            .get("voltage", None)
                            .value()
                            if hasattr(self, "phase_inputs")
                            and "3-1" in self.phase_inputs
                            else 0
                        ),
                    },
                },
                "results": self.calculation_results,
            }
        except (ValueError, ZeroDivisionError, OverflowError, TypeError) as e:
            logger.exception("Error getting current state: %s", e)
            return {"calculator_name": "ElectrodeAdvisor", "error": str(e)}

    def restore_state(self, state_data: dict[str, Any]) -> bool:
        """Restore state from saved data"""
        try:
            if state_data.get("calculator_name") != "ElectrodeAdvisor":
                logger.info("State data mismatch for ElectrodeAdvisor")
                return False

            # Restore configuration
            if "config" in state_data:
                config_data = state_data["config"]
                if "bath_radius" in config_data:
                    self.bath_diameter_input.setValue(config_data["bath_radius"] * 2.0)
                if "electrode_radius" in config_data:
                    self.electrode_diameter_combo.setCurrentText(
                        str(config_data["electrode_radius"] * 2.0)
                    )
                if "metal_height" in config_data:
                    self.config.metal_layer_height = config_data["metal_height"]
                if "glass_height" in config_data:
                    self.config.glass_depth = config_data["glass_height"]
                if "bath_temperature" in config_data:
                    self.config.bath_temperature_base = config_data["bath_temperature"]
                if "vertical_spreading" in config_data:
                    self.config.vertical_spreading_factor = config_data[
                        "vertical_spreading"
                    ]
                if "horizontal_spreading" in config_data:
                    self.config.horizontal_spreading_factor = config_data[
                        "horizontal_spreading"
                    ]

            # Restore electrode depths
            if "electrode_depths" in state_data and hasattr(self, "depth_inputs"):
                depths_data = state_data["electrode_depths"]
                if "E1" in depths_data and len(self.depth_inputs) > 0:
                    self.depth_inputs[0].setValue(depths_data["E1"])
                if "E2" in depths_data and len(self.depth_inputs) > 1:
                    self.depth_inputs[1].setValue(depths_data["E2"])
                if "E3" in depths_data and len(self.depth_inputs) > 2:
                    self.depth_inputs[2].setValue(depths_data["E3"])

            # Restore electrical measurements
            if "electrical_measurements" in state_data and hasattr(
                self, "phase_inputs"
            ):
                measurements_data = state_data["electrical_measurements"]
                for phase in ["1-2", "2-3", "3-1"]:
                    if phase in measurements_data and phase in self.phase_inputs:
                        if "current" in measurements_data[phase]:
                            cast(
                                QDoubleSpinBox, self.phase_inputs[phase]["current"]
                            ).setValue(measurements_data[phase]["current"])
                        if "voltage" in measurements_data[phase]:
                            cast(
                                QDoubleSpinBox, self.phase_inputs[phase]["voltage"]
                            ).setValue(measurements_data[phase]["voltage"])

            # Trigger recalculation
            QTimer.singleShot(100, self._calculate_system)

            logger.info("State restored for ElectrodeAdvisor")
            return True

        except (RuntimeError, AttributeError) as e:
            logger.exception("Error restoring state for ElectrodeAdvisor: %s", e)
            return False

    def set_electrode_depths(self, depths: list[float]) -> None:
        """Set electrode depths programmatically"""
        for i, depth in enumerate(depths[:3]):
            self.depth_inputs[i].setValue(depth)

    def set_electrical_measurements(
        self, currents: list[float], voltages: list[float]
    ) -> None:
        """Set electrical measurements programmatically"""
        assert currents is not None, "currents must be provided"
        phases = ["1-2", "2-3", "3-1"]
        for i, phase in enumerate(phases[:3]):
            if phase in self.phase_inputs:
                cast(QDoubleSpinBox, self.phase_inputs[phase]["current"]).setValue(
                    currents[i]
                )
                cast(QDoubleSpinBox, self.phase_inputs[phase]["voltage"]).setValue(
                    voltages[i]
                )

    def export_results(self, filename: str) -> bool:
        """Export calculation results to file"""
        try:
            import json

            with open(filename, "w") as f:
                # Prepare exportable data
                export_data = {
                    "timestamp": datetime.now().isoformat(),
                    "configuration": {
                        "bath_diameter": self.bath_diameter_input.value(),
                        "electrode_diameter": float(
                            self.electrode_diameter_combo.currentText()
                        ),
                        "metal_height": self.metal_layer_height_input.value(),
                        "glass_height": self.glass_layer_height_input.value(),
                        "bath_temperature": self.bath_temp_input.value(),
                        "vertical_spreading": self.config.vertical_spreading_factor,
                        "horizontal_spreading": self.config.horizontal_spreading_factor,
                    },
                    "electrode_depths": {
                        "E1": self.depth_inputs[0].value(),
                        "E2": self.depth_inputs[1].value(),
                        "E3": self.depth_inputs[2].value(),
                    },
                    "results": self.calculation_results,
                }
                json.dump(export_data, f, indent=2)
                logger.info("Results exported to %s", filename)
                return True
        except (PermissionError, OSError) as e:
            logger.exception("Error exporting results: %s", e)
            return False

    def _export_3d_plot(self) -> None:
        """Export the 3D plot to an image file"""
        try:
            from PyQt6.QtWidgets import QFileDialog

            filename, _ = QFileDialog.getSaveFileName(
                self,  # type: ignore[arg-type]
                "Export 3D Plot",
                "electrode_3d_plot.png",
                "PNG files (*.png);;PDF files (*.pdf);;SVG files (*.svg);;JPG files (*.jpg)",
            )

            if filename and self.electrode_fig is not None:
                # Save the main 3D plot
                self.electrode_fig.savefig(filename, dpi=300, bbox_inches="tight")
                self._update_status(f"3D plot exported to {filename}", "ok")

        except ImportError as e:
            self._update_status(f"Error exporting 3D plot: {e!s}", "error")
            logger.exception("Export error: %s", e)

    def _export_charts(self) -> None:
        """Export the current and power charts to image files"""
        try:
            from PyQt6.QtWidgets import QFileDialog

            filename, _ = QFileDialog.getSaveFileName(
                self,  # type: ignore[arg-type]
                "Export Charts",
                "electrode_charts.png",
                "PNG files (*.png);;PDF files (*.pdf);;SVG files (*.svg);;JPG files (*.jpg)",
            )

            if filename:
                # Create a combined figure with both charts
                # Performance: Use Figure() directly instead of plt.figure() to avoid
                # pyplot state machine memory leaks in long-running sessions
                combined_fig = Figure(
                    figsize=(6, 3)
                )  # REDUCED: Smaller figure size to allow flexible sizing

                # Copy current chart
                ax1 = combined_fig.add_subplot(1, 2, 1)

                # Recreate current distribution chart
                phase_currents = [
                    self.phase_inputs["1-2"]["current"].value(),
                    self.phase_inputs["2-3"]["current"].value(),
                    self.phase_inputs["3-1"]["current"].value(),
                ]
                phases = ["1-2", "2-3", "3-1"]
                colors = ["#FF4444", "#44FF44", "#4444FF"]
                x = np.arange(len(phases))
                width = 0.35

                ax1.bar(
                    x - width / 2,
                    phase_currents,
                    width,
                    label="Phase Current",
                    color=colors,
                    alpha=0.8,
                )
                line_currents = [current * 0.8 for current in phase_currents]
                ax1.bar(
                    x + width / 2,
                    line_currents,
                    width,
                    label="Line Current",
                    color=colors,
                    alpha=0.5,
                )

                ax1.set_xlabel("Phase")
                ax1.set_ylabel("Current (A)")
                ax1.set_title("Current Distribution")
                ax1.set_xticks(x)
                ax1.set_xticklabels(phases)
                ax1.legend(loc="upper right")
                ax1.grid(True, alpha=0.3)

                # Copy power chart
                ax2 = combined_fig.add_subplot(1, 2, 2)

                # Recreate power distribution chart
                powers = []
                colors2 = ["#FF8C00", "#32CD32", "#1E90FF"]
                for phase in phases:
                    current = self.phase_inputs[phase]["current"].value()
                    voltage = self.phase_inputs[phase]["voltage"].value()
                    power = current * voltage / 1000
                    powers.append(power)

                ax2.bar(phases, powers, color=colors2, alpha=0.7)
                ax2.set_xlabel("Phase")
                ax2.set_ylabel("Power (kW)")
                ax2.set_title("Power Distribution")
                ax2.grid(True, alpha=0.3)

                combined_fig.tight_layout()
                combined_fig.savefig(filename, dpi=300, bbox_inches="tight")
                plt.close(combined_fig)

                self._update_status(f"Charts exported to {filename}", "ok")

        except ImportError as e:
            self._update_status(f"Error exporting charts: {e!s}", "error")
            logger.exception("Export error: %s", e)
