"""Electrode Advisor Main Window - PyQt6 GUI.

This is the consolidated GUI that uses the shared electrical model engine
from Tools/src/shared/python/upstream_drift_tools/calculators/electrical/.
"""

from __future__ import annotations

import logging
import os
from typing import Any

# Force matplotlib to use PyQt6 backend
os.environ["QT_API"] = "pyqt6"

import matplotlib as mpl

if os.environ.get("HEADLESS", "false").lower() == "true":
    try:
        mpl.use("Agg")
    except Exception:
        pass
else:
    try:
        mpl.use("QtAgg")
    except Exception:
        mpl.use("Agg")

from PyQt6.QtCore import Qt, QTimer, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QCloseEvent, QColor, QPalette
from PyQt6.QtWidgets import (
    QDoubleSpinBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QLabel,
    QScrollArea,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

# Import the shared engine from Tools
from shared.python.upstream_drift_tools.calculators.electrical import (
    ElectrodeConfig,
    GlassPropertiesInterface,
    ThreePhaseElectricalModelEnhanced,
)

logger = logging.getLogger(__name__)


class ElectrodeAdvisorWidget(QWidget):
    """Main widget for Electrode Advisor using the shared electrical engine.

    This is a consolidated version of the electrode advisor that uses
    the shared engine from upstream_drift_tools.
    """

    # Signals for external communication
    data_updated = pyqtSignal(dict)
    optimization_complete = pyqtSignal(dict)
    glass_properties_requested = pyqtSignal()

    def __init__(
        self,
        config: ElectrodeConfig | None = None,
        glass_interface: GlassPropertiesInterface | None = None,
        parent: QWidget | None = None,
    ) -> None:
        """Initialize the Electrode Advisor widget.

        Args:
            config: Electrode configuration. Creates default if None.
            glass_interface: Glass properties interface. Creates default if None.
            parent: Parent widget.
        """
        super().__init__(parent)

        # Initialize configuration and interfaces
        self.config = config or ElectrodeConfig()
        self.glass_interface = glass_interface or GlassPropertiesInterface()

        # Initialize the shared electrical model
        self.electrical_model = ThreePhaseElectricalModelEnhanced(
            self.config,
            self.glass_interface,
        )

        # Storage for results
        self.calculation_results: dict[str, Any] = {}

        # Input widget references
        self.phase_inputs: dict[str, dict[str, Any]] = {}
        self.depth_inputs: dict[int, QDoubleSpinBox] = {}

        # Initialize UI
        self._init_ui()
        self._apply_styling()

        # Defer initial calculation
        QTimer.singleShot(100, self.calculate_system)

    def _init_ui(self) -> None:
        """Initialize the user interface."""
        main_layout = QVBoxLayout(self)

        # Create main splitter
        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        main_splitter.setHandleWidth(6)
        main_splitter.setChildrenCollapsible(False)

        # Left panel - inputs
        left_panel = self._create_input_panel()
        left_scroll = QScrollArea()
        left_scroll.setWidget(left_panel)
        left_scroll.setWidgetResizable(True)
        main_splitter.addWidget(left_scroll)

        # Center panel - results and status
        center_widget = QWidget()
        center_layout = QVBoxLayout(center_widget)

        # Status panel
        self.status_panel = self._create_status_panel()
        center_layout.addWidget(self.status_panel)

        # Results tabs
        self.results_tabs = QTabWidget()

        # Results tab
        results_panel = self._create_results_panel()
        self.results_tabs.addTab(results_panel, "Results")

        # System Info tab
        system_info_panel = self._create_system_info_panel()
        self.results_tabs.addTab(system_info_panel, "System Info")

        center_layout.addWidget(self.results_tabs)
        main_splitter.addWidget(center_widget)

        # Set splitter proportions
        main_splitter.setSizes([400, 600])

        main_layout.addWidget(main_splitter)

    def _create_input_panel(self) -> QWidget:
        """Create the input parameter panel."""
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # Electrical measurements group
        elec_group = QGroupBox("3-Phase Electrical Measurements")
        elec_layout = QGridLayout(elec_group)

        # Headers
        headers = ["Phase", "Current (A)", "Voltage (V)", "Power (kW)"]
        for i, header in enumerate(headers):
            label = QLabel(header)
            label.setStyleSheet("font-weight: bold;")
            elec_layout.addWidget(label, 0, i)

        # Phase inputs
        phases = ["1-2", "2-3", "3-1"]
        for i, phase in enumerate(phases):
            elec_layout.addWidget(QLabel(phase), i + 1, 0)

            current_input = QDoubleSpinBox()
            current_input.setRange(0, 10000)
            current_input.setValue(300)
            current_input.setDecimals(1)
            current_input.valueChanged.connect(self._on_input_changed)

            voltage_input = QDoubleSpinBox()
            voltage_input.setRange(0, 1000)
            voltage_input.setValue(100)
            voltage_input.setDecimals(1)
            voltage_input.valueChanged.connect(self._on_input_changed)

            power_display = QLabel("30.0")
            power_display.setStyleSheet(
                "background-color: #f0f0f0; border: 1px solid #ccc; padding: 2px;"
            )

            elec_layout.addWidget(current_input, i + 1, 1)
            elec_layout.addWidget(voltage_input, i + 1, 2)
            elec_layout.addWidget(power_display, i + 1, 3)

            self.phase_inputs[phase] = {
                "current": current_input,
                "voltage": voltage_input,
                "power": power_display,
            }

        layout.addWidget(elec_group)

        # Electrode depths group
        depth_group = QGroupBox("Electrode Depths (inches)")
        depth_layout = QFormLayout(depth_group)

        for i in range(3):
            depth_input = QDoubleSpinBox()
            depth_input.setRange(0, 50)
            depth_input.setValue(12.0)
            depth_input.setDecimals(1)
            depth_input.valueChanged.connect(self._on_input_changed)
            depth_layout.addRow(f"Electrode {i + 1}:", depth_input)
            self.depth_inputs[i] = depth_input

        layout.addWidget(depth_group)

        # Physical parameters group
        physical_group = QGroupBox("Physical Parameters")
        physical_layout = QFormLayout(physical_group)

        self.bath_diameter_input = QDoubleSpinBox()
        self.bath_diameter_input.setRange(10, 500)
        self.bath_diameter_input.setValue(120.0)
        self.bath_diameter_input.setDecimals(1)
        self.bath_diameter_input.valueChanged.connect(self._on_input_changed)
        physical_layout.addRow("Bath Diameter (in):", self.bath_diameter_input)

        self.tip_diameter_input = QDoubleSpinBox()
        self.tip_diameter_input.setRange(1, 100)
        self.tip_diameter_input.setValue(24.0)
        self.tip_diameter_input.setDecimals(1)
        self.tip_diameter_input.valueChanged.connect(self._on_input_changed)
        physical_layout.addRow("Tip Diameter (in):", self.tip_diameter_input)

        self.metal_depth_input = QDoubleSpinBox()
        self.metal_depth_input.setRange(0, 20)
        self.metal_depth_input.setValue(2.0)
        self.metal_depth_input.setDecimals(1)
        self.metal_depth_input.valueChanged.connect(self._on_input_changed)
        physical_layout.addRow("Metal Depth (in):", self.metal_depth_input)

        self.bath_temp_input = QDoubleSpinBox()
        self.bath_temp_input.setRange(500, 2000)
        self.bath_temp_input.setValue(1350.0)
        self.bath_temp_input.setDecimals(0)
        self.bath_temp_input.valueChanged.connect(self._on_input_changed)
        physical_layout.addRow("Bath Temperature (C):", self.bath_temp_input)

        layout.addWidget(physical_group)

        layout.addStretch()
        return panel

    def _create_status_panel(self) -> QGroupBox:
        """Create the status panel."""
        status_group = QGroupBox("Status")
        status_layout = QVBoxLayout(status_group)
        status_group.setMaximumHeight(60)

        self.status_label = QLabel("System Ready")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        status_layout.addWidget(self.status_label)

        return status_group

    def _create_results_panel(self) -> QWidget:
        """Create the results display panel."""
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # Power summary
        power_group = QGroupBox("Power Summary")
        power_layout = QGridLayout(power_group)

        self.total_power_label = QLabel("0.0 kW")
        self.total_power_label.setStyleSheet(
            "font-size: 18px; font-weight: bold; color: #2c3e50;"
        )
        power_layout.addWidget(QLabel("Total Power:"), 0, 0)
        power_layout.addWidget(self.total_power_label, 0, 1)

        self.avg_resistance_label = QLabel("0.0 Ohms")
        power_layout.addWidget(QLabel("Avg Resistance:"), 1, 0)
        power_layout.addWidget(self.avg_resistance_label, 1, 1)

        layout.addWidget(power_group)

        # Phase results
        phase_group = QGroupBox("Phase Results")
        phase_layout = QGridLayout(phase_group)

        headers = ["Phase", "Resistance (Ω)", "Current (A)", "Power (kW)"]
        for i, header in enumerate(headers):
            label = QLabel(header)
            label.setStyleSheet("font-weight: bold;")
            phase_layout.addWidget(label, 0, i)

        self.phase_results: dict[str, dict[str, QLabel]] = {}
        phases = ["1-2", "2-3", "3-1"]
        for i, phase in enumerate(phases):
            phase_layout.addWidget(QLabel(phase), i + 1, 0)

            resistance_label = QLabel("--")
            current_label = QLabel("--")
            power_label = QLabel("--")

            phase_layout.addWidget(resistance_label, i + 1, 1)
            phase_layout.addWidget(current_label, i + 1, 2)
            phase_layout.addWidget(power_label, i + 1, 3)

            self.phase_results[phase] = {
                "resistance": resistance_label,
                "current": current_label,
                "power": power_label,
            }

        layout.addWidget(phase_group)

        layout.addStretch()
        return panel

    def _create_system_info_panel(self) -> QWidget:
        """Create the system info panel."""
        panel = QWidget()
        layout = QVBoxLayout(panel)

        info_group = QGroupBox("Engine Information")
        info_layout = QFormLayout(info_group)

        info_layout.addRow("Engine:", QLabel("ThreePhaseElectricalModelEnhanced"))
        info_layout.addRow("Source:", QLabel("Tools/shared/upstream_drift_tools"))
        info_layout.addRow("Version:", QLabel("1.0.0 (Consolidated)"))

        layout.addWidget(info_group)

        config_group = QGroupBox("Current Configuration")
        config_layout = QFormLayout(config_group)

        self.config_display = QLabel("--")
        self.config_display.setWordWrap(True)
        config_layout.addRow(self.config_display)

        layout.addWidget(config_group)

        layout.addStretch()
        return panel

    def _apply_styling(self) -> None:
        """Apply styling to the widget."""
        palette = self.palette()
        palette.setColor(QPalette.ColorRole.Window, QColor(230, 240, 255))
        self.setPalette(palette)

        self.results_tabs.setStyleSheet(
            """
            QTabWidget::pane {
                border: 1px solid #cccccc;
                background: white;
            }
            QTabBar::tab {
                background: #f0f0f0;
                padding: 8px 16px;
                margin-right: 2px;
                border: 1px solid #cccccc;
                border-bottom: none;
            }
            QTabBar::tab:selected {
                background: white;
                border-bottom: 1px solid white;
            }
            QTabBar::tab:hover {
                background: #e0e0e0;
            }
        """
        )

    def _on_input_changed(self) -> None:
        """Handle input parameter changes."""
        self.calculate_system()

    @pyqtSlot()
    def calculate_system(self) -> None:
        """Calculate the electrical system state using the shared engine."""
        try:
            self._update_status("Calculating...", "info")

            # Gather inputs
            import numpy as np

            depths = np.array([self.depth_inputs[i].value() for i in range(3)])
            voltages = np.array(
                [
                    self.phase_inputs[phase]["voltage"].value()
                    for phase in ["1-2", "2-3", "3-1"]
                ]
            )

            # Call the shared engine
            results = self.electrical_model.calculate_system_state(
                depths=depths,
                bath_diameter=self.bath_diameter_input.value(),
                tip_diameter=self.tip_diameter_input.value(),
                metal_depth=self.metal_depth_input.value(),
                k_factors={"K_tt": 1.0, "K_vert": 1.0},
                bath_temperature=self.bath_temp_input.value(),
                voltages=voltages,
                conductive_height=2.0,
            )

            self.calculation_results = results
            self._update_results_display(results)
            self._update_status("Calculation complete", "ok")

            # Emit signal for external listeners
            self.data_updated.emit(results)

        except Exception as e:
            logger.exception("Calculation failed")
            self._update_status(f"Error: {e}", "error")

    def _update_results_display(self, results: dict[str, Any]) -> None:
        """Update the results display with calculation results."""
        # Update power summary
        total_power = results.get("total_power", 0)
        self.total_power_label.setText(f"{total_power:.1f} kW")

        resistances = results.get("resistances", {})
        if resistances:
            avg_resistance = sum(r.get("total", 0) for r in resistances.values()) / len(
                resistances
            )
            self.avg_resistance_label.setText(f"{avg_resistance:.4f} Ω")

        # Update phase results
        currents = results.get("phase_currents", {})
        powers = results.get("phase_powers", {})

        for phase, labels in self.phase_results.items():
            if phase in resistances:
                res = resistances[phase].get("total", 0)
                labels["resistance"].setText(f"{res:.4f}")

            if phase in currents:
                curr = currents[phase]
                labels["current"].setText(f"{curr:.1f}")

            if phase in powers:
                pwr = powers[phase]
                labels["power"].setText(f"{pwr:.2f}")

        # Update power displays in input panel
        for _phase, inputs in self.phase_inputs.items():
            current = inputs["current"].value()
            voltage = inputs["voltage"].value()
            power = current * voltage / 1000  # kW
            inputs["power"].setText(f"{power:.1f}")

        # Update config display
        self.config_display.setText(
            f'Bath: {self.bath_diameter_input.value()}" dia, '
            f'Tip: {self.tip_diameter_input.value()}", '
            f"Temp: {self.bath_temp_input.value()}°C"
        )

    def _update_status(self, message: str, status_type: str = "ok") -> None:
        """Update the status display."""
        self.status_label.setText(message)

        colors = {
            "ok": "#c8ffc8",
            "info": "#e0e0e0",
            "warn": "#ffffb4",
            "error": "#ff9696",
        }
        color = colors.get(status_type, colors["ok"])
        self.status_panel.setStyleSheet(f"background-color: {color}")

    def get_current_state(self) -> dict[str, Any]:
        """Get current state for saving."""
        return {
            "depths": [self.depth_inputs[i].value() for i in range(3)],
            "bath_diameter": self.bath_diameter_input.value(),
            "tip_diameter": self.tip_diameter_input.value(),
            "metal_depth": self.metal_depth_input.value(),
            "bath_temperature": self.bath_temp_input.value(),
            "phase_inputs": {
                phase: {
                    "current": inputs["current"].value(),
                    "voltage": inputs["voltage"].value(),
                }
                for phase, inputs in self.phase_inputs.items()
            },
            "results": self.calculation_results,
        }

    def set_current_state(self, state: dict[str, Any]) -> None:
        """Restore state from saved data."""
        if "depths" in state:
            for i, depth in enumerate(state["depths"]):
                if i in self.depth_inputs:
                    self.depth_inputs[i].setValue(depth)

        if "bath_diameter" in state:
            self.bath_diameter_input.setValue(state["bath_diameter"])
        if "tip_diameter" in state:
            self.tip_diameter_input.setValue(state["tip_diameter"])
        if "metal_depth" in state:
            self.metal_depth_input.setValue(state["metal_depth"])
        if "bath_temperature" in state:
            self.bath_temp_input.setValue(state["bath_temperature"])

        if "phase_inputs" in state:
            for phase, values in state["phase_inputs"].items():
                if phase in self.phase_inputs:
                    self.phase_inputs[phase]["current"].setValue(
                        values.get("current", 300)
                    )
                    self.phase_inputs[phase]["voltage"].setValue(
                        values.get("voltage", 100)
                    )

        self.calculate_system()

    def connect_to_glass_calculator(self, glass_calculator_widget: Any) -> None:
        """Connect to external glass calculator widget."""
        if hasattr(glass_calculator_widget, "glass_properties_calculated"):
            glass_calculator_widget.glass_properties_calculated.connect(
                self._on_glass_properties_updated
            )

    @pyqtSlot(dict)
    def _on_glass_properties_updated(self, properties: dict[str, Any]) -> None:
        """Handle glass properties updates from external calculator."""
        if hasattr(self.glass_interface, "update_properties"):
            self.glass_interface.update_properties(properties)
        self.calculate_system()

    def closeEvent(self, event: QCloseEvent | None) -> None:
        """Handle widget close event."""
        super().closeEvent(event)
