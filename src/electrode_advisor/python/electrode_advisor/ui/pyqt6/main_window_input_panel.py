"""InputPanelMixin -- input panel creation for ElectrodeAdvisorWidget.

Builds the left-hand input panel: electrical measurements, physical
parameters, electrode depths, model K-factors, glass integration, and
the "Calculate" action button.
"""

from __future__ import annotations

import logging
from typing import Any

from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QLabel,
    QPushButton,
    QScrollArea,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from ...widget_factory import (
    create_checkbox,
    create_combobox,
    create_double_spinbox,
    create_readonly_lineedit,
)

logger = logging.getLogger(__name__)


class InputPanelMixin:
    """Mixin providing input panel construction methods.

    Expected attributes on the host class (declared for mypy):
    """

    # -- Attributes provided by the host class --
    config: Any
    glass_properties_requested: Any
    _on_input_changed: Any
    _validate_glass_height: Any
    _on_glass_integration_changed: Any
    _connect_to_glass_calculator: Any
    _run_optimization: Any

    def _create_input_panel(self) -> None:
        """Create input parameter panel."""
        self.input_panel = QWidget()
        input_layout = QVBoxLayout(self.input_panel)

        # Scroll area for inputs
        scroll = QScrollArea()
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)

        # Create input sections
        self._create_electrical_measurements_section(scroll_layout)
        self._create_power_factor_section(scroll_layout)
        self._create_physical_parameters_section(scroll_layout)
        self._create_electrode_depths_section(scroll_layout)
        self._create_model_parameters_section(scroll_layout)
        self._create_glass_properties_section(scroll_layout)
        self._create_optimization_button(scroll_layout)

        scroll_layout.addStretch()
        scroll.setWidget(scroll_widget)
        scroll.setWidgetResizable(True)
        input_layout.addWidget(scroll)

    def _create_electrical_measurements_section(
        self, parent_layout: QVBoxLayout
    ) -> None:
        """Create the 3-Phase Electrical Measurements section."""
        elec_group = QGroupBox("3-Phase Electrical Measurements")
        elec_layout = QGridLayout(elec_group)

        # Headers
        headers = [
            "Phase",
            "Current (A)",
            "Voltage (V)",
            "Power (kW)",
            "Resistance (Ω)",
        ]
        for i, header in enumerate(headers):
            label = QLabel(header)
            label.setFont(QFont("Arial", 9, QFont.Weight.Bold))
            elec_layout.addWidget(label, 0, i)

        # Phase inputs
        self.phase_inputs = {}
        phases = ["1-2", "2-3", "3-1"]
        for i, phase in enumerate(phases):
            elec_layout.addWidget(QLabel(phase), i + 1, 0)

            current_input = create_double_spinbox(
                min_value=0.0,
                max_value=10000.0,
                default_value=300.0,
                decimals=1,
                value_changed_callback=self._on_input_changed,
            )

            voltage_input = create_double_spinbox(
                min_value=0.0,
                max_value=1000.0,
                default_value=100.0,
                decimals=1,
                value_changed_callback=self._on_input_changed,
            )

            power_display = create_readonly_lineedit("30.0")
            resistance_display = create_readonly_lineedit("0.0")

            elec_layout.addWidget(current_input, i + 1, 1)
            elec_layout.addWidget(voltage_input, i + 1, 2)
            elec_layout.addWidget(power_display, i + 1, 3)
            elec_layout.addWidget(resistance_display, i + 1, 4)

            self.phase_inputs[phase] = {
                "current": current_input,
                "voltage": voltage_input,
                "power": power_display,
                "resistance": resistance_display,
            }

        parent_layout.addWidget(elec_group)

    def _create_power_factor_section(self, parent_layout: QVBoxLayout) -> None:
        """Create the System Power Factor section."""
        power_factor_group = QGroupBox("System Power Factor")
        power_factor_layout = QFormLayout(power_factor_group)

        self.power_factor_input = create_double_spinbox(
            min_value=0.1,
            max_value=1.0,
            default_value=0.9,
            decimals=3,
            value_changed_callback=self._on_input_changed,
        )
        self.power_factor_input.setSingleStep(0.01)
        power_factor_layout.addRow("Power Factor (cos φ):", self.power_factor_input)

        # Add explanatory note
        pf_note = QLabel(
            "Note: Individual phase powers are resistive (PF=1.0).\n"
            "Power factor applies only to total system power."
        )
        pf_note.setStyleSheet("QLabel { color: #666666; font-size: 8pt; }")
        power_factor_layout.addRow(pf_note)

        # Total power display
        self.total_power_display = create_readonly_lineedit("90.0")
        self.total_power_display.setStyleSheet(
            "background-color: #f0f0f0; font-weight: bold;"
        )
        power_factor_layout.addRow(
            "Total 3-Phase Real Power (kW):", self.total_power_display
        )

        parent_layout.addWidget(power_factor_group)

    def _create_physical_parameters_section(self, parent_layout: QVBoxLayout) -> None:
        """Create the Physical Parameters section."""
        phys_group = QGroupBox("Physical Parameters")
        phys_layout = QFormLayout(phys_group)

        self.bath_diameter_input = create_double_spinbox(
            min_value=10.0,
            max_value=200.0,
            default_value=50.0,
            suffix=" in",
            value_changed_callback=self._on_input_changed,
        )
        phys_layout.addRow("Bath Diameter:", self.bath_diameter_input)

        # Electrode diameter selector
        self.electrode_diameter_combo = create_combobox(
            items=["1.25", "2.0", "3.0"],
            default_item="2.0",
            current_text_changed_callback=self._on_input_changed,
        )
        phys_layout.addRow("Electrode Diameter:", self.electrode_diameter_combo)

        # Metal layer height (0-4 inches)
        self.metal_layer_height_input = create_double_spinbox(
            min_value=0.0,
            max_value=4.0,
            default_value=2.0,
            decimals=1,
            suffix=" in",
            value_changed_callback=self._on_input_changed,
        )
        phys_layout.addRow("Metal Layer Height:", self.metal_layer_height_input)

        # Glass layer height (above metal layer)
        self.glass_layer_height_input = create_double_spinbox(
            min_value=1.0,
            max_value=30.0,
            default_value=15.0,
            suffix=" in",
            value_changed_callback=self._validate_glass_height,
        )
        phys_layout.addRow("Glass Layer Height:", self.glass_layer_height_input)

        # Conductive layer height (for current path visualization)
        self.conductive_layer_height_input = create_double_spinbox(
            min_value=0.5,
            max_value=10.0,
            default_value=2.0,
            decimals=1,
            value_changed_callback=self._on_input_changed,
        )
        phys_layout.addRow(
            "Conductive Layer Height:", self.conductive_layer_height_input
        )

        self.bath_temp_input = create_double_spinbox(
            min_value=800.0,
            max_value=1600.0,
            default_value=1200.0,
            suffix=" °C",
            value_changed_callback=self._on_input_changed,
        )
        phys_layout.addRow("Bath Temperature:", self.bath_temp_input)

        # Refractory layer thickness
        self.refractory_thickness_input = create_double_spinbox(
            min_value=2.0,
            max_value=24.0,
            default_value=8.0,
            decimals=1,
            suffix=" in",
            value_changed_callback=self._on_input_changed,
        )
        phys_layout.addRow("Refractory Thickness:", self.refractory_thickness_input)

        # Spreading factors
        self.vertical_spreading_input = create_double_spinbox(
            min_value=1.0,
            max_value=3.0,
            default_value=1.5,
            decimals=2,
            value_changed_callback=self._on_input_changed,
        )
        self.vertical_spreading_input.setSingleStep(0.1)
        phys_layout.addRow("Vertical Spreading Factor:", self.vertical_spreading_input)

        self.horizontal_spreading_input = create_double_spinbox(
            min_value=1.0,
            max_value=3.0,
            default_value=1.2,
            decimals=2,
            value_changed_callback=self._on_input_changed,
        )
        self.horizontal_spreading_input.setSingleStep(0.1)
        phys_layout.addRow(
            "Horizontal Spreading Factor:", self.horizontal_spreading_input
        )

        parent_layout.addWidget(phys_group)

    def _create_electrode_depths_section(self, parent_layout: QVBoxLayout) -> None:
        """Create the Current Electrode Depths section."""
        pos_group = QGroupBox("Current Electrode Depths")
        pos_layout = QGridLayout(pos_group)

        self.depth_inputs = {}
        for i in range(3):
            pos_layout.addWidget(QLabel(f"Electrode {i + 1}:"), i, 0)

            depth_input = create_double_spinbox(
                min_value=0.0,
                max_value=25.0,
                default_value=12.0,
                decimals=1,
                suffix=" in",
                value_changed_callback=self._on_input_changed,
            )
            pos_layout.addWidget(depth_input, i, 1)

            self.depth_inputs[i] = depth_input

        parent_layout.addWidget(pos_group)

    def _create_model_parameters_section(self, parent_layout: QVBoxLayout) -> None:
        """Create the Advanced Model Parameters section."""
        model_group = QGroupBox("Advanced Model Parameters")
        model_layout = QFormLayout(model_group)

        # K_tt Factor with explanation
        k_tt_label = QLabel("K_tt Factor (Tip-to-Tip):")
        k_tt_label.setToolTip(
            "Controls current conduction between electrode tips through glass.\n"
            "Higher values = more direct conduction, lower resistance.\n"
            "Typical range: 0.05-0.2 for glass melting furnaces."
        )
        self.k_tt_input = create_double_spinbox(
            min_value=0.001,
            max_value=10.0,
            default_value=0.1,
            decimals=3,
            value_changed_callback=self._on_input_changed,
        )
        self.k_tt_input.setToolTip("Tip-to-tip conduction scaling factor")
        model_layout.addRow(k_tt_label, self.k_tt_input)

        # K_vert Factor with explanation
        k_vert_label = QLabel("K_vert Factor (Vertical):")
        k_vert_label.setToolTip(
            "Controls vertical current flow from electrodes to metal layer.\n"
            "Higher values = more vertical conduction through glass depth.\n"
            "Typical range: 0.08-0.15 for standard electrode configurations."
        )
        self.k_vert_input = create_double_spinbox(
            min_value=0.001,
            max_value=10.0,
            default_value=0.1,
            decimals=3,
            value_changed_callback=self._on_input_changed,
        )
        self.k_vert_input.setToolTip("Vertical conduction scaling factor")
        model_layout.addRow(k_vert_label, self.k_vert_input)

        parent_layout.addWidget(model_group)

    def _create_glass_properties_section(self, parent_layout: QVBoxLayout) -> None:
        """Create the Glass Properties Integration section."""
        # Glass properties button
        glass_btn = QPushButton("Configure Glass Properties")
        glass_btn.clicked.connect(self.glass_properties_requested.emit)
        parent_layout.addWidget(glass_btn)

        # Glass Properties Integration
        glass_integration_group = QGroupBox("Glass Properties Integration")
        glass_integration_layout = QVBoxLayout(glass_integration_group)

        # Checkbox to enable glass properties integration
        self.glass_integration_checkbox = create_checkbox(
            text="Enable Glass Properties Integration",
            checked=False,
            state_changed_callback=self._on_glass_integration_changed,
        )
        self.glass_integration_checkbox.setToolTip(
            "Connect to external glass properties calculator"
            " for accurate conductivity calculations"
        )
        glass_integration_layout.addWidget(self.glass_integration_checkbox)

        # Status label for glass integration
        self.glass_integration_status = QLabel(
            "Glass Properties Calculator: Not Available"
        )
        self.glass_integration_status.setStyleSheet(
            "QLabel { color: #666666; font-size: 9pt; }"
        )
        glass_integration_layout.addWidget(self.glass_integration_status)

        # Connect button
        self.connect_glass_calculator_btn = QPushButton("Connect to Glass Calculator")
        self.connect_glass_calculator_btn.clicked.connect(
            self._connect_to_glass_calculator
        )
        self.connect_glass_calculator_btn.setEnabled(False)
        glass_integration_layout.addWidget(self.connect_glass_calculator_btn)

        # Properties display
        self.glass_properties_display = QTextEdit()
        self.glass_properties_display.setMaximumHeight(100)
        self.glass_properties_display.setReadOnly(True)
        self.glass_properties_display.setPlaceholderText(
            "Glass properties will be displayed here when connected..."
        )
        self.glass_properties_display.setStyleSheet(
            "QTextEdit { background-color: #f8f8f8; font-size: 8pt; }"
        )
        glass_integration_layout.addWidget(self.glass_properties_display)

        parent_layout.addWidget(glass_integration_group)

    def _create_optimization_button(self, parent_layout: QVBoxLayout) -> None:
        """Create the optimization button."""
        optimize_btn = QPushButton("Calculate Electrode Advancement")
        optimize_btn.clicked.connect(self._run_optimization)
        optimize_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-weight: bold;
                padding: 10px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        parent_layout.addWidget(optimize_btn)
